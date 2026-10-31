"""No-update acceptance using the real workflow and approved shared vLLM layout."""

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

from opd_pilot.context_audit import manifest, require_valid_response, validate_config
from opd_pilot.integration import plain


class GuardedStudent:
    def __init__(self, model, rows, prompt_limit, teacher_limit):
        self.model, self.rows = model, rows
        self.prompt_limit, self.teacher_limit = prompt_limit, teacher_limit
        self.evaluator_backend = None

    async def chat_async(self, messages, **kwargs):
        responses = await self.model.chat_async(messages, **kwargs)
        for response in responses:
            self.rows.append(
                dict(
                    messages=plain(messages),
                    tokens=plain(response.tokens),
                    prompt_length=response.prompt_length,
                    response=response.response_text,
                    truncate_status=response.truncate_status,
                    action_mask=plain(response.action_mask),
                )
            )
            require_valid_response(response)
            if self.evaluator_backend is not None:
                expected = self.evaluator_backend.encode_messages(
                    messages, self.model.config.enable_thinking
                )
                if expected != plain(response.tokens[: response.prompt_length]):
                    raise ValueError("Training/evaluator prompt token IDs differ")
                self.rows[-1]["evaluation_prompt_ids_equal"] = True
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("Expected one independent user message")
            if (
                response.prompt_length > self.prompt_limit
                or len(response.tokens) >= self.teacher_limit
            ):
                raise ValueError("Student or teacher input budget exceeded")
        return responses


class GuardedTeacher:
    def __init__(self, model, rows):
        self.model, self.rows, self.index = model, rows, 0

    def get_openai_async_client(self):
        return self.model.get_openai_async_client()

    async def logprobs_async(self, tokens, **kwargs):
        row = self.rows[self.index]
        if tokens != row["tokens"]:
            raise ValueError("Teacher received different token IDs")
        scores = await self.model.logprobs_async(tokens, **kwargs)
        if len(scores) != len(tokens) - 1:
            raise ValueError("Teacher scoring contract mismatch")
        p = row["prompt_length"]
        row.update(
            teacher_input_ids_equal=True,
            teacher_full_length=len(scores),
            response_length=len(tokens) - p,
            slice_start=p - 1,
            teacher_response_logprobs=plain(scores[p - 1 :]),
        )
        self.index += 1
        return scores


async def prepare_wrappers(actors, wrapper_factory):
    """Match Explorer: prepare engines before querying their API endpoints."""
    await asyncio.gather(*(actor.prepare.remote() for actor in actors))
    wrappers = [wrapper_factory(actor) for actor in actors]
    for wrapper in wrappers:
        await wrapper.prepare()
    return wrappers


async def exercise(config, output, report):
    from trinity.common.models.model import ModelWrapper
    from trinity.common.models.vllm_model import vLLMRolloutModel
    from trinity.common.shared_inference import create_shared_inference
    from trinity.common.workflows import Task
    from trinity.common.workflows.envs.TCOD.alfworld.OPD_workflow import (
        OnPolicyDistillVerlAgentAlfworldWorkflow,
    )

    students, teachers = create_shared_inference(config, vLLMRolloutModel)
    student, teacher = await prepare_wrappers([students[0], teachers[0][0]], ModelWrapper)
    taskset = config.buffer.explorer_input.tasksets[0]
    tasks = [
        json.loads(line) for line in Path(taskset.path).read_text().splitlines() if line.strip()
    ][:2]
    if len(tasks) != 2:
        raise ValueError("Need two existing ALFWorld tasks")
    report["planned_tasks"] = [row.get("task_id", row["game_file"]) for row in tasks]
    for raw in tasks:
        entry = dict(task_id=raw.get("task_id", raw["game_file"]), turns=[], status="running")
        report["episodes"].append(entry)
        task = Task(
            raw_task=raw,
            format_args=taskset.format,
            rollout_args=taskset.rollout_args,
            workflow_args=taskset.workflow_args,
        )
        guarded = GuardedStudent(
            student,
            entry["turns"],
            config.model.max_prompt_tokens,
            config.explorer.auxiliary_models[0].max_model_len,
        )
        from transformers import AutoTokenizer

        from opd_pilot.diagnostic_backend import TransformersBackend

        guarded.evaluator_backend = TransformersBackend.__new__(TransformersBackend)
        guarded.evaluator_backend.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_path, local_files_only=True
        )
        workflow = OnPolicyDistillVerlAgentAlfworldWorkflow(
            task=task, model=guarded, auxiliary_models=[GuardedTeacher(teacher, entry["turns"])]
        )
        experiences = await workflow.run_async()
        if len(experiences) != len(entry["turns"]) or not experiences:
            raise ValueError("Missing per-turn experiences")
        entry["status"] = "passed"
        (output / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    import ray
    import torch
    import yaml

    from trinity.common.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", episodes=[], planned_tasks=[], training_validated=False)
    try:
        raw = yaml.safe_load(args.config.read_text())
        if not validate_config(raw):
            raise ValueError("Expected per_step_h2")
        manifest(raw, args.output / "context-manifest.json")
        if torch.cuda.device_count() != 2 or any(
            "H200" not in torch.cuda.get_device_name(i) for i in range(2)
        ):
            raise ValueError("Acceptance requires the approved two H200 allocation")
        config = load_config(str(args.config))
        config.name += "-context-check-" + os.environ["SLURM_JOB_ID"]
        config.ray_namespace += "-context-check-" + os.environ["SLURM_JOB_ID"]
        logdir = Path(os.environ["LOG_ROOT"]) / "context" / os.environ["SLURM_JOB_ID"]
        logdir.mkdir(parents=True, exist_ok=True)
        config.log.save_dir = str(logdir)
        ray.init(
            address="local",
            namespace=config.ray_namespace,
            num_gpus=2,
            num_cpus=int(os.environ["SLURM_CPUS_PER_TASK"]),
            include_dashboard=False,
            _temp_dir="/tmp/opd-context-" + os.environ["SLURM_JOB_ID"],
            runtime_env={"env_vars": config.get_envs()},
        )
        config.check_and_update()
        config.log.save_dir = str(logdir)
        config.save(str(args.output / "resolved-config.yaml"))
        asyncio.run(exercise(config, args.output, report))
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        if report["episodes"] and report["episodes"][-1]["status"] == "running":
            report["episodes"][-1]["status"] = "failed"
        raise
    finally:
        (args.output / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        ray.shutdown()
        rayroot = Path("/tmp/opd-context-" + os.environ["SLURM_JOB_ID"])
        target = Path(os.environ["LOG_ROOT"]) / "context" / os.environ["SLURM_JOB_ID"]
        for i, logs in enumerate(rayroot.glob("session_*/logs")):
            shutil.copytree(logs, target / f"ray-session-{i+1}", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
