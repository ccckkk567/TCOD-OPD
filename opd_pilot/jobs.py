"""Lightweight, user-invoked Slurm submission and reconciliation; no remote access."""

import argparse
import json
import re
import subprocess
from pathlib import Path

STAGES = {
    "selection_analysis",
    "report",
    "task_protocol",
    "paired_stats",
    "train",
    "pool",
    "smoke",
    "webshop_index",
    "export",
    "branch_analysis",
    "evaluate",
    "continuation",
    "candidates",
    "analyze",
    "audit",
    "probes",
    "checkpoints",
}


def order(tasks):
    """Validate the whole DAG before submitting any jobs."""
    by_name = {task["name"]: task for task in tasks}
    if len(by_name) != len(tasks) or not tasks:
        raise ValueError("Require nonempty unique job names")
    pending, result = dict(by_name), []
    for name, task in pending.items():
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
            raise ValueError("Use readable job names")
        if task["stage"] not in STAGES:
            raise ValueError("Unsupported stage")
        if not re.fullmatch(r"[1-9][0-9]*[MG]", task.get("memory", "32G")):
            raise ValueError("Memory must be a positive M/G size")
        hours, minutes, seconds = map(int, task["time"].split(":"))
        if not (
            0 <= minutes < 60
            and 0 <= seconds < 60
            and 0 < hours * 3600 + minutes * 60 + seconds <= 64800
        ):
            raise ValueError("Job duration must be within 18 hours")
        if any(dep not in by_name for dep in task.get("after", [])):
            raise ValueError("Unknown dependency")
        if task.get("dependency_mode", "afterok") not in ("afterok", "afterany"):
            raise ValueError("Use afterok or afterany dependencies")
        if not Path(task["spec"]).is_absolute() or not Path(task["spec"]).is_file():
            raise ValueError("Each job requires an existing absolute spec path")
    while pending:
        ready = [
            name
            for name, task in pending.items()
            if all(dep not in pending for dep in task.get("after", []))
        ]
        if not ready:
            raise ValueError("Dependency cycle")
        for name in ready:
            result.append(pending.pop(name))
    return result


def save(path, payload):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def submit(spec, registry):
    tasks = order(spec["tasks"])
    repo = Path(spec["repo"]).resolve()
    script = repo / "orion/slurm/diagnostic_stage.sbatch"
    if not script.is_file():
        raise FileNotFoundError(script)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != spec["revision"]:
        raise ValueError("Checkout differs from requested revision")
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(
            "Submit from a clean checkout and keep it unchanged while jobs are queued/running"
        )
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    state = dict(revision=revision, repo=str(repo), jobs={})
    with registry.open("x", encoding="utf-8") as stream:
        json.dump(state, stream)
    for task in tasks:
        name = task["name"]
        logs = Path(spec["logs"]) / name
        logs.mkdir(parents=True, exist_ok=True)
        args = [
            "sbatch",
            "--parsable",
            "--account=prjsnnn0000269",
            "--partition=testqueue",
            "--gres=gpu:8" if task["stage"] == "train" else "--gres=gpu:1",
            "--ntasks=1",
            "--cpus-per-task=32" if task["stage"] == "train" else "--cpus-per-task=4",
            "--mem=" + task.get("memory", "256G" if task["stage"] == "train" else "32G"),
            "--time=" + task["time"],
            "--job-name=opd-" + name,
            "--chdir=" + str(repo),
            "--output=" + str(logs / "%j.log"),
        ]
        if task.get("after"):
            args.append(
                "--dependency="
                + task.get("dependency_mode", "afterok")
                + ":"
                + ":".join(state["jobs"][dep]["job_id"] for dep in task["after"])
            )
        if task["stage"] == "train":
            args.extend([str(repo / "orion/slurm/train.sbatch"), task["spec"], revision])
        else:
            args.extend([str(script), task["stage"], task["spec"], revision])
        state["jobs"][name] = dict(task=task, status="submission_in_progress")
        save(registry, state)
        response = subprocess.run(args, capture_output=True, text=True)
        if response.returncode:
            state["jobs"][name]["status"] = "submission_failed"
            save(registry, state)
            raise RuntimeError(f"sbatch failed for {name}: {response.stderr.strip()}")
        job_id = response.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError("Unrecognized submission response; inspect squeue before retrying")
        state["jobs"][name].update(job_id=job_id, status="submitted")
        save(registry, state)
    return state


def reconcile(registry):
    path = Path(registry)
    state = json.loads(path.read_text(encoding="utf-8"))
    ids = [job["job_id"] for job in state["jobs"].values() if "job_id" in job]
    if not ids:
        return state
    result = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(ids),
            "--format=JobIDRaw,State,ExitCode,Elapsed",
        ],
        text=True,
    )
    observations = {}
    for line in result.splitlines():
        fields = line.split("|")
        if len(fields) >= 4:
            observations[fields[0]] = dict(
                slurm_state=fields[1], exit_code=fields[2], elapsed=fields[3]
            )
    for job in state["jobs"].values():
        if job.get("job_id") in observations:
            job.update(observations[job["job_id"]])
    save(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validate", "submit", "reconcile"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    if args.operation == "reconcile":
        if not args.registry:
            parser.error("--registry is required")
        result = reconcile(args.registry)
    else:
        if not args.spec:
            parser.error("--spec is required")
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if args.operation == "validate":
            result = dict(order=[task["name"] for task in order(spec["tasks"])])
        else:
            if not args.registry:
                parser.error("--registry is required")
            result = submit(spec, args.registry)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
