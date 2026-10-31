"""Independent diagnostic episodes using TCOD's existing prompt/parser helpers."""

import importlib
from copy import deepcopy


class Episode:
    """Keep diagnostic environment state and full assistant history together."""

    def __init__(
        self, benchmark, task_desc, max_steps, utilities=None, context_mode="legacy_accumulated"
    ):
        if benchmark not in ("alfworld", "webshop", "scienceworld"):
            raise ValueError("Unknown benchmark")
        self.benchmark, self.max_steps = benchmark, max_steps
        self.u = utilities or importlib.import_module(
            f"trinity.common.workflows.envs.TCOD.{benchmark}.utils"
        )
        self.context_mode = context_mode
        if benchmark == "alfworld" and context_mode != "legacy_accumulated":
            from trinity.common.workflows.envs.TCOD.alfworld.context import (
                validate_context_mode,
            )

            validate_context_mode(context_mode)
        elif context_mode != "legacy_accumulated":
            raise ValueError("per_step_h2 is ALFWorld-only")
        self.env = None
        self.history, self.memory = [], []
        self.done, self.reward, self.best_score = False, None, None
        try:
            self._reset(task_desc)
        except BaseException:
            self.close()
            raise

    def _reset(self, task):
        if self.benchmark == "alfworld":
            self.env = self.u._create_alfworld_env(task)
            self.observation, self.info = self.env.reset()
            self.task_description = self.u._extract_task(self.observation)
        elif self.benchmark == "webshop":
            self.env = self.u._create_webshop_env()
            self.env.reset(session=int(task))
            self.observation, self.info = self.env.observation, None
            self.task_description = self.u._extract_task_description(self.observation)
        else:
            self.env = self.u._create_scienceworld_env(
                task, max_env_steps=self.max_steps, generate_gold_path=False
            )
            self.observation, self.info, self.task_description = self.u._reset_scienceworld_env(
                self.env
            )
            self.best_score = self.info.get("score", 0)

    def prompt(self):
        if self.done or len(self.history) >= self.max_steps:
            raise ValueError("Episode has ended")
        u, step = self.u, len(self.history)
        self.formatted = u.format_observation(self.observation)
        if self.benchmark == "alfworld" and self.context_mode == "per_step_h2":
            from trinity.common.workflows.envs.TCOD.alfworld.context import (
                build_alfworld_step_messages,
            )

            self.pending_messages = build_alfworld_step_messages(
                task_description=self.task_description,
                observation=self.observation,
                admissible_commands=self.info.get("admissible_commands", []),
                history=self.history,
                step_index=step,
            )
            return deepcopy(self.pending_messages)
        values = dict(
            task_description=self.task_description,
            current_observation=self.formatted,
            step_count=step,
            history_length=min(u.HISTORY_LENGTH, step),
            action_history="\n".join(self.history[-u.HISTORY_LENGTH :]),
            current_step=step + 1,
        )
        if self.benchmark == "alfworld":
            actions = self.info.get("admissible_commands", [])
            if actions and isinstance(actions[0], list):
                actions = actions[0]
            values["admissible_actions"] = "\n ".join(f"'{a}'" for a in actions if a != "help")
            prefix = "ALFWORLD"
        elif self.benchmark == "webshop":
            self.available_actions = self.env.get_available_actions()
            values["available_actions"] = u._format_available_actions(self.available_actions)
            prefix = "WEBSHOP"
        else:
            actions, objects = u._get_compact_action_info(self.env)
            values["action_templates"] = ", ".join(f"'{a}'" for a in actions if a != "help")
            values["objects"] = ", ".join(f"'{o}'" for o in objects)
            prefix = "SCIWORLD"
        suffix = "_TEMPLATE_NO_HIS" if step < u.HISTORY_LENGTH else "_TEMPLATE"
        self.pending_messages = self.memory + [
            {"role": "user", "content": getattr(u, prefix + suffix).format(**values)}
        ]
        return deepcopy(self.pending_messages)

    def apply(self, response):
        """Execute one full response, including its reasoning in subsequent history."""
        action = self.u.parse_action(response)
        if self.context_mode == "legacy_accumulated":
            self.memory = self.pending_messages + [{"role": "assistant", "content": response}]
        self.history.append(self.u._format_history(self.formatted, len(self.history) + 1, action))
        before, before_info = deepcopy(self.observation), deepcopy(self.info)
        valid, error = True, ""
        if self.benchmark == "webshop":
            valid, error = self.u.validate_action(action, self.available_actions)
        if valid:
            self.observation, self.reward, self.done, self.info = self.env.step(action)
        else:
            self.observation, self.reward, self.done, self.info = error, 0.0, False, None
        if self.benchmark == "scienceworld":
            self.best_score = max(
                self.best_score, self.info.get("score", self.best_score + self.reward)
            )
        return dict(
            action=action,
            observation_before=before,
            info_before=before_info,
            observation_after=deepcopy(self.observation),
            info_after=deepcopy(self.info),
            raw_reward=self.reward,
            episode_done=self.done,
            environment_step_executed=valid,
        )

    def outcome(self):
        result = dict(
            episode_done=self.done,
            task_success=None,
            raw_score=self.reward,
            technical_failure=False,
            termination_reason="env_terminal" if self.done else "horizon",
        )
        if self.benchmark == "scienceworld":
            result.update(score_final=self.info.get("score"), score_best=self.best_score)
        return result

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
