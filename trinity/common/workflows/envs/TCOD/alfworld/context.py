"""Explicit ALFWorld input protocol reconstructed from TCOD issue #9."""

from .utils import ALFWORLD_TEMPLATE, HISTORY_LENGTH, format_observation


def validate_context_mode(mode):
    if mode not in ("legacy_accumulated", "per_step_h2"):
        raise ValueError(f"Unsupported ALFWorld context_mode: {mode}")
    if mode == "per_step_h2" and HISTORY_LENGTH != 2:
        raise ValueError("per_step_h2 requires HISTORY_LENGTH == 2")
    return mode


def build_alfworld_step_messages(
    *, task_description, observation, admissible_commands, history, step_index
):
    validate_context_mode("per_step_h2")
    commands = admissible_commands
    if commands and isinstance(commands[0], list):
        commands = commands[0]
    recent = history[-HISTORY_LENGTH:]
    content = ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        current_observation=format_observation(observation),
        admissible_actions="\n ".join(f"'{command}'" for command in commands if command != "help"),
        step_count=step_index,
        current_step=step_index + 1,
        history_length=len(recent),
        action_history="\n".join(recent),
    )
    return [{"role": "user", "content": content}]
