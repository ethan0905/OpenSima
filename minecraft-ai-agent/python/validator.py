import json
import math
from typing import Any

from schemas import ActionPlan


ALLOWED_ACTIONS = {
    "move_to",
    "follow_player",
    "collect_block",
    "craft_item",
    "place_block",
    "build_structure",
    "attack_entity",
    "eat_food",
    "stop",
}

SUSPICIOUS_STRINGS = (
    "/give",
    "/tp",
    "/op",
    "/execute",
    "/summon",
    "/kill",
    "/setblock",
    "/fill",
    "command",
    "servercommand",
    "execute",
)

MAX_COUNT = 128
MAX_FOLLOW_DISTANCE = 64
MAX_ATTACK_DISTANCE = 32
MAX_MOVE_DISTANCE_WITHOUT_CONFIRMATION = 200


ACTION_PARAM_KEYS = {
    "move_to": {"x", "y", "z", "range"},
    "follow_player": {"player_name", "distance"},
    "collect_block": {"block", "count"},
    "craft_item": {"item", "count"},
    "place_block": {"block", "x", "y", "z"},
    "build_structure": {"structure", "material"},
    "attack_entity": {"entity_type", "max_distance"},
    "eat_food": {"food"},
    "stop": set(),
}


def validate_action_plan(plan: ActionPlan, state: dict[str, Any] | None = None) -> ActionPlan:
    for step in plan.plan:
        if step.action not in ALLOWED_ACTIONS:
            raise ValueError(f"unknown action: {step.action}")

        _reject_suspicious_values(step.params)
        _validate_param_keys(step.action, step.params)
        _validate_action_params(step.action, step.params, plan, state or {})

    return plan


def _validate_param_keys(action: str, params: dict[str, Any]) -> None:
    allowed = ACTION_PARAM_KEYS[action]
    extra = set(params) - allowed
    if extra:
        raise ValueError(f"{action} contains unsupported params: {sorted(extra)}")
    if action != "stop" and not params:
        raise ValueError(f"{action} requires params")


def _validate_action_params(
    action: str,
    params: dict[str, Any],
    plan: ActionPlan,
    state: dict[str, Any],
) -> None:
    if action in {"collect_block", "craft_item"}:
        _require_string(params, "block" if action == "collect_block" else "item")
        _require_count(params)

    if action == "move_to":
        x = _require_number(params, "x")
        y = _require_number(params, "y")
        z = _require_number(params, "z")
        _require_number(params, "range", minimum=0, maximum=32)
        distance = _distance_from_agent(state, x, y, z)
        if (
            distance is not None
            and distance > MAX_MOVE_DISTANCE_WITHOUT_CONFIRMATION
            and not plan.requires_confirmation
        ):
            raise ValueError("movement over 200 blocks requires confirmation")

    if action == "follow_player":
        _require_string(params, "player_name")
        _require_number(params, "distance", minimum=1, maximum=MAX_FOLLOW_DISTANCE)

    if action == "place_block":
        _require_string(params, "block")
        _require_number(params, "x")
        _require_number(params, "y")
        _require_number(params, "z")

    if action == "build_structure":
        _require_string(params, "material")
        structure = params.get("structure")
        if structure not in {"basic_shelter", "small_house", "pillar", "wall"}:
            raise ValueError("unsupported structure")

    if action == "attack_entity":
        _require_string(params, "entity_type")
        _require_number(params, "max_distance", minimum=1, maximum=MAX_ATTACK_DISTANCE)

    if action == "eat_food":
        food = params.get("food")
        if food is not None and not isinstance(food, str):
            raise ValueError("food must be a string or null")

    if action == "stop" and params:
        raise ValueError("stop params must be empty")


def _reject_suspicious_values(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True).lower()
    if any(token in serialized for token in SUSPICIOUS_STRINGS):
        raise ValueError("plan contains suspicious command-like content")


def _require_string(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_count(params: dict[str, Any]) -> int:
    value = params.get("count")
    if not isinstance(value, int) or value < 1 or value > MAX_COUNT:
        raise ValueError(f"count must be an integer from 1 to {MAX_COUNT}")
    return value


def _require_number(
    params: dict[str, Any],
    key: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = params.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return float(value)


def _distance_from_agent(
    state: dict[str, Any],
    x: float,
    y: float,
    z: float,
) -> float | None:
    try:
        pos = state["agent"]["position"]
        return math.dist((pos["x"], pos["y"], pos["z"]), (x, y, z))
    except (KeyError, TypeError):
        return None
