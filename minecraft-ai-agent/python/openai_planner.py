import json
from typing import Any

from openai import OpenAI

from schemas import ActionPlan, ProgressDecision


SYSTEM_PROMPT = """You are a Minecraft planning agent.
Your role is to convert a user's natural-language request into a safe, executable JSON action plan for a Mineflayer agent.

You do not execute actions yourself.
You only produce JSON matching the provided schema.
You must only use allowed actions.
You must consider the current Minecraft state.
Prefer simple robust actions over complex fragile plans.
If the user asks for something impossible with current resources, create a plan to gather required resources first.
If the user is in danger, prioritize survival.
If health is low, eat or retreat first.
If it is night and hostile mobs are nearby, prioritize shelter or safety.
Never generate arbitrary Minecraft server commands.
Never generate slash commands.
Never generate code.
Never generate keyboard/mouse instructions.
Never invent actions outside the whitelist.
Do not include explanations outside JSON."""


VERIFIER_SYSTEM_PROMPT = """You are a Minecraft task verifier.
Your job is to decide whether a Minecraft agent has completed, should continue, or is blocked on a user's objective.

You do not produce actions.
You only produce JSON matching the provided schema.
Use the execution result and the latest Minecraft state.
If the last plan failed but there is an obvious safe next attempt, choose continue.
If the objective is clearly satisfied, choose done.
If the objective cannot be completed with the current action capabilities or repeated failures show no progress, choose blocked.
Do not include explanations outside JSON."""


ALLOWED_ACTIONS = [
    "move_to",
    "follow_player",
    "collect_block",
    "craft_item",
    "place_block",
    "build_structure",
    "attack_entity",
    "eat_food",
    "stop",
]


LIMITATIONS = [
    "The agent can only execute whitelisted actions.",
    "The agent cannot run Minecraft slash commands.",
    "The agent can only collect nearby blocks detected by the Mineflayer state scan.",
    "Only basic_shelter is implemented for robust MVP structure building.",
    "Combat is simple and should only be used for nearby threats.",
]


def _step_schema(action: str, params: dict[str, Any], required_params: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "step_id": {"type": "string"},
            "action": {"type": "string", "enum": [action]},
            "params": {
                "type": "object",
                "additionalProperties": False,
                "properties": params,
                "required": required_params,
            },
        },
        "required": ["step_id", "action", "params"],
    }


OPENAI_ACTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "message_to_user": {"type": "string"},
        "goal": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "plan": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "anyOf": [
                    _step_schema(
                        "move_to",
                        {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "range": {"type": "number"},
                        },
                        ["x", "y", "z", "range"],
                    ),
                    _step_schema(
                        "follow_player",
                        {
                            "player_name": {"type": "string"},
                            "distance": {"type": "number"},
                        },
                        ["player_name", "distance"],
                    ),
                    _step_schema(
                        "collect_block",
                        {
                            "block": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        ["block", "count"],
                    ),
                    _step_schema(
                        "craft_item",
                        {
                            "item": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        ["item", "count"],
                    ),
                    _step_schema(
                        "place_block",
                        {
                            "block": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                        ["block", "x", "y", "z"],
                    ),
                    _step_schema(
                        "build_structure",
                        {
                            "structure": {"type": "string", "enum": ["basic_shelter", "small_house", "pillar", "wall"]},
                            "material": {"type": "string"},
                        },
                        ["structure", "material"],
                    ),
                    _step_schema(
                        "attack_entity",
                        {
                            "entity_type": {"type": "string"},
                            "max_distance": {"type": "number"},
                        },
                        ["entity_type", "max_distance"],
                    ),
                    _step_schema(
                        "eat_food",
                        {
                            "food": {"type": ["string", "null"]},
                        },
                        ["food"],
                    ),
                    _step_schema("stop", {}, []),
                ]
            },
        },
    },
    "required": ["message_to_user", "goal", "requires_confirmation", "risk_level", "plan"],
}


OPENAI_PROGRESS_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["continue", "done", "blocked"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "next_focus": {"type": "string", "minLength": 1, "maxLength": 300},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["decision", "reason", "next_focus", "confidence"],
}


class OpenAIPlanner:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_plan(self, user_request: str, state: dict[str, Any]) -> ActionPlan:
        prompt = {
            "user_request": user_request,
            "current_minecraft_state": state,
            "allowed_actions": ALLOWED_ACTIONS,
            "json_schema": OPENAI_ACTION_PLAN_SCHEMA,
            "current_limitations": LIMITATIONS,
        }

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "minecraft_action_plan",
                    "strict": True,
                    "schema": OPENAI_ACTION_PLAN_SCHEMA,
                }
            },
        )
        return ActionPlan.model_validate_json(response.output_text)

    def create_next_plan(
        self,
        user_request: str,
        state: dict[str, Any],
        history: list[dict[str, Any]],
        max_steps: int,
    ) -> ActionPlan:
        prompt = {
            "user_request": user_request,
            "current_minecraft_state": state,
            "recent_history": history,
            "allowed_actions": ALLOWED_ACTIONS,
            "json_schema": OPENAI_ACTION_PLAN_SCHEMA,
            "current_limitations": LIMITATIONS,
            "planning_instruction": (
                f"Create only the next minimal action batch, with 1 to {max_steps} steps. "
                "Do not try to solve the whole task in one large plan. "
                "Use recent_history to avoid repeating failed actions unless there is a clear reason."
            ),
        }

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "minecraft_next_action_plan",
                    "strict": True,
                    "schema": OPENAI_ACTION_PLAN_SCHEMA,
                }
            },
        )
        return ActionPlan.model_validate_json(response.output_text)

    def verify_progress(
        self,
        user_request: str,
        previous_state: dict[str, Any],
        plan: ActionPlan,
        result: dict[str, Any],
        current_state: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> ProgressDecision:
        prompt = {
            "user_request": user_request,
            "previous_minecraft_state": previous_state,
            "executed_plan": plan.model_dump(),
            "execution_result": result,
            "current_minecraft_state": current_state,
            "recent_history": history,
            "decision_meanings": {
                "continue": "More safe progress is possible and needed.",
                "done": "The user's objective is satisfied.",
                "blocked": "The task cannot currently be completed or repeated attempts are not making progress.",
            },
        }

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "minecraft_progress_decision",
                    "strict": True,
                    "schema": OPENAI_PROGRESS_DECISION_SCHEMA,
                }
            },
        )
        return ProgressDecision.model_validate_json(response.output_text)
