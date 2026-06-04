import json
import re
from typing import Any

from google import genai

from schemas import ActionPlan, ProgressDecision


SYSTEM_PROMPT = """You are a Minecraft planning agent.
Your role is to convert a user's natural-language request into a safe, executable JSON action plan for a Mineflayer agent.

You do not execute actions yourself.
You only produce JSON matching the provided schema.
You must only use allowed actions.
You must consider the current Minecraft state.
Prefer simple robust actions over complex fragile plans.
Every action except stop must include all required params from action_param_guide.
Never use empty params for move_to, follow_player, collect_block, craft_item, place_block, build_structure, attack_entity, or eat_food.
Required param examples:
- move_to: {"x": 10, "y": 64, "z": -4, "range": 1}
- collect_block: {"block": "sand", "count": 1}
- attack_entity: {"entity_type": "squid", "max_distance": 16}
- craft_item: {"item": "oak_planks", "count": 4}
- follow_player: {"player_name": "Steve", "distance": 3}
- build_structure: {"structure": "basic_shelter", "material": "oak_planks"}
- eat_food: {"food": null}
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


ACTION_PARAM_GUIDE = {
    "move_to": {"x": "number", "y": "number", "z": "number", "range": "number"},
    "follow_player": {"player_name": "string", "distance": "number"},
    "collect_block": {"block": "string", "count": "integer"},
    "craft_item": {"item": "string", "count": "integer"},
    "place_block": {"block": "string", "x": "number", "y": "number", "z": "number"},
    "build_structure": {"structure": "basic_shelter", "material": "string"},
    "attack_entity": {"entity_type": "string", "max_distance": "number"},
    "eat_food": {"food": "string or null"},
    "stop": {},
}


GEMINI_ACTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "message_to_user": {"type": "string"},
        "goal": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    "action": {"type": "string", "enum": ALLOWED_ACTIONS},
                    "params": {"type": "object"},
                },
                "required": ["step_id", "action", "params"],
            },
        },
    },
    "required": ["message_to_user", "goal", "requires_confirmation", "risk_level", "plan"],
}


GEMINI_PROGRESS_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["continue", "done", "blocked"]},
        "reason": {"type": "string"},
        "next_focus": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["decision", "reason", "next_focus", "confidence"],
}


class GeminiPlanner:
    def __init__(self, api_key: str, model: str, fallback_model: str | None = None) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.fallback_model = fallback_model or ""

    def create_plan(self, user_request: str, state: dict[str, Any]) -> ActionPlan:
        prompt = {
            "user_request": user_request,
            "current_minecraft_state": state,
            "allowed_actions": ALLOWED_ACTIONS,
            "action_param_guide": ACTION_PARAM_GUIDE,
            "json_schema": GEMINI_ACTION_PLAN_SCHEMA,
            "current_limitations": LIMITATIONS,
        }

        return self._generate_json(
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            schema=GEMINI_ACTION_PLAN_SCHEMA,
            model_type=ActionPlan,
            repair_context={
                "user_request": user_request,
                "current_minecraft_state": state,
                "recent_history": [],
                "allowed_actions": ALLOWED_ACTIONS,
                "action_param_guide": ACTION_PARAM_GUIDE,
                "current_limitations": LIMITATIONS,
                "max_steps": 12,
                "allow_repair": True,
            },
        )

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
            "action_param_guide": ACTION_PARAM_GUIDE,
            "json_schema": GEMINI_ACTION_PLAN_SCHEMA,
            "current_limitations": LIMITATIONS,
            "planning_instruction": (
                f"Create only the next minimal action batch, with 1 to {max_steps} steps. "
                "Do not try to solve the whole task in one large plan. "
                "Use recent_history to avoid repeating failed actions unless there is a clear reason. "
                "For each step, params must match action_param_guide exactly. "
                "Only stop may use empty params."
            ),
        }

        return self._generate_json(
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            schema=GEMINI_ACTION_PLAN_SCHEMA,
            model_type=ActionPlan,
            repair_context={
                "user_request": user_request,
                "current_minecraft_state": state,
                "recent_history": history,
                "allowed_actions": ALLOWED_ACTIONS,
                "action_param_guide": ACTION_PARAM_GUIDE,
                "current_limitations": LIMITATIONS,
                "max_steps": max_steps,
                "allow_repair": True,
            },
        )

    def repair_plan(
        self,
        user_request: str,
        state: dict[str, Any],
        history: list[dict[str, Any]],
        invalid_plan: dict[str, Any],
        validation_error: str,
        max_steps: int,
    ) -> ActionPlan:
        prompt = {
            "user_request": user_request,
            "current_minecraft_state": state,
            "recent_history": history,
            "invalid_plan": invalid_plan,
            "validation_error": validation_error,
            "allowed_actions": ALLOWED_ACTIONS,
            "action_param_guide": ACTION_PARAM_GUIDE,
            "json_schema": GEMINI_ACTION_PLAN_SCHEMA,
            "current_limitations": LIMITATIONS,
            "repair_instruction": (
                f"Return a corrected action plan with 1 to {max_steps} steps. "
                "Fix the validation error. "
                "Every non-stop action must include all required params from action_param_guide. "
                "Only stop may use empty params. "
                "Do not add unsupported param keys."
            ),
        }

        return self._generate_json(
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            schema=GEMINI_ACTION_PLAN_SCHEMA,
            model_type=ActionPlan,
            repair_context={
                "user_request": user_request,
                "current_minecraft_state": state,
                "recent_history": history,
                "allowed_actions": ALLOWED_ACTIONS,
                "action_param_guide": ACTION_PARAM_GUIDE,
                "current_limitations": LIMITATIONS,
                "max_steps": max_steps,
                "allow_repair": False,
            },
        )

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

        return self._generate_json(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            prompt=prompt,
            schema=GEMINI_PROGRESS_DECISION_SCHEMA,
            model_type=ProgressDecision,
        )

    def _generate_json(
        self,
        system_prompt: str,
        prompt: dict[str, Any],
        schema: dict[str, Any],
        model_type: type[ActionPlan] | type[ProgressDecision],
        repair_context: dict[str, Any] | None = None,
    ) -> ActionPlan | ProgressDecision:
        try:
            return self._generate_json_with_model(
                model=self.model,
                system_prompt=system_prompt,
                prompt=prompt,
                schema=schema,
                model_type=model_type,
                repair_context=repair_context,
            )
        except Exception:
            if not self.fallback_model or self.fallback_model == self.model:
                raise
            return self._generate_json_with_model(
                model=self.fallback_model,
                system_prompt=system_prompt,
                prompt=prompt,
                schema=schema,
                model_type=model_type,
                repair_context=repair_context,
            )

    def _generate_json_with_model(
        self,
        model: str,
        system_prompt: str,
        prompt: dict[str, Any],
        schema: dict[str, Any],
        model_type: type[ActionPlan] | type[ProgressDecision],
        repair_context: dict[str, Any] | None = None,
    ) -> ActionPlan | ProgressDecision:
        response = self.client.models.generate_content(
            model=model,
            contents=json.dumps(prompt, indent=2),
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        if model_type is ActionPlan:
            return self._parse_action_plan_response(response.text, repair_context)
        return model_type.model_validate_json(response.text)

    def _parse_action_plan_response(
        self,
        response_text: str,
        repair_context: dict[str, Any] | None,
    ) -> ActionPlan:
        try:
            raw_plan = json.loads(response_text)
        except json.JSONDecodeError as exc:
            if repair_context is None or not repair_context.get("allow_repair", False):
                raise
            return self._repair_raw_action_plan(response_text, str(exc), repair_context)

        if repair_context is not None:
            _normalize_action_plan(
                raw_plan,
                repair_context.get("current_minecraft_state", {}),
                repair_context.get("user_request", ""),
            )

        try:
            return ActionPlan.model_validate(raw_plan)
        except Exception as exc:
            if repair_context is None or not repair_context.get("allow_repair", False):
                raise
            return self._repair_raw_action_plan(json.dumps(raw_plan), str(exc), repair_context)

    def _repair_raw_action_plan(
        self,
        invalid_response_text: str,
        validation_error: str,
        repair_context: dict[str, Any],
    ) -> ActionPlan:
        try:
            invalid_plan = json.loads(invalid_response_text)
        except json.JSONDecodeError:
            invalid_plan = {"raw_response": invalid_response_text}

        return self.repair_plan(
            user_request=repair_context["user_request"],
            state=repair_context["current_minecraft_state"],
            history=repair_context["recent_history"],
            invalid_plan=invalid_plan,
            validation_error=validation_error,
            max_steps=repair_context["max_steps"],
        )


BLOCK_ALIASES = {
    "red flower": "poppy",
    "red_flower": "poppy",
    "poppy": "poppy",
    "yellow flower": "dandelion",
    "yellow_flower": "dandelion",
    "dandelion": "dandelion",
    "flower": "poppy",
    "sand": "sand",
    "cactus": "cactus",
    "dirt": "dirt",
    "grass": "grass_block",
    "stone": "stone",
    "cobblestone": "cobblestone",
    "tree": "oak_log",
    "log": "oak_log",
    "wood": "oak_log",
    "oak": "oak_log",
    "oak log": "oak_log",
}

ITEM_ALIASES = {
    "plank": "oak_planks",
    "planks": "oak_planks",
    "oak plank": "oak_planks",
    "oak planks": "oak_planks",
    "stick": "stick",
    "sticks": "stick",
    "crafting table": "crafting_table",
    "torch": "torch",
}

ENTITY_ALIASES = {
    "squid": "squid",
    "cow": "cow",
    "pig": "pig",
    "sheep": "sheep",
    "chicken": "chicken",
    "zombie": "zombie",
    "skeleton": "skeleton",
    "creeper": "creeper",
    "spider": "spider",
}


def _normalize_action_plan(raw_plan: Any, state: dict[str, Any], user_request: str) -> None:
    if not isinstance(raw_plan, dict):
        return
    steps = raw_plan.get("plan")
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        params = step.get("params")
        if params is None:
            step["params"] = {}
            params = step["params"]
        if not isinstance(params, dict) or params:
            continue
        if action == "move_to":
            step["params"] = _fallback_move_params(state)
        elif action == "collect_block":
            block = _infer_block_name(user_request, state)
            if block is not None:
                step["params"] = {"block": block, "count": _infer_count(user_request)}
        elif action == "craft_item":
            item = _infer_alias(user_request, ITEM_ALIASES)
            if item is not None:
                step["params"] = {"item": item, "count": _infer_count(user_request)}
        elif action == "attack_entity":
            entity_type = _infer_entity_type(user_request, state)
            if entity_type is not None:
                step["params"] = {"entity_type": entity_type, "max_distance": 16}
        elif action == "follow_player":
            player_name = _nearest_player_name(state)
            if player_name is not None:
                step["params"] = {"player_name": player_name, "distance": 3}
        elif action == "build_structure":
            material = _infer_build_material(user_request, state)
            if material is not None:
                step["params"] = {"structure": "basic_shelter", "material": material}
        elif action == "eat_food":
            step["params"] = {"food": _infer_food(state)}


def _fallback_move_params(state: dict[str, Any]) -> dict[str, float]:
    target = _first_position(state.get("nearby_blocks")) or _first_position(state.get("nearby_entities"))
    if target is None:
        target = state.get("agent", {}).get("position") or {"x": 0, "y": 64, "z": 0}
    return {
        "x": float(target.get("x", 0)),
        "y": float(target.get("y", 64)),
        "z": float(target.get("z", 0)),
        "range": 1,
    }


def _first_position(items: Any) -> dict[str, Any] | None:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        position = item.get("position")
        if isinstance(position, dict):
            return position
    return None


def _infer_count(text: str) -> int:
    match = re.search(r"\b(\d{1,3})\b", text)
    if match is None:
        return 1
    return max(1, min(int(match.group(1)), 128))


def _infer_block_name(text: str, state: dict[str, Any]) -> str | None:
    alias = _infer_alias(text, BLOCK_ALIASES)
    if alias is not None:
        return alias

    nearby_names = _nearby_names(state.get("nearby_blocks"))
    for name in nearby_names:
        if name in text.lower():
            return name
    return nearby_names[0] if nearby_names else None


def _infer_entity_type(text: str, state: dict[str, Any]) -> str | None:
    alias = _infer_alias(text, ENTITY_ALIASES)
    if alias is not None:
        return alias

    nearby_names = _nearby_names(state.get("nearby_entities"))
    for name in nearby_names:
        if name in text.lower():
            return name
    return nearby_names[0] if nearby_names else None


def _infer_alias(text: str, aliases: dict[str, str]) -> str | None:
    normalized = text.lower().replace("_", " ")
    for phrase in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return aliases[phrase]
    return None


def _nearby_names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("type")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _nearest_player_name(state: dict[str, Any]) -> str | None:
    if not isinstance(state.get("nearby_entities"), list):
        return None
    for entity in state["nearby_entities"]:
        if not isinstance(entity, dict):
            continue
        if entity.get("type") == "player" and isinstance(entity.get("name"), str):
            return entity["name"]
    return None


def _infer_build_material(text: str, state: dict[str, Any]) -> str | None:
    requested = _infer_alias(text, BLOCK_ALIASES)
    if requested is not None:
        return requested

    if not isinstance(state.get("inventory"), list):
        return "dirt"
    preferred = ["oak_planks", "cobblestone", "dirt", "stone", "oak_log"]
    inventory_names = {
        item.get("name")
        for item in state["inventory"]
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("count", 0) > 0
    }
    for material in preferred:
        if material in inventory_names:
            return material
    return "dirt"


def _infer_food(state: dict[str, Any]) -> str | None:
    foods = {
        "apple",
        "bread",
        "cooked_beef",
        "cooked_porkchop",
        "cooked_chicken",
        "cooked_mutton",
        "baked_potato",
        "carrot",
    }
    if not isinstance(state.get("inventory"), list):
        return None
    for item in state["inventory"]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name in foods and item.get("count", 0) > 0:
            return name
    return None
