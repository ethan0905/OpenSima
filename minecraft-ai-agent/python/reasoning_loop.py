import json
from dataclasses import dataclass, field
from typing import Any, Callable

from minecraft_client import MinecraftClient
from openai_planner import OpenAIPlanner
from schemas import ActionPlan, ProgressDecision
from validator import validate_action_plan


ConfirmCallback = Callable[[ActionPlan], bool]


@dataclass
class ReasoningLoopResult:
    ok: bool
    status: str
    reason: str
    iterations: int
    history: list[dict[str, Any]] = field(default_factory=list)


class ReasoningLoop:
    def __init__(
        self,
        minecraft: MinecraftClient,
        planner: OpenAIPlanner,
        max_iterations: int = 5,
        max_steps_per_plan: int = 4,
        verbose: bool = True,
    ) -> None:
        self.minecraft = minecraft
        self.planner = planner
        self.max_iterations = max(1, max_iterations)
        self.max_steps_per_plan = max(1, min(max_steps_per_plan, 12))
        self.verbose = verbose

    def run(self, goal: str, confirm: ConfirmCallback | None = None) -> ReasoningLoopResult:
        history: list[dict[str, Any]] = []
        confirmed = False

        for iteration in range(1, self.max_iterations + 1):
            previous_state = self.minecraft.get_state()
            compact_history = _compact_history(history)

            if self.verbose:
                _print_trace("State before plan", _state_summary(previous_state))

            plan = self.planner.create_next_plan(
                goal,
                previous_state,
                compact_history,
                self.max_steps_per_plan,
            )
            validate_action_plan(plan, previous_state)
            _validate_plan_size(plan, self.max_steps_per_plan)

            print(f"\nIteration {iteration}/{self.max_iterations}")
            print(f"Assistant> {plan.message_to_user}")
            _print_plan(plan)

            if not confirmed and confirm is not None:
                confirmed = confirm(plan)
                if not confirmed:
                    return ReasoningLoopResult(
                        ok=False,
                        status="cancelled",
                        reason="Execution cancelled before the first action batch.",
                        iterations=iteration - 1,
                        history=history,
                    )

            result = self.minecraft.send_action_plan(plan)
            current_state = self.minecraft.get_state()

            if self.verbose:
                _print_trace("Execution result", result)
                _print_trace("State after plan", _state_summary(current_state))

            decision = self.planner.verify_progress(
                goal,
                previous_state,
                plan,
                result,
                current_state,
                compact_history,
            )

            history_entry = _history_entry(iteration, plan, result, decision, current_state)
            history.append(history_entry)

            print(
                "Verifier> "
                f"{decision.decision.upper()} - {decision.reason} "
                f"(next: {decision.next_focus}, confidence: {decision.confidence:.2f})"
            )

            if decision.decision == "done":
                return ReasoningLoopResult(
                    ok=True,
                    status="done",
                    reason=decision.reason,
                    iterations=iteration,
                    history=history,
                )

            if decision.decision == "blocked":
                return ReasoningLoopResult(
                    ok=False,
                    status="blocked",
                    reason=decision.reason,
                    iterations=iteration,
                    history=history,
                )

        return ReasoningLoopResult(
            ok=False,
            status="max_iterations",
            reason=f"Stopped after {self.max_iterations} reasoning iterations.",
            iterations=self.max_iterations,
            history=history,
        )


def _validate_plan_size(plan: ActionPlan, max_steps: int) -> None:
    if len(plan.plan) > max_steps:
        raise ValueError(f"planner returned {len(plan.plan)} steps; max is {max_steps}")


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    agent = state.get("agent", {})
    return {
        "agent": {
            "position": agent.get("position"),
            "health": agent.get("health"),
            "food": agent.get("food"),
            "held_item": agent.get("held_item"),
        },
        "inventory": state.get("inventory", []),
        "nearby_entities": [
            {
                "name": entity.get("name"),
                "type": entity.get("type"),
                "distance": entity.get("distance"),
            }
            for entity in state.get("nearby_entities", [])[:8]
        ],
        "nearby_blocks": [
            {
                "name": block.get("name"),
                "distance": block.get("distance"),
            }
            for block in state.get("nearby_blocks", [])[:12]
        ],
        "world": state.get("world", {}),
    }


def _history_entry(
    iteration: int,
    plan: ActionPlan,
    result: dict[str, Any],
    decision: ProgressDecision,
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "plan_goal": plan.goal,
        "steps": [
            {
                "step_id": step.step_id,
                "action": step.action,
                "params": step.params,
            }
            for step in plan.plan
        ],
        "result": _result_summary(result),
        "decision": decision.model_dump(),
        "state_after": _state_summary(state),
    }


def _compact_history(history: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    return history[-limit:]


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "failed_step_id": result.get("failed_step_id"),
        "error": result.get("error"),
        "completed_steps": result.get("completed_steps", []),
        "step_results": [
            {
                key: value
                for key, value in step_result.items()
                if key in {"step_id", "action", "ok", "partial", "message", "error"}
            }
            for step_result in result.get("step_results", [])
        ],
    }


def _print_plan(plan: ActionPlan) -> None:
    for index, step in enumerate(plan.plan, start=1):
        print(f"  {index}. {step.action} {step.params}")


def _print_trace(label: str, value: Any) -> None:
    print(f"\n{label}>")
    print(json.dumps(value, indent=2, sort_keys=True))
