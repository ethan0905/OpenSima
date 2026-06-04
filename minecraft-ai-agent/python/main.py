from pydantic import ValidationError
from requests import RequestException

from config import get_settings
from minecraft_client import MinecraftClient
from gemini_planner import GeminiPlanner
from reasoning_loop import ReasoningLoop


def main() -> None:
    settings = get_settings()
    minecraft = MinecraftClient(settings.agent_base_url)
    planner = GeminiPlanner(
        settings.gemini_api_key,
        settings.gemini_model,
        fallback_model=settings.gemini_fallback_model,
    )
    loop = ReasoningLoop(
        minecraft=minecraft,
        planner=planner,
        max_iterations=settings.reasoning_max_iterations,
        max_steps_per_plan=settings.reasoning_max_steps_per_plan,
        verbose=settings.reasoning_verbose,
    )

    print("Minecraft AI Agent CLI")
    print("Type a request, or 'quit' to exit.")
    print(
        "Reasoning loop enabled: "
        f"{settings.reasoning_max_iterations} iterations, "
        f"{settings.reasoning_max_steps_per_plan} steps per plan."
    )

    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        try:
            result = loop.run(user_input, confirm=_confirm_plan)
            print(f"\nFinal> {result.status}: {result.reason}")
        except ValidationError as exc:
            print(f"Validation error: {exc}")
        except ValueError as exc:
            print(f"Planner/validation error: {exc}")
        except RequestException as exc:
            print(f"Minecraft agent API error: {exc}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


def _confirm_plan(plan) -> bool:
    risk_note = f" Risk: {plan.risk_level}." if plan.risk_level != "low" else ""
    if plan.requires_confirmation:
        risk_note += " Planner marked this batch as requiring confirmation."
    answer = input(f"Execute this objective with the reasoning loop?{risk_note} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


if __name__ == "__main__":
    main()
