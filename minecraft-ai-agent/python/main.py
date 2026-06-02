from pydantic import ValidationError
from requests import RequestException

from config import get_settings
from minecraft_client import MinecraftClient
from openai_planner import OpenAIPlanner
from validator import validate_action_plan


def main() -> None:
    settings = get_settings()
    minecraft = MinecraftClient(settings.agent_base_url)
    planner = OpenAIPlanner(settings.openai_api_key, settings.openai_model)

    print("Minecraft AI Agent CLI")
    print("Type a request, or 'quit' to exit.")

    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        try:
            state = minecraft.get_state()
            plan = planner.create_plan(user_input, state)
            validate_action_plan(plan, state)

            print(f"\nAssistant> {plan.message_to_user}")
            if plan.requires_confirmation:
                answer = input("This plan requires confirmation. Execute? [y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("Execution cancelled.")
                    continue

            result = minecraft.send_action_plan(plan)
            print(f"Result> {result}")
        except ValidationError as exc:
            print(f"Validation error: {exc}")
        except ValueError as exc:
            print(f"Planner/validation error: {exc}")
        except RequestException as exc:
            print(f"Minecraft agent API error: {exc}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
