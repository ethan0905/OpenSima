from typing import Any

import requests

from schemas import ActionPlan


class MinecraftClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def send_action_plan(self, plan: ActionPlan) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/action-plan",
            json=plan.model_dump(),
            timeout=max(self.timeout, 120.0),
        )
        response.raise_for_status()
        return response.json()
