from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AllowedAction = Literal[
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

RiskLevel = Literal["low", "medium", "high"]
ReasoningDecision = Literal["continue", "done", "blocked"]


class ActionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, max_length=32)
    action: AllowedAction
    params: dict[str, Any]

    @model_validator(mode="after")
    def validate_params_presence(self) -> "ActionStep":
        if self.action != "stop" and not self.params:
            raise ValueError(f"{self.action} requires non-empty params")
        if self.action == "stop" and self.params is None:
            self.params = {}
        return self


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_to_user: str = Field(min_length=1, max_length=500)
    goal: str = Field(min_length=1, max_length=200)
    requires_confirmation: bool
    risk_level: RiskLevel
    plan: list[ActionStep] = Field(min_length=1, max_length=12)

    @field_validator("plan")
    @classmethod
    def unique_step_ids(cls, steps: list[ActionStep]) -> list[ActionStep]:
        ids = [step.step_id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id values must be unique")
        return steps

    @model_validator(mode="after")
    def validate_risk_confirmation(self) -> "ActionPlan":
        if self.risk_level == "high" and not self.requires_confirmation:
            raise ValueError("high risk plans require confirmation")
        return self


class ProgressDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ReasoningDecision
    reason: str = Field(min_length=1, max_length=500)
    next_focus: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)
