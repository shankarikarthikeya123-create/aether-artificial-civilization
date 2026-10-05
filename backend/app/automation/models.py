from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AutomationStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class TriggerType(str, Enum):
    RESOURCE = "resource"
    POPULATION = "population"
    ECONOMY = "economy"
    DISASTER = "disaster"
    CITIZEN = "citizen"
    SECURITY = "security"
    KNOWLEDGE = "knowledge"
    CUSTOM = "custom"


@dataclass
class AutomationTrigger:
    id: str
    name: str
    trigger_type: TriggerType
    condition: str
    threshold: float | None = None
    resource: str | None = None
    enabled: bool = True


@dataclass
class AutomationAction:
    id: str
    name: str
    action_type: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    completed: bool = False


@dataclass
class AutomationWorkflow:
    id: str
    name: str
    description: str
    trigger_id: str
    priority: int
    status: AutomationStatus = AutomationStatus.PENDING
    actions: list[AutomationAction] = field(default_factory=list)
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    result: str | None = None


@dataclass
class AutomationEvent:
    id: str
    timestamp: str
    event_type: str
    title: str
    description: str
    severity: str
    workflow_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)