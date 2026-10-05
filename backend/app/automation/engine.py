from datetime import datetime, timedelta
from typing import Any

from .models import (
    AutomationAction,
    AutomationEvent,
    AutomationStatus,
    AutomationTrigger,
    AutomationWorkflow,
    TriggerType,
)


class AutomationEngine:
    """
    Autonomous decision and workflow engine for AETHER.

    The engine:
    1. Watches civilization conditions.
    2. Detects situations that require action.
    3. Starts the appropriate protocol.
    4. Executes its actions.
    5. Records what happened.
    6. Uses cooldowns so the same condition does not trigger
       continuously every simulation tick.
    """

    def __init__(self, world_engine, citizen_engine, memory_engine):
        self.world_engine = world_engine
        self.citizen_engine = citizen_engine
        self.memory_engine = memory_engine

        self.triggers: list[AutomationTrigger] = []
        self.workflows: list[AutomationWorkflow] = []
        self.events: list[AutomationEvent] = []

        # Prevents identical automation protocols from firing
        # repeatedly while their condition remains unchanged.
        self.cooldowns: dict[str, datetime] = {}

        # Minimum time between executions of the same workflow.
        self.default_cooldown_minutes = 60

        self._create_default_triggers()
        self._create_default_workflows()

    # ---------------------------------------------------------
    # SETUP
    # ---------------------------------------------------------

    def _create_default_triggers(self):
        self.triggers = [
            AutomationTrigger(
                id="trigger_food_crisis",
                name="Food Crisis Detector",
                trigger_type=TriggerType.RESOURCE,
                condition="resource percentage below threshold",
                threshold=20,
                resource="food",
            ),
            AutomationTrigger(
                id="trigger_water_crisis",
                name="Water Crisis Detector",
                trigger_type=TriggerType.RESOURCE,
                condition="resource percentage below threshold",
                threshold=20,
                resource="water",
            ),
            AutomationTrigger(
                id="trigger_energy_crisis",
                name="Energy Crisis Detector",
                trigger_type=TriggerType.RESOURCE,
                condition="resource percentage below threshold",
                threshold=20,
                resource="energy",
            ),
            AutomationTrigger(
                id="trigger_employment",
                name="Employment Imbalance Detector",
                trigger_type=TriggerType.ECONOMY,
                condition="unemployment percentage above threshold",
                threshold=15,
            ),
            AutomationTrigger(
                id="trigger_knowledge",
                name="Knowledge Opportunity Detector",
                trigger_type=TriggerType.KNOWLEDGE,
                condition="research population available",
                threshold=1,
            ),
        ]

    def _create_default_workflows(self):
        self.workflows = [
            AutomationWorkflow(
                id="protocol_food",
                name="FOOD CRISIS PROTOCOL",
                description="Automatically respond to a shortage of food.",
                trigger_id="trigger_food_crisis",
                priority=1,
                actions=[
                    AutomationAction(
                        id="food_action_1",
                        name="Increase food production",
                        action_type="resource_production",
                        description="Increase food production capacity.",
                        parameters={
                            "resource": "food",
                            "amount": 300,
                        },
                    ),
                    AutomationAction(
                        id="food_action_2",
                        name="Allocate agricultural workers",
                        action_type="citizen_allocation",
                        description="Move available citizens toward food production.",
                        parameters={
                            "occupation": "farmer",
                            "count": 5,
                        },
                    ),
                    AutomationAction(
                        id="food_action_3",
                        name="Record food crisis response",
                        action_type="memory",
                        description="Record the civilization response.",
                        parameters={
                            "content": "AETHER responded automatically to a food shortage.",
                        },
                    ),
                ],
            ),
            AutomationWorkflow(
                id="protocol_water",
                name="WATER CRISIS PROTOCOL",
                description="Automatically respond to a shortage of water.",
                trigger_id="trigger_water_crisis",
                priority=1,
                actions=[
                    AutomationAction(
                        id="water_action_1",
                        name="Increase water production",
                        action_type="resource_production",
                        description="Increase water production.",
                        parameters={
                            "resource": "water",
                            "amount": 150,
                        },
                    ),
                    AutomationAction(
                        id="water_action_2",
                        name="Record water crisis response",
                        action_type="memory",
                        description="Record the civilization response.",
                        parameters={
                            "content": "AETHER responded automatically to a water shortage.",
                        },
                    ),
                ],
            ),
            AutomationWorkflow(
                id="protocol_energy",
                name="ENERGY CRISIS PROTOCOL",
                description="Automatically respond to an energy shortage.",
                trigger_id="trigger_energy_crisis",
                priority=1,
                actions=[
                    AutomationAction(
                        id="energy_action_1",
                        name="Increase energy production",
                        action_type="resource_production",
                        description="Increase energy production.",
                        parameters={
                            "resource": "energy",
                            "amount": 200,
                        },
                    ),
                    AutomationAction(
                        id="energy_action_2",
                        name="Record energy crisis response",
                        action_type="memory",
                        description="Record the civilization response.",
                        parameters={
                            "content": "AETHER responded automatically to an energy shortage.",
                        },
                    ),
                ],
            ),
            AutomationWorkflow(
                id="protocol_employment",
                name="EMPLOYMENT BALANCE PROTOCOL",
                description="Automatically respond to employment imbalance.",
                trigger_id="trigger_employment",
                priority=2,
                actions=[
                    AutomationAction(
                        id="employment_action_1",
                        name="Analyze available occupations",
                        action_type="analysis",
                        description="Analyze current occupational distribution.",
                        parameters={},
                    ),
                    AutomationAction(
                        id="employment_action_2",
                        name="Create employment opportunities",
                        action_type="employment",
                        description="Create opportunities in underrepresented occupations.",
                        parameters={},
                    ),
                    AutomationAction(
                        id="employment_action_3",
                        name="Record employment response",
                        action_type="memory",
                        description="Record the employment balancing response.",
                        parameters={
                            "content": "AETHER automatically responded to employment imbalance.",
                        },
                    ),
                ],
            ),
            AutomationWorkflow(
                id="protocol_knowledge",
                name="KNOWLEDGE DISCOVERY PROTOCOL",
                description="Automatically use available researchers to generate new knowledge.",
                trigger_id="trigger_knowledge",
                priority=3,
                actions=[
                    AutomationAction(
                        id="knowledge_action_1",
                        name="Identify researchers",
                        action_type="research",
                        description="Identify citizens capable of research.",
                        parameters={},
                    ),
                    AutomationAction(
                        id="knowledge_action_2",
                        name="Generate discovery",
                        action_type="knowledge",
                        description="Create a new civilization knowledge event.",
                        parameters={},
                    ),
                    AutomationAction(
                        id="knowledge_action_3",
                        name="Record discovery",
                        action_type="memory",
                        description="Store the discovery in civilization memory.",
                        parameters={
                            "content": "AETHER discovered new knowledge through autonomous research.",
                        },
                    ),
                ],
            ),
        ]

    # ---------------------------------------------------------
    # TRIGGER EVALUATION
    # ---------------------------------------------------------

    def evaluate(self):
        """
        Return workflows whose triggers are currently satisfied
        and which are not inside their cooldown period.
        """

        eligible = []

        for workflow in sorted(
            self.workflows,
            key=lambda workflow: workflow.priority,
        ):
            trigger = self._get_trigger(workflow.trigger_id)

            if trigger is None or not trigger.enabled:
                continue

            if not self._trigger_condition_met(trigger):
                continue

            if self._is_on_cooldown(workflow.id):
                continue

            eligible.append(workflow)

        return eligible

    def _get_trigger(self, trigger_id: str):
        for trigger in self.triggers:
            if trigger.id == trigger_id:
                return trigger

        return None

    def _trigger_condition_met(self, trigger: AutomationTrigger):
        world = self.world_engine.get_world()

        if trigger.trigger_type == TriggerType.RESOURCE:
            resource = next(
                (
                    item
                    for item in world.resources
                    if item.name == trigger.resource
                ),
                None,
            )

            if resource is None or resource.capacity <= 0:
                return False

            percentage = (
                resource.amount / resource.capacity
            ) * 100

            return percentage < (trigger.threshold or 0)

        if trigger.trigger_type == TriggerType.ECONOMY:
            citizens = self.citizen_engine.get_all()

            if not citizens:
                return False

            unemployed = [
                citizen
                for citizen in citizens
                if citizen.alive
                and citizen.occupation.lower()
                in {"unemployed", "jobless"}
            ]

            unemployment_percentage = (
                len(unemployed) / len(citizens)
            ) * 100

            return unemployment_percentage > (trigger.threshold or 0)

        if trigger.trigger_type == TriggerType.KNOWLEDGE:
            researchers = [
                citizen
                for citizen in self.citizen_engine.get_all()
                if citizen.alive
                and any(
                    keyword in citizen.occupation.lower()
                    for keyword in [
                        "research",
                        "scientist",
                        "engineer",
                        "scholar",
                    ]
                )
            ]

            return len(researchers) >= (trigger.threshold or 1)

        return False

    # ---------------------------------------------------------
    # COOLDOWN SYSTEM
    # ---------------------------------------------------------

    def _is_on_cooldown(self, workflow_id: str):
        expires_at = self.cooldowns.get(workflow_id)

        if expires_at is None:
            return False

        now = datetime.now()

        if now >= expires_at:
            del self.cooldowns[workflow_id]
            return False

        return True

    def _start_cooldown(self, workflow_id: str):
        self.cooldowns[workflow_id] = (
            datetime.now()
            + timedelta(minutes=self.default_cooldown_minutes)
        )

    def get_cooldown_status(self):
        now = datetime.now()

        status = {}

        for workflow_id, expires_at in self.cooldowns.items():
            remaining = expires_at - now

            if remaining.total_seconds() <= 0:
                status[workflow_id] = 0
            else:
                status[workflow_id] = int(
                    remaining.total_seconds()
                )

        return status

    # ---------------------------------------------------------
    # AUTOMATION CYCLE
    # ---------------------------------------------------------

    def run_cycle(self):
        """
        Evaluate and execute eligible autonomous workflows.
        """

        results = []

        eligible_workflows = self.evaluate()

        for workflow in eligible_workflows:
            result = self.execute_workflow(workflow)
            results.append(result)

            # Start cooldown only after successful execution.
            if result.status == AutomationStatus.COMPLETED:
                self._start_cooldown(workflow.id)

        return results

    # ---------------------------------------------------------
    # WORKFLOW EXECUTION
    # ---------------------------------------------------------

    def execute_workflow(self, workflow: AutomationWorkflow):
        now = datetime.now().isoformat()

        workflow.status = AutomationStatus.ACTIVE
        workflow.started_at = now

        self._add_event(
            event_type="automation_started",
            title=f"Automation started: {workflow.name}",
            description=(
                f"Automation trigger "
                f"'{self._get_trigger(workflow.trigger_id).name}' "
                f"condition was satisfied."
            ),
            severity="medium",
            workflow_id=workflow.id,
        )

        try:
            for action in workflow.actions:
                self._execute_action(workflow, action)

            workflow.status = AutomationStatus.COMPLETED
            workflow.completed_at = datetime.now().isoformat()
            workflow.result = (
                f"{workflow.name} completed successfully."
            )

            self._add_event(
                event_type="automation_completed",
                title=f"Automation completed: {workflow.name}",
                description=workflow.result,
                severity="info",
                workflow_id=workflow.id,
            )

        except Exception as error:
            workflow.status = AutomationStatus.FAILED
            workflow.completed_at = datetime.now().isoformat()
            workflow.result = str(error)

            self._add_event(
                event_type="automation_failed",
                title=f"Automation failed: {workflow.name}",
                description=str(error),
                severity="high",
                workflow_id=workflow.id,
            )

        return workflow

    # ---------------------------------------------------------
    # ACTION EXECUTION
    # ---------------------------------------------------------

    def _execute_action(
        self,
        workflow: AutomationWorkflow,
        action: AutomationAction,
    ):
        action.completed = False

        action_type = action.action_type
        parameters = action.parameters

        if action_type == "resource_production":
            self._resource_production(
                parameters.get("resource"),
                parameters.get("amount", 0),
            )

        elif action_type == "citizen_allocation":
            self._citizen_allocation(
                parameters.get("occupation"),
                parameters.get("count", 0),
            )

        elif action_type == "employment":
            self._balance_employment()

        elif action_type == "analysis":
            self._analyze_employment()

        elif action_type == "research":
            self._identify_researchers()

        elif action_type == "knowledge":
            self._generate_knowledge()

        elif action_type == "memory":
            self.memory_engine.remember(
                citizen_id="civilization",
                content=parameters.get(
                    "content",
                    f"AETHER completed {workflow.name}.",
                ),
                memory_type="procedural",
                importance=0.8,
                tags=[
                    "automation",
                    workflow.id,
                ],
            )

        else:
            raise ValueError(
                f"Unknown automation action type: {action_type}"
            )

        action.completed = True

    # ---------------------------------------------------------
    # RESOURCE ACTIONS
    # ---------------------------------------------------------

    def _resource_production(
        self,
        resource_name: str,
        amount: float,
    ):
        world = self.world_engine.get_world()

        resource = next(
            (
                item
                for item in world.resources
                if item.name == resource_name
            ),
            None,
        )

        if resource is None:
            return

        resource.amount = min(
            resource.capacity,
            resource.amount + amount,
        )

        world.events.append(
            f"Automation increased {resource_name} "
            f"by {amount:.0f} units."
        )

        world.events = world.events[-50:]

    def _citizen_allocation(
        self,
        occupation: str,
        count: int,
    ):
        citizens = self.citizen_engine.get_all()

        candidates = [
            citizen
            for citizen in citizens
            if citizen.alive
            and citizen.occupation.lower()
            in {"unemployed", "jobless"}
        ]

        assigned = 0

        for citizen in candidates[:count]:
            citizen.occupation = occupation
            citizen.current_action = (
                f"working as {occupation}"
            )
            assigned += 1

        self.world_engine.get_world().events.append(
            f"Automation assigned {assigned} citizens "
            f"to {occupation}."
        )

    # ---------------------------------------------------------
    # EMPLOYMENT
    # ---------------------------------------------------------

    def _analyze_employment(self):
        citizens = self.citizen_engine.get_all()

        occupation_counts: dict[str, int] = {}

        for citizen in citizens:
            if not citizen.alive:
                continue

            occupation = citizen.occupation

            occupation_counts[occupation] = (
                occupation_counts.get(occupation, 0) + 1
            )

        return occupation_counts

    def _balance_employment(self):
        citizens = self.citizen_engine.get_all()

        unemployed = [
            citizen
            for citizen in citizens
            if citizen.alive
            and citizen.occupation.lower()
            in {"unemployed", "jobless"}
        ]

        opportunities = [
            "farmer",
            "engineer",
            "researcher",
            "builder",
            "trader",
        ]

        for index, citizen in enumerate(unemployed):
            occupation = opportunities[
                index % len(opportunities)
            ]

            citizen.occupation = occupation
            citizen.current_action = (
                f"working as {occupation}"
            )

        self.world_engine.get_world().events.append(
            f"Automation balanced employment for "
            f"{len(unemployed)} citizens."
        )

    # ---------------------------------------------------------
    # KNOWLEDGE
    # ---------------------------------------------------------

    def _identify_researchers(self):
        return [
            citizen
            for citizen in self.citizen_engine.get_all()
            if citizen.alive
            and any(
                keyword in citizen.occupation.lower()
                for keyword in [
                    "research",
                    "scientist",
                    "engineer",
                    "scholar",
                ]
            )
        ]

    def _generate_knowledge(self):
        world = self.world_engine.get_world()

        knowledge = next(
            (
                resource
                for resource in world.resources
                if resource.name == "knowledge"
            ),
            None,
        )

        if knowledge:
            knowledge.amount = min(
                knowledge.capacity,
                knowledge.amount + 5,
            )

        discoveries = [
            "AETHER researchers identified a new optimization pattern.",
            "AETHER researchers discovered a new agricultural technique.",
            "AETHER researchers identified an energy efficiency opportunity.",
            "AETHER researchers discovered a new social coordination pattern.",
            "AETHER researchers developed a new theoretical insight.",
        ]

        index = len(self.events) % len(discoveries)
        discovery = discoveries[index]

        world.events.append(discovery)
        world.events = world.events[-50:]

        return discovery

    # ---------------------------------------------------------
    # EVENT SYSTEM
    # ---------------------------------------------------------

    def _add_event(
        self,
        event_type: str,
        title: str,
        description: str,
        severity: str,
        workflow_id: str | None = None,
        data: dict[str, Any] | None = None,
    ):
        event = AutomationEvent(
            id=f"automation_event_{len(self.events) + 1:06d}",
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            title=title,
            description=description,
            severity=severity,
            workflow_id=workflow_id,
            data=data or {},
        )

        self.events.append(event)

        # Keep the event stream bounded.
        self.events = self.events[-200:]

        return event

    # ---------------------------------------------------------
    # PUBLIC STATE
    # ---------------------------------------------------------

    def get_status(self):
        return {
            "triggers": len(self.triggers),
            "workflows": len(self.workflows),
            "events": len(self.events),
            "active_workflows": sum(
                1
                for workflow in self.workflows
                if workflow.status == AutomationStatus.ACTIVE
            ),
            "cooldowns": self.get_cooldown_status(),
        }

    def get_workflows(self):
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "trigger_id": workflow.trigger_id,
                "priority": workflow.priority,
                "status": workflow.status.value,
                "actions": [
                    {
                        "id": action.id,
                        "name": action.name,
                        "action_type": action.action_type,
                        "description": action.description,
                        "parameters": action.parameters,
                        "completed": action.completed,
                    }
                    for action in workflow.actions
                ],
                "created_at": workflow.created_at,
                "started_at": workflow.started_at,
                "completed_at": workflow.completed_at,
                "result": workflow.result,
                "cooldown_remaining_seconds": self._cooldown_remaining(
                    workflow.id
                ),
            }
            for workflow in self.workflows
        ]

    def get_events(self):
        return self.events

    def _cooldown_remaining(self, workflow_id: str):
        expires_at = self.cooldowns.get(workflow_id)

        if expires_at is None:
            return 0

        remaining = (
            expires_at - datetime.now()
        ).total_seconds()

        return max(0, int(remaining))