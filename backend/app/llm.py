import os
import ollama


class LocalLLM:
    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model
        self.mode = "local"

    def _fallback(self, prompt: str) -> str:
        lower = prompt.lower()

        if (
            "hunger_high" in lower
            or "food_required" in lower
            or "hungry" in lower
        ):
            return """ACTION: find_food

REASON: The citizen's hunger is the highest immediate wellbeing concern.

PLAN: Find an available food source, obtain food, and restore hunger.

LEARNING: Prioritize food when hunger becomes a dominant need."""

        if (
            "energy_low_personal" in lower
            or "rest_required" in lower
            or "tired" in lower
        ):
            return """ACTION: rest

REASON: The citizen's energy is low, so recovery is the safest immediate action.

PLAN: Find a safe place, rest, and restore energy.

LEARNING: Rest when personal energy becomes critically low."""

        if (
            "social_low" in lower
            or "social_interaction_required" in lower
        ):
            return """ACTION: socialize

REASON: The citizen's social need is low, so social interaction supports wellbeing.

PLAN: Find a suitable nearby social setting and interact with other citizens.

LEARNING: Maintain social connections when the social need falls too low."""

        if "safety_low" in lower:
            return """ACTION: seek_safety

REASON: The citizen's safety is low, so reaching a safer location is the priority.

PLAN: Move toward a safer location and avoid unnecessary risk.

LEARNING: Prioritize safety when perceived safety becomes low."""

        return """ACTION: observe

REASON: No critical immediate need was detected, so continued observation is appropriate.

PLAN: Observe the environment, monitor needs, and remain ready to respond.

LEARNING: Continue monitoring the world when no urgent need dominates."""

    def generate(self, prompt: str) -> str:
        requested_mode = os.getenv(
            "AETHER_LLM_MODE",
            "auto"
        ).lower()

        if requested_mode != "fallback":
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    think=False,
                    options={
                        "num_predict": 256,
                    },
                )

                content = response["message"].get(
                    "content",
                    ""
                ).strip()

                if content:
                    self.mode = "local"
                    return content

            except Exception:
                if requested_mode == "local":
                    raise

        self.mode = "fallback"
        return self._fallback(prompt)


llm = LocalLLM()