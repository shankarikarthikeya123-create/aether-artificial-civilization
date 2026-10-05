import ollama


class LocalLLM:
    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model

    def generate(self, prompt: str) -> str:
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

        content = response["message"].get("content", "").strip()

        return content or "No reasoning output was generated."


llm = LocalLLM()