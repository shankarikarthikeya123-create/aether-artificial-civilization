"""AETHER context orchestration with optional LangChain acceleration."""
from typing import Any

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    LANGCHAIN_AVAILABLE = True
except ImportError:
    ChatPromptTemplate = None
    RunnableLambda = None
    LANGCHAIN_AVAILABLE = False

def build_cognitive_context(*, situation: str, world: Any, memories: list[Any], knowledge: str, rules: list[str]) -> dict[str, Any]:
    packet = {
        "situation": str(situation),
        "world": world,
        "memories": [getattr(m, "content", str(m)) for m in memories],
        "knowledge": str(knowledge or ""),
        "rules": [str(r) for r in rules],
    }
    if LANGCHAIN_AVAILABLE:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are the orchestration layer of AETHER. Ground every decision in supplied live state."),
            ("human", "Situation: {situation}\nWorld: {world}\nMemories: {memories}\nKnowledge: {knowledge}\nRules: {rules}"),
        ])
        normalize = RunnableLambda(lambda x: x)
        packet = normalize.invoke(packet)
        rendered = prompt_template.format_messages(**packet)
        prompt = rendered[-1].content
        system = rendered[0].content
        framework = "LangChain"
    else:
        system = "AETHER orchestration layer: ground every decision in supplied live state."
        prompt = f"Situation: {packet['situation']}\nWorld: {packet['world']}\nMemories: {packet['memories']}\nKnowledge: {packet['knowledge']}\nRules: {packet['rules']}"
        framework = "LangChain-compatible fallback (package pending)"
    return {"packet": packet, "prompt": prompt, "system": system, "framework": framework, "langchain_installed": LANGCHAIN_AVAILABLE}
