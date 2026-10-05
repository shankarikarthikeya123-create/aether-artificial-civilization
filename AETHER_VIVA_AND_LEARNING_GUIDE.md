# AETHER — Learn It Before You Present It

## 1. What is AETHER?

AETHER is an artificial civilization simulation. It contains 100 citizens, a world state, resources, locations, memories, knowledge, decisions, actions and learning.

The important idea is integration. AETHER is not just a chatbot and not just a simulation.

Its cognitive loop is:

PERCEIVE -> REMEMBER -> RETRIEVE -> REASON -> PLAN -> DECIDE -> ACT -> LEARN

## 2. Perception

The citizen observes internal needs and external world state.

Examples:
- hunger
- energy
- social need
- money pressure
- safety
- food/water/energy resources
- location
- occupation

Code: backend/app/simulation.py

## 3. Memory

A citizen retrieves previous experiences relevant to the current need.

Memory answers:
"What have I experienced before that is relevant now?"

AETHER persists memory in SQLite.

## 4. RAG

RAG means Retrieval-Augmented Generation.

AETHER searches its knowledge store for civilization knowledge and live world information before reasoning.

It prevents reasoning from depending only on a language model's internal knowledge.

The live stream exposes:
- query
- retrieved documents
- matched terms
- scores

Code: backend/app/knowledge/engine.py

## 5. Symbolic reasoning

Rules derive new facts from existing facts.

Example:

food_shortage -> food_crisis
citizen_hungry -> food_required

This is deterministic reasoning. It is useful because we can inspect exactly why a fact was derived.

Code: backend/app/reasoning/logic.py and fai_engine.py

## 6. Search algorithms

AETHER demonstrates several classical AI search methods.

Uniform Cost Search:
Finds a minimum-cost path.

A*:
Uses path cost plus a heuristic to guide search.

Best-First:
Uses heuristic information to prioritize promising nodes.

The live trace exposes the path, cost and expanded nodes.

Code: backend/app/reasoning/search.py

## 7. Bayesian reasoning

Bayesian inference estimates probabilities instead of returning only yes/no.

AETHER uses evidence to compare states such as:
- shortage
- normal

The live trace shows the posterior distribution and most likely state.

Code: backend/app/reasoning/bayesian.py

## 8. Planning

Planning converts a goal into steps.

Example:

Goal: satisfy hunger

Plan:
1. Locate available food
2. Acquire food
3. Consume food
4. Update hunger state

The project also contains state-space planning.

Code: backend/app/reasoning/planner.py and simulation.py

## 9. Decision

The decision layer selects an action such as:
- eat
- rest
- socialize
- work
- research
- explore

The system can use citizen reasoning, deterministic fallbacks and learning signals.

## 10. Action

An action changes the simulated civilization.

Examples:
- eating reduces hunger and consumes food
- working changes money/resources
- research increases knowledge
- socializing changes relationships/social state
- resting changes energy

This is important: the AI is connected to an environment. It is not only generating text.

## 11. Learning

After an action, AETHER records the experience and updates Q-values.

A Q-learning record contains:
state -> action -> reward -> next state

Example:

hunger_low | energy_high | social_high
    -> work
    -> reward
    -> next state

The learning store persists experiences and Q-values in SQLite.

Code: backend/app/learning_store.py

## 12. Multi-agent council

AETHER also has specialized civilization-level roles:
- resource agent
- citizen agent
- planning agent
- governance agent
- learning agent

They inspect different aspects of the same civilization and produce a consensus.

Code: backend/app/integrations/multi_agent.py

## 13. LangChain

LangChain is used as an orchestration/context layer.

It is not the intelligence by itself.

Its job here is to organize:
- situation
- world state
- memories
- knowledge
- rules

Code: backend/app/integrations/orchestration.py

## 14. MCP

MCP exposes AETHER capabilities as tools.

Current tools include:
- get_world_state
- get_citizen
- search_aether_knowledge
- get_civilization_summary
- get_resources
- inspect_location
- execute_action

This makes the civilization externally tool-accessible.

Code: backend/app/integrations/mcp_server.py and mcp_tools.py

## 15. Why the live cognitive stream matters

The stream is not fake UI text.

It comes from the backend's cognitive trace.

Each tick can expose:

PERCEPTION
MEMORY
RAG
REASONING
PLANNING
DECISION
ACTION
LEARNING

Clicking a stage reveals evidence from that stage.

## 16. Best demo

Open COMMAND.

Show:
1. 100 citizens
2. live map
3. resources
4. cognitive pipeline
5. live cognitive evidence

Click:
PERCEPTION -> show needs and world state
MEMORY -> show retrieved experiences
RAG -> show knowledge documents
REASONING -> show symbolic/search/Bayesian evidence
PLANNING -> show goal and plan
DECISION -> show selected action
ACTION -> show world-changing result
LEARNING -> show persistent experiences and Q-state count

Then open COGNITION and run:

"I am hungry and need to find food."

Explain how the situation moves through the same architecture.

Finally run the 30-day civilization experiment and show learning persistence.

## 17. Viva questions you should be able to answer

Q: Is AETHER a single AI model?
A: No. It is an integrated cognitive architecture combining symbolic AI, search, probability, planning, memory, RAG, optional local LLM reasoning and reinforcement learning.

Q: Why use symbolic reasoning if an LLM can reason?
A: Symbolic rules are deterministic, inspectable and reproducible. They provide explicit evidence for conclusions.

Q: Why use Bayesian inference?
A: Real decisions involve uncertainty. Bayesian reasoning represents confidence rather than only a binary conclusion.

Q: Why use A*?
A: A* efficiently finds low-cost paths when a useful heuristic is available.

Q: What does RAG add?
A: It grounds reasoning in retrieved civilization knowledge and live state.

Q: What does Q-learning add?
A: It lets the system learn from action outcomes instead of treating every decision as independent.

Q: What makes this an agent?
A: It observes an environment, maintains internal state, reasons, chooses actions, acts on the environment and learns from the result.

Q: What is the strongest engineering idea?
A: The cognitive loop and persistent state connecting all the components.

## 18. Important honesty point

Do not claim that every decision is produced by the local LLM.

The system contains LLM integration, but it also has deterministic reasoning and fallback decision mechanisms. The correct explanation is that AETHER is a hybrid cognitive architecture.

## 19. Important files

Frontend:
frontend/src/App.tsx
frontend/src/App.css

Backend:
backend/app/simulation.py
backend/app/cognition/
backend/app/memory/
backend/app/knowledge/
backend/app/reasoning/
backend/app/learning_store.py
backend/app/integrations/

Entry point:
backend/app/main.py

Deployment:
render.yaml
netlify.toml

## 20. Final mental model

Think of one citizen like this:

I SEE something.
-> I REMEMBER relevant experiences.
-> I RETRIEVE knowledge.
-> I REASON about the situation.
-> I PLAN what could solve it.
-> I DECIDE.
-> I ACT.
-> I OBSERVE the result.
-> I LEARN.

Then repeat this across 100 citizens inside one shared civilization.

That is AETHER.
