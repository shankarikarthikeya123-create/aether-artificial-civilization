# AETHER — The Digital Mind

AETHER is a local-first artificial civilization with 100 autonomous citizens living in a simulated world.

## Cognitive loop

Perceive -> Remember -> Retrieve -> Reason -> Plan -> Decide -> Act -> Learn

## Core stack

- React + TypeScript + Vite
- FastAPI + Python
- SQLite persistent memory and learning
- RAG knowledge retrieval
- symbolic forward chaining
- UCS, A* and Best-First Search
- Bayesian inference
- state-space planning
- local Ollama LLM integration with deterministic fallback
- Q-Learning
- automation workflows
- multi-agent civilization council
- MCP tools

## Local run

Backend:

    cd backend
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000

Frontend:

    cd frontend
    npm install
    npm run dev

Open http://localhost:5173

## Publishing

Deploy `backend/` as a Render Web Service using `render.yaml`. Deploy `frontend/` to Netlify using `netlify.toml`. Set the Netlify environment variable `VITE_API_URL` to the public Render API URL.

The SQLite files are suitable for a demo but free/ephemeral hosting should not be treated as permanent storage. For production, move persistent state to a managed database or persistent disk.

## Demo

1. COMMAND: show the live world and cognitive pipeline.
2. Click each pipeline node to inspect real evidence.
3. CITIZENS: inspect autonomous minds.
4. COGNITION: run a situation through the reasoning engine.
5. KNOWLEDGE: ask a civilization question and inspect retrieved context.
6. INTELLIGENCE: show learning, events and strategy.
7. Run the 30-day civilization experiment.

## Viva explanation

AETHER is not one neural network. It is an integrated cognitive architecture. Perception turns world state into needs; memory retrieves experience; RAG grounds reasoning; symbolic rules derive facts; search evaluates paths; Bayesian inference handles uncertainty; planning creates goal-directed steps; the decision layer chooses an action; action changes the simulated world; learning stores the outcome and updates Q-values.
