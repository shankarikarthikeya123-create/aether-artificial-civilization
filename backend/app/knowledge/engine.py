import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import KnowledgeDocument


class KnowledgeEngine:
    """
    Persistent local knowledge base for AETHER.

    Provides:
    - SQLite persistence
    - knowledge ingestion
    - lexical retrieval
    - relevance scoring
    - RAG context construction
    - knowledge statistics
    """

    def __init__(self, database_path: str | None = None):
        if database_path is None:
            database_path = str(
                Path(__file__).resolve().parents[2] / "aether_knowledge.db"
            )

        self.database_path = database_path

        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._create_tables()
        self._seed_knowledge()

    # ============================================================
    # DATABASE
    # ============================================================

    def _create_tables(self):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                knowledge_type TEXT NOT NULL,
                tags TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        self.connection.commit()

    # ============================================================
    # SEED KNOWLEDGE
    # ============================================================

    def _seed_knowledge(self):
        cursor = self.connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) AS count FROM knowledge"
        )

        count = cursor.fetchone()["count"]

        if count > 0:
            return

        initial_documents = [
            {
                "id": "knowledge_aether_001",
                "title": "AETHER Civilization Overview",
                "content": (
                    "AETHER is an artificial civilization containing autonomous "
                    "citizens, locations, resources, institutions, events and "
                    "decision-making systems. Citizens perceive their environment, "
                    "retrieve relevant knowledge and memories, reason about "
                    "situations, create plans, perform actions and learn from "
                    "outcomes."
                ),
                "source": "aether_core",
                "knowledge_type": "civilization",
                "tags": ["aether", "civilization", "citizens", "agents"],
                "importance": 1.0,
            },
            {
                "id": "knowledge_ai_agents_001",
                "title": "Intelligent Agent Architecture",
                "content": (
                    "An intelligent agent perceives an environment through available "
                    "information, maintains an internal state, reasons about goals, "
                    "selects actions and acts upon the environment. AETHER citizens "
                    "use perception, memory, reasoning, planning and action."
                ),
                "source": "ai_knowledge",
                "knowledge_type": "artificial_intelligence",
                "tags": ["agent", "perception", "reasoning", "action"],
                "importance": 0.95,
            },
            {
                "id": "knowledge_rag_001",
                "title": "Retrieval Augmented Generation",
                "content": (
                    "Retrieval Augmented Generation combines information retrieval "
                    "with language generation. Relevant documents are retrieved "
                    "before an LLM generates an answer. This allows the model to "
                    "use external knowledge instead of depending entirely on its "
                    "internal parameters."
                ),
                "source": "ai_knowledge",
                "knowledge_type": "rag",
                "tags": ["rag", "retrieval", "llm", "generation"],
                "importance": 0.95,
            },
            {
                "id": "knowledge_planning_001",
                "title": "AI Planning",
                "content": (
                    "Planning determines a sequence of actions that can transform "
                    "a current state into a desired goal state. A plan may contain "
                    "multiple actions and should consider available resources, "
                    "constraints and expected outcomes."
                ),
                "source": "ai_knowledge",
                "knowledge_type": "planning",
                "tags": ["planning", "goals", "actions", "state"],
                "importance": 0.9,
            },
            {
                "id": "knowledge_search_001",
                "title": "A Star Search",
                "content": (
                    "A* search combines the cost already spent to reach a state "
                    "with an estimated cost from that state to the goal. The "
                    "evaluation function is f(n) = g(n) + h(n). It can efficiently "
                    "find paths when the heuristic provides useful guidance."
                ),
                "source": "ai_knowledge",
                "knowledge_type": "search",
                "tags": ["astar", "search", "heuristic", "pathfinding"],
                "importance": 0.85,
            },
            {
                "id": "knowledge_bayesian_001",
                "title": "Bayesian Reasoning",
                "content": (
                    "Bayesian reasoning updates the probability of a hypothesis "
                    "when new evidence becomes available. The posterior probability "
                    "depends on the prior probability and the likelihood of the "
                    "observed evidence."
                ),
                "source": "ai_knowledge",
                "knowledge_type": "probabilistic_reasoning",
                "tags": ["bayesian", "probability", "inference", "evidence"],
                "importance": 0.85,
            },
            {
                "id": "knowledge_learning_001",
                "title": "Learning From Experience",
                "content": (
                    "An intelligent system can improve by recording outcomes of "
                    "previous actions and using those experiences when making "
                    "future decisions. Important successful and unsuccessful "
                    "experiences should become persistent memories."
                ),
                "source": "ai_knowledge",
                "knowledge_type": "learning",
                "tags": ["learning", "experience", "memory", "adaptation"],
                "importance": 0.9,
            },
            {
                "id": "knowledge_resources_001",
                "title": "AETHER Resource Management",
                "content": (
                    "AETHER manages food, water, energy, money and knowledge. "
                    "Civilization stability depends on balancing resource "
                    "production, consumption, storage capacity and population needs."
                ),
                "source": "aether_core",
                "knowledge_type": "resources",
                "tags": ["resources", "food", "water", "energy", "economy"],
                "importance": 0.9,
            },
        ]

        for document in initial_documents:
            self.add_document(
                title=document["title"],
                content=document["content"],
                source=document["source"],
                knowledge_type=document["knowledge_type"],
                tags=document["tags"],
                importance=document["importance"],
                document_id=document["id"],
            )

    # ============================================================
    # TOKENIZATION
    # ============================================================

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        words = re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            text.lower(),
        )

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "is",
            "are",
            "was",
            "were",
            "be",
            "this",
            "that",
            "from",
            "by",
            "as",
            "it",
            "its",
        }

        return {
            word
            for word in words
            if word not in stop_words
        }

    # ============================================================
    # ADD KNOWLEDGE
    # ============================================================

    def add_document(
        self,
        title: str,
        content: str,
        source: str = "aether",
        knowledge_type: str = "general",
        tags: list[str] | None = None,
        importance: float = 0.5,
        document_id: str | None = None,
    ) -> KnowledgeDocument:

        if document_id is None:
            document_id = (
                f"knowledge_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            )

        created_at = datetime.now().isoformat()

        document = KnowledgeDocument(
            id=document_id,
            title=title,
            content=content,
            source=source,
            knowledge_type=knowledge_type,
            tags=tags or [],
            importance=importance,
            created_at=created_at,
        )

        self.connection.execute(
            """
            INSERT OR REPLACE INTO knowledge
            (
                id,
                title,
                content,
                source,
                knowledge_type,
                tags,
                importance,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.title,
                document.content,
                document.source,
                document.knowledge_type,
                ",".join(document.tags),
                document.importance,
                document.created_at,
            ),
        )

        self.connection.commit()

        return document

    # ============================================================
    # GET DOCUMENTS
    # ============================================================

    def get_all(self) -> list[KnowledgeDocument]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM knowledge
            ORDER BY importance DESC, created_at DESC
            """
        ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: str):
        row = self.connection.execute(
            """
            SELECT *
            FROM knowledge
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_document(row)

    def count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM knowledge"
        ).fetchone()

        return int(row["count"])

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        documents = self.get_all()

        scored_documents = []

        for document in documents:

            document_text = " ".join(
                [
                    document.title,
                    document.content,
                    document.source,
                    document.knowledge_type,
                    " ".join(document.tags),
                ]
            )

            document_tokens = self._tokenize(document_text)

            overlap = query_tokens.intersection(document_tokens)

            if not overlap:
                continue

            # Basic TF-style relevance
            overlap_score = len(overlap) / max(
                len(query_tokens),
                1,
            )

            # Importance influences ranking
            importance_score = document.importance * 0.25

            # Title/tag matches are stronger
            title_tokens = self._tokenize(document.title)
            tag_tokens = self._tokenize(
                " ".join(document.tags)
            )

            title_matches = len(
                query_tokens.intersection(title_tokens)
            )

            tag_matches = len(
                query_tokens.intersection(tag_tokens)
            )

            title_score = min(title_matches * 0.20, 0.40)
            tag_score = min(tag_matches * 0.10, 0.20)

            score = (
                overlap_score
                + importance_score
                + title_score
                + tag_score
            )

            scored_documents.append(
                {
                    "document": document,
                    "score": round(score, 4),
                    "matched_terms": sorted(overlap),
                }
            )

        scored_documents.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored_documents[:limit]

    # ============================================================
    # RAG CONTEXT
    # ============================================================

    def build_rag_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        results = self.retrieve(
            query=query,
            limit=limit,
        )

        if not results:
            return "No relevant knowledge was retrieved."

        sections = []

        for index, result in enumerate(results, start=1):
            document = result["document"]

            sections.append(
                f"""
KNOWLEDGE {index}
Title: {document.title}
Type: {document.knowledge_type}
Source: {document.source}
Relevance: {result["score"]:.3f}
Matched terms: {", ".join(result["matched_terms"])}

{document.content}
""".strip()
            )

        return "\n\n".join(sections)

    # ============================================================
    # RAG QUERY
    # ============================================================

    def query_rag(
        self,
        query: str,
        limit: int = 5,
    ) -> dict[str, Any]:

        results = self.retrieve(
            query=query,
            limit=limit,
        )

        return {
            "query": query,
            "results": [
                {
                    "id": item["document"].id,
                    "title": item["document"].title,
                    "content": item["document"].content,
                    "source": item["document"].source,
                    "knowledge_type": item["document"].knowledge_type,
                    "importance": item["document"].importance,
                    "score": item["score"],
                    "matched_terms": item["matched_terms"],
                }
                for item in results
            ],
            "context": self.build_rag_context(
                query=query,
                limit=limit,
            ),
        }

    # ============================================================
    # LIVE CIVILIZATION KNOWLEDGE
    # ============================================================

    def sync_live_knowledge(self, world, citizens, events=None, decisions=None):
        """Continuously refresh knowledge from the current AETHER state."""
        events = events or []
        decisions = decisions or []
        resources = ", ".join(f"{r.name}={r.amount:.1f}/{r.capacity:.1f}" for r in world.resources)

        self.add_document(
            title="AETHER Live World State",
            content=(
                f"AETHER currently has population {world.population}. "
                f"Live resources: {resources}. "
                f"World time: day {world.day}, {world.hour:02d}:{world.minute:02d}."
            ),
            source="live_world", knowledge_type="world_state",
            tags=["live","world","resources","population"],
            importance=1.0, document_id="live_world_state",
        )

        citizen_lines = []
        for citizen in citizens[:100]:
            citizen_lines.append(
                f"{citizen.name} ({citizen.id}) is a {citizen.occupation} "
                f"in {citizen.location_id}; hunger={citizen.needs.hunger:.1f}, "
                f"energy={citizen.needs.energy:.1f}, social={citizen.needs.social:.1f}, "
                f"money={citizen.needs.money:.1f}; current action={citizen.current_action}."
            )
        self.add_document(
            title="AETHER Citizen State",
            content=" ".join(citizen_lines) or "No citizens are currently active.",
            source="live_citizens", knowledge_type="citizens",
            tags=["live","citizens","population","needs","actions"],
            importance=0.95, document_id="live_citizen_state",
        )

        event_text = " ".join(
            f"[{e.get('type','event')}] {e.get('message','')}" for e in events[-15:]
        )
        self.add_document(
            title="AETHER Recent Civilization Events",
            content=event_text or "No recent civilization events recorded.",
            source="live_events", knowledge_type="events",
            tags=["live","events","history","civilization"],
            importance=0.9, document_id="live_events",
        )

        if decisions:
            decision_text = " ".join(
                f"{d.get('citizen','Citizen')} chose {d.get('action','unknown')}: {d.get('result','')}"
                for d in decisions[-20:]
            )
            self.add_document(
                title="AETHER Recent AI Decisions",
                content=decision_text,
                source="live_decisions", knowledge_type="ai_decisions",
                tags=["live","ai","decisions","reasoning","actions"],
                importance=0.9, document_id="live_decisions",
            )

    def answer_question(self, query: str, world=None, citizens=None, events=None, decisions=None):
        """Answer a live civilization question with grounded data first, then local AI."""
        self.sync_live_knowledge(world, citizens or [], events or [], decisions or [])
        citizens = citizens or []
        events = events or []
        decisions = decisions or []
        q = query.lower().strip()
        resources = {r.name.lower(): r for r in (world.resources if world else [])}

        if world is not None and any(k in q for k in ["population", "how many citizens", "number of citizens"]):
            answer = f"AETHER currently has {world.population} citizens."
        elif world is not None and any(k in q for k in ["resource", "food", "water", "energy", "money"]):
            if any(k in q for k in ["food", "water", "energy", "money"]):
                names = [k for k in ["food", "water", "energy", "money", "knowledge"] if k in q]
                selected = [resources[n] for n in names if n in resources] or list(resources.values())
            else:
                selected = list(resources.values())
            answer = "Current AETHER resources: " + "; ".join(
                f"{r.name} {r.amount:.1f}/{r.capacity:.1f}" for r in selected
            ) + "."
        elif citizens and any(k in q for k in ["work", "works", "job", "do", "does", "occupation", "role"]):
            # Direct citizen facts must not depend on lexical RAG or the LLM.
            # Resolve the named citizen first, then answer from the live object.
            normalized_q = re.sub(r"[^a-z0-9_ ]+", " ", q)
            named = []
            for c in citizens:
                name_tokens = self._tokenize(c.name)
                if name_tokens and name_tokens.issubset(set(normalized_q.split())):
                    named.append(c)
            if not named:
                # Also support a citizen ID such as citizen_002.
                named = [c for c in citizens if c.id.lower() in q]

            if named:
                c = named[0]
                answer = (
                    f"{c.name} currently works as a {c.occupation} in {c.location_id}. "
                    f"Their current action is {c.current_action}."
                )
            else:
                terms = self._tokenize(q)
                matches = []
                for c in citizens:
                    hay = f"{c.name} {c.id} {c.occupation} {c.location_id}".lower()
                    if any(term in hay for term in terms):
                        matches.append(f"{c.name} ({c.occupation}, {c.location_id})")
                answer = (
                    "Matching citizens: " + ", ".join(matches[:12]) + "."
                    if matches else f"AETHER has {len(citizens)} citizens; no citizen matched that query exactly."
                )
        elif citizens and any(k in q for k in ["citizen", "people", "who", "worker", "occupation"]):
            terms = self._tokenize(q)
            matches = []
            for c in citizens:
                hay = f"{c.name} {c.id} {c.occupation} {c.location_id}".lower()
                if any(term in hay for term in terms):
                    matches.append(f"{c.name} ({c.occupation}, {c.location_id})")
            answer = (
                "Matching citizens: " + ", ".join(matches[:12]) + "."
                if matches else f"AETHER has {len(citizens)} citizens; no citizen matched that query exactly."
            )
        elif any(k in q for k in ["event", "happened", "history", "recent"]):
            recent = events[-8:]
            answer = "Recent AETHER events: " + " | ".join(
                str(e.get("message", e)) for e in recent
            ) if recent else "AETHER has no recorded events yet."
        elif any(k in q for k in ["decision", "chose", "action", "agent"]):
            recent = decisions[-8:]
            answer = "Recent AI decisions: " + " | ".join(
                f"{d.get('citizen','Citizen')} chose {d.get('action','unknown')} ({d.get('result','')})"
                for d in recent
            ) if recent else "AETHER has no recorded AI decisions yet."
        else:
            results = self.query_rag(query, limit=8)
            context = results["context"]
            try:
                from ..llm import llm
                prompt = f"""You are AETHER's Knowledge Intelligence.
Answer ONLY from the supplied AETHER context. Never invent civilization facts.
QUESTION:
{query}
AETHER CONTEXT:
{context}
Give a concise grounded answer and mention uncertainty when context is insufficient."""
                answer = llm.generate(prompt)

                # Keep the answer understandable even when the local/cloud
                # fallback returns its internal ACTION / REASON / PLAN format.
                lines = [line.strip() for line in str(answer).splitlines() if line.strip()]
                fields = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip().upper()] = value.strip()

                if "ACTION" in fields and ("REASON" in fields or "PLAN" in fields):
                    action = fields["ACTION"].replace("_", " ")
                    reason = fields.get("REASON", "AETHER found this to be the safest useful next step.")
                    plan = fields.get("PLAN")
                    learning = fields.get("LEARNING")

                    answer = f"AETHER recommends **{action}**. {reason}"
                    if plan:
                        answer += f" {plan}"
                    if learning:
                        answer += f" {learning}"
            except Exception:
                answer = "The retrieved AETHER knowledge does not contain enough information to answer that question."

        results = self.query_rag(query, limit=8)
        return {
            "query": query, "answer": answer,
            "sources": results["results"], "context": results["context"],
            "knowledge_count": self.count(),
        }

    # ============================================================
    # DATABASE STATS
    # ============================================================

    def statistics(self) -> dict[str, Any]:

        total = self.count()

        type_rows = self.connection.execute(
            """
            SELECT knowledge_type, COUNT(*) AS count
            FROM knowledge
            GROUP BY knowledge_type
            ORDER BY count DESC
            """
        ).fetchall()

        source_rows = self.connection.execute(
            """
            SELECT source, COUNT(*) AS count
            FROM knowledge
            GROUP BY source
            ORDER BY count DESC
            """
        ).fetchall()

        return {
            "total_documents": total,
            "types": {
                row["knowledge_type"]: row["count"]
                for row in type_rows
            },
            "sources": {
                row["source"]: row["count"]
                for row in source_rows
            },
            "database": self.database_path,
        }

    # ============================================================
    # DELETE
    # ============================================================

    def delete_document(self, document_id: str) -> bool:

        cursor = self.connection.execute(
            """
            DELETE FROM knowledge
            WHERE id = ?
            """,
            (document_id,),
        )

        self.connection.commit()

        return cursor.rowcount > 0

    # ============================================================
    # INTERNAL CONVERSION
    # ============================================================

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> KnowledgeDocument:

        tags = [
            tag.strip()
            for tag in row["tags"].split(",")
            if tag.strip()
        ]

        return KnowledgeDocument(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            source=row["source"],
            knowledge_type=row["knowledge_type"],
            tags=tags,
            importance=row["importance"],
            created_at=row["created_at"],
        )

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self):
        self.connection.close()


knowledge_engine = KnowledgeEngine()