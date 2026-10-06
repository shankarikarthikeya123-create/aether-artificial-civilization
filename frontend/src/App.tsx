import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

type Location = {
  id: string;
  name: string;
  x: number;
  y: number;
  location_type: string;
  population: number;
};

type Resource = {
  name: string;
  amount: number;
  capacity: number;
};

type World = {
  name: string;
  date: string;
  time: string;
  day_of_week: string;
  timezone: string;
  day: number;
  hour: number;
  minute: number;
  locations: Location[];
  resources: Resource[];
  population: number;
  events: string[];
};

type Citizen = {
  id: string;
  name: string;
  age: number;
  occupation: string;
  location_id: string;
  current_action: string;
  needs: {
    hunger: number;
    energy: number;
    social: number;
    money: number;
    safety: number;
  };
  goals: string[];
  beliefs: string[];
  skills: Record<string, number>;
  relationships: Record<string, number>;
  personality?: Record<string, number>;
  memories?: Memory[];
  current_plan?: string[];
  alive: boolean;
};

type SimulationStatus = {
  civilization: string;
  status: string;
  tick: number;
  day: number;
  hour: number;
  minute: number;
  population: number;
  resources: Record<
    string,
    {
      amount: number;
      capacity: number;
    }
  >;
  total_decisions: number;
  total_memories: number;
  total_relationships: number;
  total_discoveries: number;
  total_crises: number;
  total_career_changes: number;
  event_count: number;
};

type Memory = {
  id: string;
  citizen_id: string;
  content: string;
  memory_type: string;
  importance: number;
  timestamp: string;
  tags: string[];
};

type CognitiveStage = {
  stage: string;
  state: string;
  value: unknown;
};

type CognitiveStream = {
  tick: number;
  citizen: string;
  citizen_id?: string | null;
  stages: CognitiveStage[];
  trace?: Record<string, unknown>;
  world?: { population?: number; day?: number; time?: string; resources?: Record<string, number> };
};

type CognitionResult = {
  citizen?: Citizen;
  situation?: string;
  perception?: {
    citizen?: { id?: string; name?: string; occupation?: string; location?: string };
    needs?: { hunger?: number; energy?: number; social?: number; money?: number; safety?: number };
    world?: { day?: number; time?: string; population?: number; resources?: Record<string, { amount?: number; capacity?: number; percentage?: number }>; recent_events?: string[] };
  };
  memories?: Memory[];
  memory?: { retrieved?: Memory[]; new_memory?: Memory };
  knowledge?: string;
  rag?: { query?: string; results?: Array<Record<string, unknown>> };
  symbolic_reasoning?: {
    facts?: string[];
    new_facts?: Array<{ rule?: string; conditions?: string[]; conclusion?: string }>;
    goal?: string;
    plan?: { success?: boolean; goal?: string; steps?: Array<{ action?: string; description?: string; cost?: number }> };
    bayesian?: { posterior?: Record<string, number>; most_likely?: string };
    navigation?: { found?: boolean; path?: string[]; cost?: number | null };
  };
  bayesian_reasoning?: string;
  plan?: string[];
  decision?: { raw?: string; action?: string; reason?: string; plan?: string; learning?: string; scores?: Record<string, number> };
  action?: string;
  reasoning?: string;
  [key: string]: unknown;
};

type NavItem =
  | "command"
  | "citizens"
  | "cognition"
  | "events"
  | "knowledge"
  | "intelligence";

const emptyWorld: World = {
  name: "AETHER",
  date: "",
  time: "",
  day_of_week: "",
  timezone: "Asia/Kolkata",
  day: 1,
  hour: 0,
  minute: 0,
  locations: [],
  resources: [],
  population: 0,
  events: [],
};

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 0,
  }).format(value);
}

function percentage(amount: number, capacity: number) {
  if (!capacity) return 0;
  return Math.max(
    0,
    Math.min(100, (amount / capacity) * 100),
  );
}

function resourceIcon(name: string) {
  const key = name.toLowerCase();

  if (key.includes("food")) return "◉";
  if (key.includes("water")) return "≈";
  if (key.includes("energy")) return "ϟ";
  if (key.includes("money")) return "¤";
  if (key.includes("knowledge")) return "✦";

  return "◆";
}

function resourceEntries(
  resources: World["resources"] | undefined,
) {
  if (!resources) return [];

  if (Array.isArray(resources)) {
    return resources;
  }

  return Object.entries(resources).map(
    ([name, value]) => ({
      name,
      amount:
        typeof value === "object" &&
        value !== null &&
        "amount" in value
          ? Number(
              (value as { amount: number }).amount,
            )
          : Number(value),
      capacity:
        typeof value === "object" &&
        value !== null &&
        "capacity" in value
          ? Number(
              (value as { capacity: number }).capacity,
            )
          : 100,
    }),
  );
}

function WorldMap({
  locations,
  citizens,
  events,
  onLocationSelect,
}: {
  locations: Location[];
  citizens: Citizen[];
  events: string[];
  onLocationSelect: (location: Location) => void;
}) {
  const normalizedLocations = useMemo(() => {
    return locations.map((location) => {
      const rawX = Number(location.x ?? 50);
      const rawY = Number(location.y ?? 50);

      const x = rawX <= 1 ? rawX * 100 : rawX;
      const y = rawY <= 1 ? rawY * 100 : rawY;

      const type = String(
        location.location_type ?? "",
      ).toLowerCase();

      let nodeType = "food";

      if (
        type.includes("capital") ||
        type.includes("city") ||
        type.includes("central")
      ) {
        nodeType = "capital";
      } else if (
        type.includes("energy") ||
        type.includes("power")
      ) {
        nodeType = "energy";
      } else if (
        type.includes("water") ||
        type.includes("reservoir")
      ) {
        nodeType = "water";
      } else if (
        type.includes("medical") ||
        type.includes("hospital")
      ) {
        nodeType = "medical";
      }

      return {
        ...location,
        mapX: Math.max(
          7,
          Math.min(93, x),
        ),
        mapY: Math.max(
          10,
          Math.min(90, y),
        ),
        nodeType,
      };
    });
  }, [locations]);

  const connections = useMemo(() => {
    if (normalizedLocations.length < 2) {
      return [];
    }

    const center = normalizedLocations.reduce(
      (best, current) => {
        if (!best) return current;

        const bestDistance =
          Math.pow(best.mapX - 50, 2) +
          Math.pow(best.mapY - 50, 2);

        const currentDistance =
          Math.pow(current.mapX - 50, 2) +
          Math.pow(current.mapY - 50, 2);

        return currentDistance < bestDistance
          ? current
          : best;
      },
      normalizedLocations[0],
    );

    return normalizedLocations
      .filter(
        (location) =>
          location.id !== center.id,
      )
      .map((location) => {
        const dx =
          location.mapX - center.mapX;
        const dy =
          location.mapY - center.mapY;

        const distance = Math.sqrt(
          dx * dx + dy * dy,
        );

        const angle =
          (Math.atan2(dy, dx) * 180) /
          Math.PI;

        return {
          id: `${center.id}-${location.id}`,
          x: center.mapX,
          y: center.mapY,
          length: distance,
          angle,
        };
      });
  }, [normalizedLocations]);

  return (
    <section className="aether-map-section">
      <div className="aether-map-header">
        <div>
          <span className="eyebrow">
            AETHER TERRITORY
          </span>

          <h2>LIVE WORLD MAP</h2>
        </div>

        <div className="map-status">
          <span className="map-live-dot" />
          WORLD SYNCHRONIZED
        </div>
      </div>

      <div className="aether-map">
        <div className="map-grid" />
        <div className="map-scanline" />

        <div className="map-coordinates top-left">
          GRID // AETHER-01
        </div>

        <div className="map-coordinates bottom-right">
          {locations.length} ACTIVE LOCATIONS · {events.length} EVENTS
        </div>

        <div className="map-compass">
          <span>N</span>
          <div>+</div>
        </div>

        {connections.map((connection) => (
          <div
            key={connection.id}
            className="map-connection"
            style={{
              left: `${connection.x}%`,
              top: `${connection.y}%`,
              width: `${connection.length}%`,
              transform: `rotate(${connection.angle}deg)`,
              transformOrigin: "0 0",
            }}
          />
        ))}

        {normalizedLocations.map(
          (location) => (
            <div
              key={location.id}
              className={`map-node ${location.nodeType}`}
              style={{
                left: `${location.mapX}%`,
                top: `${location.mapY}%`,
              }}
              title={`${location.name} — ${location.location_type}`}
              role="button"
              tabIndex={0}
              onClick={() => onLocationSelect(location)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onLocationSelect(location);
              }}
            >
              <div className="map-node-pulse" />

              <div className="map-node-core">
                {location.nodeType ===
                "capital"
                  ? "◆"
                  : location.nodeType ===
                    "energy"
                    ? "ϟ"
                    : location.nodeType ===
                      "water"
                      ? "≈"
                      : location.nodeType ===
                        "medical"
                        ? "+"
                        : "●"}
              </div>

              <div className="map-node-label">
                <strong>
                  {location.name}
                </strong>

                <span>
                  {location.location_type}
                  {" · "}
                  {location.population} POP
                  {" · "}
                  {citizens.filter((citizen) => citizen.location_id === location.id && citizen.alive).length} ACTIVE
                </span>
              </div>
            </div>
          ),
        )}

        <div className="map-legend">
          <div>
            <span className="legend-dot capital-dot" />
            CAPITAL
          </div>

          <div>
            <span className="legend-dot energy-dot" />
            ENERGY
          </div>

          <div>
            <span className="legend-dot water-dot" />
            WATER
          </div>

          <div>
            <span className="legend-dot food-dot" />
            FOOD
          </div>
        </div>
      </div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: string | number;
  detail: string;
  icon: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-icon">
          {icon}
        </span>

        <span className="metric-label">
          {label}
        </span>
      </div>

      <div className="metric-value">
        {value}
      </div>

      <div className="metric-detail">
        {detail}
      </div>
    </div>
  );
}

function ResourceCard({
  resource,
}: {
  resource: Resource;
}) {
  const percent = percentage(
    resource.amount,
    resource.capacity,
  );

  return (
    <div className="resource-card">
      <div className="resource-header">
        <div className="resource-title">
          <span className="resource-symbol">
            {resourceIcon(resource.name)}
          </span>

          <span>
            {resource.name.toUpperCase()}
          </span>
        </div>

        <span className="resource-percent">
          {percent.toFixed(0)}%
        </span>
      </div>

      <div className="resource-number">
        {formatNumber(resource.amount)}
      </div>

      <div className="resource-capacity">
        / {formatNumber(resource.capacity)}
      </div>

      <div className="resource-bar">
        <div
          className="resource-fill"
          style={{
            width: `${percent}%`,
          }}
        />
      </div>
    </div>
  );
}

function CitizenCard({
  citizen,
  selected,
  onSelect,
}: {
  citizen: Citizen;
  selected: boolean;
  onSelect: () => void;
}) {
  const action =
    citizen.current_action || "idle";

  return (
    <button
      className={`citizen-card ${
        selected ? "selected" : ""
      }`}
      onClick={onSelect}
    >
      <div className="citizen-card-top">
        <div className="citizen-avatar">
          {citizen.name
            .split(" ")
            .map((part) => part[0])
            .join("")
            .slice(0, 2)}
        </div>

        <div className="citizen-identity">
          <strong>{citizen.name}</strong>

          <span>
            {citizen.age} ·{" "}
            {citizen.occupation}
          </span>
        </div>

        <span
          className={`alive-indicator ${
            citizen.alive
              ? "alive"
              : "dead"
          }`}
        />
      </div>

      <div className="citizen-action">
        <span>ACTION</span>
        <strong>
          {action.replaceAll("_", " ")}
        </strong>
      </div>

      <div className="citizen-needs">
        <NeedBar
          label="HUNGER"
          value={citizen.needs.hunger}
          inverse
        />

        <NeedBar
          label="ENERGY"
          value={citizen.needs.energy}
        />

        <NeedBar
          label="SOCIAL"
          value={citizen.needs.social}
        />
      </div>
    </button>
  );
}

function NeedBar({
  label,
  value,
  inverse = false,
}: {
  label: string;
  value: number;
  inverse?: boolean;
}) {
  const display = Math.max(
    0,
    Math.min(100, value),
  );

  return (
    <div className="need-row">
      <div className="need-label">
        <span>{label}</span>
        <span>
          {display.toFixed(0)}
        </span>
      </div>

      <div className="need-track">
        <div
          className={`need-fill ${
            inverse && display > 70
              ? "warning"
              : ""
          }`}
          style={{
            width: `${display}%`,
          }}
        />
      </div>
    </div>
  );
}

function CognitionPipeline({
  stages,
  activeStage,
  onStageSelect,
}: {
  stages?: CognitiveStage[];
  activeStage?: string | null;
  onStageSelect?: (stage: string) => void;
}) {
  const pipelineStages = [
    "PERCEPTION",
    "MEMORY",
    "RAG",
    "REASONING",
    "PLANNING",
    "DECISION",
    "ACTION",
    "LEARNING",
  ];

  return (
    <div className="cognition-pipeline live-pipeline">
      <div className="pipeline-signal" aria-hidden="true" />
      {pipelineStages.map((label, index) => {
        const liveStage = stages?.find((item) => item.stage === label);
        const isActive = activeStage === label;
        const isReady = Boolean(liveStage);
        return (
          <div className="pipeline-node-wrap" key={label}>
            <button
              className={"pipeline-stage " + (isActive ? "is-active" : "") + (isReady ? "is-live" : "")}
              onClick={() => onStageSelect?.(label)}
              type="button"
              title={liveStage ? formatCognitiveValue(liveStage.value) : "Awaiting live signal"}
            >
              <div className="pipeline-number">{String(index + 1).padStart(2, "0")}</div>
              <div className="pipeline-name">{label}</div>
              <span className="pipeline-status">{isActive ? "INSPECTING" : isReady ? "LIVE" : "READY"}</span>
            </button>
            {index !== pipelineStages.length - 1 && <div className="pipeline-arrow">→</div>}
          </div>
        );
      })}
    </div>
  );
}

function formatCognitiveValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function humanizeEvent(event: string): string {
  try {
    const parsed = JSON.parse(event) as {
      type?: string;
      message?: string;
      citizen?: string;
      occupation?: string;
      action?: string;
      result?: string;
      reason?: string;
      workflow_id?: string;
    };
    if (parsed.citizen && parsed.action) {
      const who = parsed.occupation ? parsed.citizen + " · " + parsed.occupation.replaceAll("_", " ") : parsed.citizen;
      const result = parsed.result ? " " + parsed.result : "";
      const why = parsed.reason ? " " + parsed.reason : "";
      return who + " → " + parsed.action.replaceAll("_", " ") + "." + result + why;
    }
    if (parsed.workflow_id) return (parsed.message ?? "Automation completed.") + " · Workflow: " + parsed.workflow_id;
    return parsed.message ?? event;
  } catch { return event; }
}

function humanizeCognitionTerm(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\\b\\w/g, (letter) => letter.toUpperCase());
}

function explainCognitiveStage(stage: string): string {
  const explanations: Record<string, string> = {
    PERCEPTION: "AETHER reads this citizen's needs, identity, location and the current civilization resources.",
    MEMORY: "AETHER recalls relevant experiences from this citizen's past so it does not reason from zero.",
    RAG: "AETHER searches its live knowledge base for facts about resources, the world and relevant civilization state.",
    REASONING: "AETHER applies rules, path-search algorithms and Bayesian probability to interpret the evidence.",
    PLANNING: "AETHER turns the situation into a sequence of practical steps that could reach a goal.",
    DECISION: "AETHER compares possible actions and selects the one that best matches the citizen's strongest current need.",
    ACTION: "The selected action changes the citizen or the shared civilization state.",
    LEARNING: "AETHER records the outcome so future decisions can learn from what happened.",
  };
  return explanations[stage] ?? "This stage contributes evidence to the citizen's cognitive loop.";
}

function App() {
  const [page, setPage] =
    useState<NavItem>("command");

  const [world, setWorld] =
    useState<World>(emptyWorld);

  const [status, setStatus] =
    useState<SimulationStatus | null>(
      null,
    );

  const [citizens, setCitizens] =
    useState<Citizen[]>([]);

  const [events, setEvents] =
    useState<string[]>([]);

  const [knowledge, setKnowledge] =
    useState<Array<{ id: string; title: string; content: string; source: string; knowledge_type: string; importance: number }>>([]);

  const [knowledgeQuestion, setKnowledgeQuestion] = useState("");
  const [knowledgeAnswer, setKnowledgeAnswer] = useState("");
  const [knowledgeSources, setKnowledgeSources] = useState<string[]>([]);
  const [knowledgeAsking, setKnowledgeAsking] = useState(false);
  const [intelligence, setIntelligence] = useState<{
    learning?: { updates?: number; dataset_size?: number; q_table_size?: number };
    knowledge?: { total_documents?: number; types?: Record<string, number> };
    events?: Array<{ type?: string; message?: string; tick?: number }>;
    decisions?: Array<{ citizen?: string; action?: string; result?: string }>;
    automation?: Record<string, unknown>;
    strategy?: unknown;
  }>({});

  const [selectedCitizenId, setSelectedCitizenId] =
    useState("");

  const [situation, setSituation] =
    useState(
      "I am hungry and need to find food.",
    );

  const [cognition, setCognition] =
    useState<CognitionResult | null>(
      null,
    );

  const [tickLoading, setTickLoading] =
    useState(false);

  const [cognitionLoading, setCognitionLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [selectedLocation, setSelectedLocation] =
    useState<Location | null>(null);
  const [locationDetails, setLocationDetails] =
    useState<Citizen[]>([]);

  const [citizenDetails, setCitizenDetails] =
    useState<Citizen | null>(null);
  const [citizenMemories, setCitizenMemories] =
    useState<Memory[]>([]);
  const [citizenDetailLoading, setCitizenDetailLoading] =
    useState(false);

  const [cognitiveStream, setCognitiveStream] =
    useState<CognitiveStream | null>(null);
  const [expandedCognitiveStage, setExpandedCognitiveStage] = useState<string | null>("REASONING");

  const [trial, setTrial] =
    useState<{ scenario?: string; timeline?: Array<Record<string, number>>; start?: Record<string, number>; end?: Record<string, number>; learning?: { updates?: number; persistent_experiences?: number; q_states?: number } } | null>(null);
  const [trialLoading, setTrialLoading] = useState(false);
  const [worldEvent, setWorldEvent] = useState<{
    label?: string;
    description?: string;
    citizen_effect?: string;
    population?: { before?: number; after?: number; delta?: number };
    resource_changes?: Record<string, { before?: number; after?: number; delta?: number }>;
  } | null>(null);
  const [worldEventLoading, setWorldEventLoading] = useState(false);

  const selectedCitizen = useMemo(
    () =>
      citizens.find(
        (citizen) =>
          citizen.id ===
          selectedCitizenId,
      ) ??
      citizens[0] ??
      null,
    [citizens, selectedCitizenId],
  );

  const loadWorld = useCallback(
    async () => {
      try {
        const response = await fetch(
          `${API}/world`,
        );

        if (!response.ok) {
          throw new Error(
            "World request failed",
          );
        }

        const data = await response.json();

        setWorld(data);
        setError("");
      } catch {
        setError(
          "Backend connection unavailable.",
        );
      }
    },
    [],
  );

  const loadStatus = useCallback(
    async () => {
      try {
        const response = await fetch(
          `${API}/simulation/status`,
        );

        if (!response.ok) return;

        const data =
          await response.json();

        setStatus(data);
      } catch {
        // Keep existing state.
      }
    },
    [],
  );

  const loadCitizens = useCallback(
    async () => {
      try {
        const response = await fetch(
          `${API}/citizens`,
        );

        if (!response.ok) return;

        const data =
          await response.json();

        const list = Array.isArray(data)
          ? data
          : data.citizens ??
            data.items ??
            [];

        setCitizens(list);

        if (
          !selectedCitizenId &&
          list.length
        ) {
          setSelectedCitizenId(
            list[0].id,
          );
        }
      } catch {
        // Keep existing state.
      }
    },
    [selectedCitizenId],
  );

  const loadKnowledge = useCallback(
    async () => {
      try {
        const response = await fetch(API + "/knowledge");
        if (!response.ok) return;
        const data = await response.json();
        setKnowledge(Array.isArray(data.documents) ? data.documents : []);
      } catch {
        // Keep existing state.
      }
    },
    [],
  );

  const askKnowledge = async () => {
    const question = knowledgeQuestion.trim();
    if (!question) return;

    setKnowledgeAsking(true);
    setKnowledgeAnswer("");
    try {
      const response = await fetch(
        `${API}/knowledge/ask?query=${encodeURIComponent(question)}`,
        { method: "POST" },
      );
      if (!response.ok) throw new Error("Knowledge query failed");
      const data = await response.json();
      setKnowledgeAnswer(data.answer ?? "AETHER could not produce an answer.");
      setKnowledgeSources(
        Array.isArray(data.sources)
          ? data.sources.map((source: { title?: string }) => source.title ?? "AETHER knowledge")
          : [],
      );
      await loadKnowledge();
    } catch {
      setKnowledgeAnswer("AETHER Knowledge Intelligence is unavailable.");
    } finally {
      setKnowledgeAsking(false);
    }
  };

  const loadIntelligence = useCallback(
    async () => {
      try {
        const response = await fetch(API + "/civilization/intelligence");
        if (!response.ok) return;
        const data = await response.json();
        setIntelligence(data);
      } catch {
        // Keep existing state.
      }
    },
    [],
  );

  const loadEvents = useCallback(
    async () => {
      try {
        const response = await fetch(
          `${API}/simulation/events`,
        );

        if (!response.ok) return;

        const data =
          await response.json();

        const list = Array.isArray(data)
          ? data
          : data.events ?? [];

        setEvents(
          list.map((event: unknown) =>
            typeof event === "string"
              ? event
              : JSON.stringify(event),
          ),
        );
      } catch {
        // Keep existing state.
      }
    },
    [],
  );

  const loadCognitiveStream = useCallback(async () => {
    try {
      const response = await fetch(API + "/civilization/cognitive-stream");
      if (!response.ok) return;
      setCognitiveStream(await response.json());
    } catch {
      // Keep the last stream snapshot.
    }
  }, []);

  const triggerWorldEvent = async (eventKey: string) => {
    if (worldEventLoading) return;
    setWorldEventLoading(true);
    try {
      const response = await fetch(
        API + "/simulation/event?event_key=" + encodeURIComponent(eventKey),
        { method: "POST" },
      );
      if (!response.ok) throw new Error("Event failed");
      setWorldEvent(await response.json());
      await refreshAll();
      await loadCognitiveStream();
    } catch {
      setError("Civilization event could not be applied.");
    } finally {
      setWorldEventLoading(false);
    }
  };

  const runThirtyDayTrial = async () => {
    if (trialLoading) return;
    setTrialLoading(true);
    setTrial(null);
    try {
      const response = await fetch(API + "/civilization/run-trial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days: 30 }),
      });
      if (!response.ok) throw new Error("Trial failed");
      const data = await response.json();
      setTrial(data);
      await refreshAll();
      await loadCognitiveStream();
    } catch {
      setError("Civilization trial could not be completed.");
    } finally {
      setTrialLoading(false);
    }
  };

  const refreshAll = useCallback(
    async () => {
      await Promise.all([
        loadWorld(),
        loadStatus(),
        loadCitizens(),
        loadEvents(),
        loadKnowledge(),
        loadIntelligence(),
        loadCognitiveStream(),
      ]);
    },
    [
      loadWorld,
      loadStatus,
      loadCitizens,
      loadEvents,
      loadKnowledge,
      loadIntelligence,
      loadCognitiveStream,
    ],
  );

  useEffect(() => {
    refreshAll();

    const interval = window.setInterval(
      refreshAll,
      5000,
    );

    return () =>
      window.clearInterval(interval);
  }, [refreshAll]);

  const openCitizenDetails = async (citizen: Citizen) => {
    setCitizenDetails(citizen);
    setCitizenMemories([]);
    setCitizenDetailLoading(true);
    try {
      const [detailResponse, memoriesResponse] = await Promise.all([
        fetch(`${API}/citizens/${citizen.id}`),
        fetch(`${API}/citizens/${citizen.id}/memories`),
      ]);
      if (detailResponse.ok) {
        const detail = await detailResponse.json();
        setCitizenDetails(detail.citizen ?? detail);
      }
      if (memoriesResponse.ok) {
        const data = await memoriesResponse.json();
        setCitizenMemories(Array.isArray(data.memories) ? data.memories : []);
      }
    } catch {
      // Keep the live citizen snapshot already shown.
    } finally {
      setCitizenDetailLoading(false);
    }
  };

  const openLocationDetails = (location: Location) => {
    setSelectedLocation(location);
    setLocationDetails(
      citizens.filter((citizen) => citizen.location_id === location.id && citizen.alive),
    );
  };

  const advanceTick = async () => {
    setTickLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API}/simulation/tick`,
        {
          method: "POST",
        },
      );

      if (!response.ok) {
        throw new Error(
          "Simulation tick failed",
        );
      }

      await refreshAll();
    } catch {
      setError(
        "Unable to advance the civilization.",
      );
    } finally {
      setTickLoading(false);
    }
  };

  const runCognition = async () => {
    if (!selectedCitizen || cognitionLoading) return;

    setCognitionLoading(true);
    setError("");
    setCognition(null);

    const citizenId = selectedCitizen.id;

    try {
      const response = await fetch(
        `${API}/citizens/${citizenId}/think?situation=${encodeURIComponent(
          situation,
        )}`,
        {
          method: "POST",
        },
      );

      if (!response.ok) {
        throw new Error(
          "Cognition request failed",
        );
      }

      const data =
        await response.json();

      // The API wraps the cognitive result inside \"reasoning\".
      // Normalize it here so the UI can render the real perception,
      // memory, RAG, reasoning, planning, decision and learning data.
      setCognition(data.reasoning ?? data);
    } catch {
      setError(
        "Cognition engine failed to respond.",
      );
    } finally {
      setCognitionLoading(false);
    }
  };

  const metrics = {
    population:
      status?.population ??
      world.population ??
      citizens.length,
    decisions:
      status?.total_decisions ?? 0,
    memories:
      status?.total_memories ?? 0,
    automation:
      status?.event_count ?? 0,
  };

  const resources = resourceEntries(
    world.resources,
  );

  return (
    <div className="aether-app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            A
          </div>

          <div>
            <div className="brand-name">
              AETHER
            </div>

            <div className="brand-subtitle">
              ARTIFICIAL CIVILIZATION
            </div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <span>
            SYSTEM ONLINE
          </span>

          <span className="status-divider" />

          <span>
            {world.day_of_week ||
              "AETHER"}{" "}
            ·{" "}
            {world.time ||
              "--:--:--"}
          </span>
        </div>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <div className="sidebar-section">
            <span className="sidebar-label">
              CIVILIZATION
            </span>

            <button
              className={`nav-item ${
                page === "command"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setPage("command")
              }
            >
              <span>◈</span>
              COMMAND
            </button>

            <button
              className={`nav-item ${
                page === "citizens"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setPage("citizens")
              }
            >
              <span>◎</span>
              CITIZENS
            </button>

            <button
              className={`nav-item ${
                page === "cognition"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setPage("cognition")
              }
            >
              <span>✦</span>
              COGNITION
            </button>

            <button
              className={`nav-item ${
                page === "events"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setPage("events")
              }
            >
              <span>◇</span>
              EVENTS
            </button>

            <button
              className={`nav-item ${
                page === "knowledge"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setPage("knowledge")
              }
            >
              <span>✦</span>
              KNOWLEDGE
            </button>

            <button
              className={`nav-item ${
                page === "intelligence"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setPage("intelligence")
              }
            >
              <span>⌁</span>
              INTELLIGENCE
            </button>
          </div>

          <div className="sidebar-bottom">
            <div className="sidebar-world">
              <span className="mini-pulse" />
              AETHER WORLD
            </div>

            <div className="sidebar-time">
              {world.date ||
                "---- -- --"}
              <br />
              {world.time ||
                "--:--:--"}
            </div>

            <div className="sidebar-version">
              DIGITAL MIND // 01
            </div>
          </div>
        </aside>

        <main className="main-content">
          {error && (
            <div className="error-banner">
              <span>⚠</span>
              {error}
            </div>
          )}

          {page === "command" && (
            <>
              <section className="hero">
                <div className="hero-kicker">
                  <span />
                  THE DIGITAL MIND
                </div>

                <h1>
                  THE MIND
                  <br />
                  <em>IS ALIVE.</em>
                </h1>

                <p>
                  A civilization of autonomous
                  artificial minds — perceiving,
                  reasoning, planning, acting,
                  learning and remembering.
                </p>

                <button
                  className="primary-button"
                  onClick={advanceTick}
                  disabled={tickLoading}
                >
                  <span>
                    {tickLoading
                      ? "PROCESSING..."
                      : "ADVANCE COGNITIVE TICK"}
                  </span>

                  <strong>→</strong>
                </button>
              </section>

              <section className="metrics-grid">
                <MetricCard
                  label="POPULATION"
                  value={metrics.population}
                  detail="AUTONOMOUS CITIZENS"
                  icon="◎"
                />

                <MetricCard
                  label="DECISIONS"
                  value={metrics.decisions}
                  detail="COGNITIVE ACTIONS"
                  icon="✦"
                />

                <MetricCard
                  label="MEMORIES"
                  value={metrics.memories}
                  detail="LONG-TERM EXPERIENCES"
                  icon="◈"
                />

                <MetricCard
                  label="AUTOMATION"
                  value={metrics.automation}
                  detail="SYSTEM EVENTS"
                  icon="ϟ"
                />
              </section>

              <section className="civilization-launch">
                <div>
                  <span className="eyebrow">EXPERIMENTAL CIVILIZATION</span>
                  <h2>RUN THE WORLD FOR 30 DAYS.</h2>
                  <p>Stress the civilization, let autonomous minds respond, watch resources evolve, and turn every outcome into learning.</p>
                </div>
                <button className="trial-button" onClick={runThirtyDayTrial} disabled={trialLoading}>
                  <span className="trial-orbit">◌</span>
                  {trialLoading ? "SIMULATING 30 DAYS..." : "START 30-DAY TRIAL"}
                  <strong>↗</strong>
                </button>
              </section>

              <section
                style={{
                  marginTop: "24px",
                  padding: "22px",
                  border: "1px solid rgba(255,255,255,0.09)",
                  background: "linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015))",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: "18px", alignItems: "flex-end", marginBottom: "16px" }}>
                  <div>
                    <span className="eyebrow">CIVILIZATION SHOCK CONTROLS</span>
                    <h2 style={{ margin: "6px 0 4px" }}>TRIGGER A WORLD EVENT.</h2>
                    <p style={{ margin: 0, opacity: 0.65 }}>
                      These are live simulation interventions. AETHER must adapt and the outputs appear below.
                    </p>
                  </div>
                  {worldEventLoading && <span className="live-tag">APPLYING...</span>}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "10px" }}>
                  {[
                    ["food_crisis", "⚠️ FOOD CRISIS", "(-40%)"],
                    ["energy_grid_failure", "⚡ ENERGY GRID FAILURE", ""],
                    ["economic_market_crash", "📉 ECONOMIC MARKET CRASH", ""],
                    ["population_boom", "👨‍👩‍👧‍👦 POPULATION BOOM", "(+50%)"],
                  ].map(([key, label, impact]) => (
                    <button
                      key={key}
                      onClick={() => void triggerWorldEvent(key)}
                      disabled={worldEventLoading}
                      style={{
                        minHeight: "72px",
                        padding: "14px",
                        textAlign: "left",
                        border: "1px solid rgba(255,255,255,0.12)",
                        background: "rgba(10,14,14,0.72)",
                        color: "inherit",
                        cursor: worldEventLoading ? "wait" : "pointer",
                        fontWeight: 800,
                        letterSpacing: "0.02em",
                      }}
                    >
                      <span style={{ display: "block", fontSize: "13px" }}>{label}</span>
                      {impact && <small style={{ display: "block", marginTop: "7px", opacity: 0.65 }}>{impact}</small>}
                    </button>
                  ))}
                </div>

                {worldEvent && (
                  <div style={{ marginTop: "16px", padding: "18px", border: "1px solid rgba(255,255,255,0.12)", background: "rgba(0,0,0,0.22)" }}>
                    <span className="eyebrow">LIVE CONSEQUENCE REPORT</span>
                    <h3 style={{ margin: "7px 0" }}>{worldEvent.label}</h3>
                    <p style={{ margin: "0 0 12px", opacity: 0.78 }}>{worldEvent.description}</p>
                    <p style={{ margin: "0 0 14px" }}><strong>CITIZEN IMPACT:</strong> {worldEvent.citizen_effect}</p>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "10px" }}>
                      {worldEvent.population && (
                        <div className="detail-panel">
                          <span className="eyebrow">POPULATION</span>
                          <strong>{worldEvent.population.before} → {worldEvent.population.after}</strong>
                          <small>{worldEvent.population.delta! >= 0 ? "+" : ""}{worldEvent.population.delta} citizens</small>
                        </div>
                      )}
                      {Object.entries(worldEvent.resource_changes ?? {}).map(([name, change]) => (
                        <div className="detail-panel" key={name}>
                          <span className="eyebrow">{name.toUpperCase()}</span>
                          <strong>{Math.round(Number(change.before ?? 0))} → {Math.round(Number(change.after ?? 0))}</strong>
                          <small>{Number(change.delta ?? 0) >= 0 ? "+" : ""}{Math.round(Number(change.delta ?? 0))} change</small>
                        </div>
                      ))}
                    </div>

                    <p style={{ margin: "14px 0 0", opacity: 0.62, fontSize: "12px" }}>
                      The result is now part of AETHER's live world state, event history and knowledge layer.
                    </p>
                  </div>
                )}
              </section>

              <WorldMap
                locations={world.locations ?? []}
                citizens={citizens}
                events={events}
                onLocationSelect={openLocationDetails}
              />

              <section className="dashboard-grid">
                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <span className="eyebrow">
                        WORLD STATE
                      </span>

                      <h2>
                        CIVILIZATION STATE
                      </h2>
                    </div>

                    <span className="live-tag">
                      LIVE
                    </span>
                  </div>

                  <div className="world-state-grid">
                    <div>
                      <span>DATE</span>
                      <strong>
                        {world.date ||
                          "—"}
                      </strong>
                    </div>

                    <div>
                      <span>TIME</span>
                      <strong>
                        {world.time ||
                          "—"}
                      </strong>
                    </div>

                    <div>
                      <span>DAY</span>
                      <strong>
                        {world.day}
                      </strong>
                    </div>

                    <div>
                      <span>TIMEZONE</span>
                      <strong>
                        {world.timezone}
                      </strong>
                    </div>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <span className="eyebrow">
                        INTELLIGENCE
                      </span>

                      <h2>
                        COGNITIVE ACTIVITY
                      </h2>
                    </div>

                    <span className="intelligence-icon">
                      ✦
                    </span>
                  </div>

                  <div className="activity-list">
                    <div>
                      <span className="activity-number">
                        01
                      </span>

                      <span>
                        Perception →
                        memory retrieval
                      </span>
                    </div>

                    <div>
                      <span className="activity-number">
                        02
                      </span>

                      <span>
                        Symbolic + Bayesian
                        reasoning
                      </span>
                    </div>

                    <div>
                      <span className="activity-number">
                        03
                      </span>

                      <span>
                        Goal planning →
                        autonomous action
                      </span>
                    </div>
                  </div>
                </div>
              </section>

              <section className="cognition-stream-panel">
                <div className="stream-head">
                  <div>
                    <span className="eyebrow">LIVE COGNITIVE STREAM</span>
                    <h2>WATCH A MIND THINK.</h2>
                  </div>
                  <span className="stream-tick">TICK {cognitiveStream?.tick ?? 0} · {cognitiveStream?.citizen ?? "AETHER"}</span>
                </div>
                <div className="stream-subject-note">
                  <strong>WHO IS THINKING?</strong> {cognitiveStream?.citizen ?? "No citizen selected yet"}
                  {cognitiveStream?.citizen_id ? ` · ${cognitiveStream.citizen_id}` : ""}
                  <span> · Every trace below belongs to this one citizen for this tick.</span>
                </div>
                <div className="stream-reading-guide">
                  <strong>HOW TO READ THIS:</strong> citizen need values are <em>pressure scores</em>; global resources show <em>amount / capacity</em>. A high energy-pressure number means the citizen is tired, not that civilization has low energy.
                </div>
                <CognitionPipeline
                  stages={cognitiveStream?.stages}
                  activeStage={expandedCognitiveStage}
                  onStageSelect={(stage) =>
                    setExpandedCognitiveStage(
                      expandedCognitiveStage === stage ? null : stage,
                    )
                  }
                />
                <div className="stream-grid">
                  {(cognitiveStream?.stages ?? [
                    {stage:"PERCEPTION",state:"Awaiting live signal",value:"Advance the civilization"},
                    {stage:"MEMORY",state:"Memory engine ready",value:"Retrieval standing by"},
                    {stage:"RAG",state:"Knowledge layer ready",value:"Grounded context standing by"},
                    {stage:"REASONING",state:"FAI engine ready",value:"Symbolic + Bayesian"},
                    {stage:"PLANNING",state:"Planner ready",value:"Goal-directed planning"},
                    {stage:"DECISION",state:"Decision layer ready",value:"Action selection"},
                    {stage:"ACTION",state:"Automation ready",value:"World-changing actions"},
                    {stage:"LEARNING",state:"Learning loop ready",value:"Q-Learning + memory"},
                  ]).map((stage, index) => (
                    <article
                      className={"stream-stage " + (expandedCognitiveStage === stage.stage ? "is-expanded" : "")}
                      key={stage.stage}
                      onClick={() => setExpandedCognitiveStage(expandedCognitiveStage === stage.stage ? null : stage.stage)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          setExpandedCognitiveStage(
                            expandedCognitiveStage === stage.stage ? null : stage.stage,
                          );
                        }
                      }}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="stream-stage-top">
                        <span>{String(index + 1).padStart(2,"0")}</span>
                        <b>{stage.stage}</b>
                        <em>{expandedCognitiveStage === stage.stage ? "COLLAPSE" : "INSPECT"}</em>
                      </div>
                      <small>{stage.state}</small>
                      <p className="stage-explanation">{explainCognitiveStage(stage.stage)}</p>
                      <pre>{formatCognitiveValue(stage.value)}</pre>
                      {expandedCognitiveStage === stage.stage ? (
                        <div className="stage-inspector">
                          <div className="inspector-label">COGNITIVE EVIDENCE · STAGE {String(index + 1).padStart(2,"0")}</div>
                          <div className="inspector-flow">
                            <span>INPUT</span><i>→</i><strong>{stage.stage}</strong><i>→</i><span>OUTPUT</span>
                          </div>
                          <pre className="evidence-json">{formatCognitiveValue(stage.value)}</pre>
                        </div>
                      ) : null}
                    </article>
                  ))}
                </div>
              </section>

              {trial?.timeline?.length ? (
                <section className="trial-results">
                  <div className="section-heading">
                    <div>
                      <span className="eyebrow">CIVILIZATION EXPERIMENT</span>
                      <h2>30 DAYS OF AETHER.</h2>
                    </div>
                    <span className="live-tag">TRIAL COMPLETE</span>
                  </div>
                  <div className="trial-summary">
                    <div><span>START FOOD</span><strong>{Math.round(Number(trial.start?.food ?? 0))}</strong></div>
                    <div><span>END FOOD</span><strong>{Math.round(Number(trial.end?.food ?? 0))}</strong></div>
                    <div><span>LEARNING</span><strong>{trial.learning?.persistent_experiences ?? 0}</strong></div>
                    <div><span>Q STATES</span><strong>{trial.learning?.q_states ?? 0}</strong></div>
                  </div>
                  <div className="trial-timeline">
                    {trial.timeline.map((day) => (
                      <div className="trial-day" key={day.day}>
                        <span>DAY {day.day}</span>
                        <div className="trial-bars">
                          <i style={{height: `${Math.min(100, Number(day.food ?? 0) / 20)}%`}} title="Food" />
                          <i style={{height: `${Math.min(100, Number(day.water ?? 0) / 40)}%`}} title="Water" />
                          <i style={{height: `${Math.min(100, Number(day.energy ?? 0) / 30)}%`}} title="Energy" />
                        </div>
                        <small>{Math.round(Number(day.food ?? 0))} food</small>
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}

              <section className="section-block">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">
                      RESOURCE SYSTEM
                    </span>

                    <h2>
                      GLOBAL RESERVES
                    </h2>
                  </div>
                </div>

                <div className="stream-reading-guide resource-guide">
                  <strong>RESOURCE SYSTEM:</strong> These are civilization-wide reserves. The first number is what AETHER currently has; the second is the maximum capacity. The percentage is current stock ÷ capacity. These are different from a citizen's personal need-pressure values.
                </div>
                <div className="resources-grid">
                  {resources.map(
                    (resource) => (
                      <ResourceCard
                        key={
                          resource.name
                        }
                        resource={
                          resource
                        }
                      />
                    ),
                  )}
                </div>
              </section>

              <section className="section-block">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">
                      WORLD SIGNALS
                    </span>

                    <h2>
                      RECENT ACTIVITY
                    </h2>
                  </div>
                </div>

                <div className="signals-panel">
                  {(
                    events.length
                      ? events
                      : world.events
                  )
                    .slice(-8)
                    .reverse()
                    .map(
                      (
                        event,
                        index,
                      ) => (
                        <div
                          className="signal-row"
                          key={`${event}-${index}`}
                        >
                          <span className="signal-time">
                            {String(
                              index + 1,
                            ).padStart(
                              2,
                              "0",
                            )}
                          </span>

                          <span className="signal-dot" />

                          <span>
                            {event}
                          </span>
                        </div>
                      ),
                    )}

                  {!events.length &&
                    !world.events.length && (
                      <div className="empty-state">
                        No civilization
                        signals yet.
                      </div>
                    )}
                </div>
              </section>
            </>
          )}

          {page === "citizens" && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <span className="eyebrow">
                    AETHER POPULATION
                  </span>

                  <h1>
                    AUTONOMOUS MINDS
                  </h1>

                  <p>
                    Every citizen perceives
                    the world, maintains
                    memories, evaluates needs,
                    creates plans and chooses
                    actions.
                  </p>
                </div>

                <div className="large-counter">
                  <strong>
                    {citizens.length ||
                      metrics.population}
                  </strong>

                  <span>
                    ACTIVE MINDS
                  </span>
                </div>
              </div>

              <div className="citizens-layout">
                <div className="citizen-grid">
                  {citizens.map(
                    (citizen) => (
                      <CitizenCard
                        key={citizen.id}
                        citizen={citizen}
                        selected={
                          selectedCitizenId ===
                          citizen.id
                        }
                        onSelect={() => {
                          setSelectedCitizenId(citizen.id);
                          openCitizenDetails(citizen);
                        }}
                      />
                    ),
                  )}

                  {!citizens.length && (
                    <div className="empty-state large">
                      Waiting for citizen
                      data from AETHER...
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}

          {page === "cognition" && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <span className="eyebrow">
                    ARTIFICIAL INTELLIGENCE
                  </span>

                  <h1>
                    COGNITIVE ENGINE
                  </h1>

                  <p>
                    Observe one artificial
                    mind move through the
                    complete perception →
                    memory → reasoning →
                    planning → action cycle.
                  </p>
                </div>
              </div>

              <div className="cognition-layout">
                <div className="panel citizen-selector-panel">
                  <div className="panel-header">
                    <div>
                      <span className="eyebrow">
                        SUBJECT
                      </span>

                      <h2>
                        SELECT CITIZEN
                      </h2>
                    </div>
                  </div>

                  <div className="selector-list">
                    {citizens
                      .map(
                        (citizen) => (
                          <button
                            key={
                              citizen.id
                            }
                            className={
                              selectedCitizen?.id ===
                              citizen.id
                                ? "selector-item active"
                                : "selector-item"
                            }
                            onClick={() => {
                              setSelectedCitizenId(citizen.id);
                              setCognition(null);
                              setError("");
                            }}
                          >
                            <span>
                              {citizen.name}
                            </span>

                            <small>
                              {
                                citizen.occupation
                              }
                            </small>
                          </button>
                        ),
                      )}
                  </div>
                </div>

                <div className="cognition-main">
                  <div className="panel situation-panel">
                    <div className="panel-header">
                      <div>
                        <span className="eyebrow">
                          INPUT
                        </span>

                        <h2>
                          SITUATION
                        </h2>
                      </div>
                    </div>

                    <textarea
                      value={situation}
                      onChange={(event) =>
                        setSituation(
                          event.target
                            .value,
                        )
                      }
                    />

                    <button
                      className="primary-button"
                      onClick={
                        runCognition
                      }
                      disabled={
                        cognitionLoading ||
                        !selectedCitizen
                      }
                    >
                      <span>
                        {cognitionLoading
                          ? "THINKING..."
                          : "RUN COGNITION"}
                      </span>

                      <strong>
                        ✦
                      </strong>
                    </button>
                  </div>

                  <CognitionPipeline />

                  {cognition && (
                    <div className="cognition-result">
                      <div className="panel result-panel">
                        <div className="panel-header">
                          <div>
                            <span className="eyebrow">
                              AI OUTPUT
                            </span>

                            <h2>
                              DECISION
                              ANALYSIS
                            </h2>
                          </div>
                        </div>

                        <div className="decision-highlight">
                          <span>
                            SELECTED ACTION
                          </span>

                          <strong>
                            {(cognition.decision?.action ??
                              cognition.symbolic_reasoning?.plan?.steps?.[0]?.action ??
                              "continue_daily_activity").replaceAll("_", " ")}
                          </strong>
                        </div>

                        <div className="reasoning-grid">
                          <div>
                            <span>WHAT AETHER SEES</span>
                            <p>
                              {(() => {
                                const needs = cognition.perception?.needs;
                                const world = cognition.perception?.world;
                                const citizen = cognition.perception?.citizen;
                                const urgent: string[] = [];
                                if (Number(needs?.hunger ?? 0) >= 70) urgent.push(`hunger is high (${Number(needs?.hunger).toFixed(0)}/100)`);
                                if (Number(needs?.energy ?? 100) <= 30) urgent.push(`energy is low (${Number(needs?.energy).toFixed(0)}/100)`);
                                if (Number(needs?.social ?? 100) <= 30) urgent.push(`social need is low (${Number(needs?.social).toFixed(0)}/100)`);
                                if (Number(needs?.safety ?? 100) <= 30) urgent.push(`safety is low (${Number(needs?.safety).toFixed(0)}/100)`);
                                const state = urgent.length
                                  ? `The main concern is that ${urgent.join(", and ")}.`
                                  : "No critical personal need is detected right now, so the citizen can continue normal activity.";
                                return `${citizen?.name ?? "The citizen"} is a ${citizen?.occupation ?? "citizen"} in ${citizen?.location ?? "an unknown location"}. ${state} AETHER is currently on day ${world?.day ?? "?"} at ${world?.time ?? "--:--"}.`;
                              })()}
                            </p>
                          </div>

                          <div>
                            <span>NEEDS AT A GLANCE</span>
                            <p>
                              {(() => {
                                const n = cognition.perception?.needs;
                                return `Hunger ${Number(n?.hunger ?? 0).toFixed(0)}/100 · Energy ${Number(n?.energy ?? 0).toFixed(0)}/100 · Social ${Number(n?.social ?? 0).toFixed(0)}/100 · Safety ${Number(n?.safety ?? 0).toFixed(0)}/100 · Money ${Number(n?.money ?? 0).toFixed(0)}`;
                              })()}
                            </p>
                            <small>These are personal state values. Lower energy/social/safety means more pressure; lower money also means more financial pressure.</small>
                          </div>

                          <div>
                            <span>WHAT AETHER FOUND</span>
                            <p>
                              {(() => {
                                const symbolic = cognition.symbolic_reasoning;
                                const facts = (symbolic?.facts ?? []).filter((fact) => fact !== "population_active");
                                const inferred = (symbolic?.new_facts ?? []).map((item) => item.conclusion).filter(Boolean) as string[];
                                const goal = symbolic?.goal ? humanizeCognitionTerm(symbolic.goal) : "Maintain wellbeing";
                                if (!facts.length && !inferred.length) return `No urgent rule was triggered. The immediate goal is to ${goal.toLowerCase()}.`;
                                const found = facts.slice(0, 3).map(humanizeCognitionTerm);
                                const conclusions = inferred.slice(0, 3).map(humanizeCognitionTerm);
                                return `AETHER detected ${found.length ? found.join(", ") : "no critical warning"}. Its rules then concluded ${conclusions.length ? conclusions.join(", ") : "that wellbeing should be maintained"}.`;
                              })()}
                            </p>
                          </div>

                          <div>
                            <span>PROBABILITY CHECK</span>
                            <p>
                              {(() => {
                                const bayesian = cognition.symbolic_reasoning?.bayesian;
                                const posterior = bayesian?.posterior ?? {};
                                const key = bayesian?.most_likely;
                                const probability = key ? Number(posterior[key]) : NaN;
                                return key
                                  ? `AETHER estimates “${humanizeCognitionTerm(key)}” as the most likely condition, with about ${Number.isFinite(probability) ? (probability * 100).toFixed(0) : "?"}% probability.`
                                  : "No strong resource shortage was detected, so AETHER is treating the situation as normal.";
                              })()}
                            </p>
                          </div>

                          <div>
                            <span>WHY THIS ACTION?</span>
                            <p>
                              {cognition.decision?.reason ??
                                `AETHER chose ${humanizeCognitionTerm(cognition.decision?.action ?? "an appropriate action")} because it best matches the citizen's current needs.`}
                            </p>
                            {cognition.decision?.scores && (
                              <small>
                                Decision scores: {Object.entries(cognition.decision.scores).sort(([, a], [, b]) => Number(b) - Number(a)).map(([action, score]) => `${humanizeCognitionTerm(action)} ${Number(score).toFixed(1)}`).join(" · ")}
                              </small>
                            )}
                          </div>

                          <div>
                            <span>MEMORY + KNOWLEDGE</span>
                            <p>
                              {(() => {
                                const memoryCount = cognition.memory?.retrieved?.length ?? cognition.memories?.length ?? 0;
                                const ragCount = cognition.rag?.results?.length ?? 0;
                                return `AETHER checked ${memoryCount} relevant past experience${memoryCount === 1 ? "" : "s"} and ${ragCount} knowledge result${ragCount === 1 ? "" : "s"} before deciding.`;
                              })()}
                            </p>
                          </div>
                        </div>

                        <div className="plan-block">
                          <span>STEP-BY-STEP PLAN</span>
                          <ol>
                            {(() => {
                              const steps = cognition.symbolic_reasoning?.plan?.steps ?? [];
                              const decisionPlan = cognition.decision?.plan?.trim();
                              if (steps.length) {
                                return steps.map((step, index) => (
                                  <li key={index}>
                                    <strong>{humanizeCognitionTerm(step.action ?? "action")}</strong>
                                    {step.description ? ` — ${step.description}` : ""}
                                  </li>
                                ));
                              }
                              if (decisionPlan) {
                                return decisionPlan
                                  .split(/\s*→\s*|\s*;\s*/)
                                  .filter(Boolean)
                                  .map((step, index) => (
                                    <li key={index}>{humanizeCognitionTerm(step)}</li>
                                  ));
                              }
                              return <li>Monitor the citizen's needs and choose the next useful action.</li>;
                            })()}
                          </ol>
                        </div>

                        <div className="plan-block">
                          <span>WHAT THE CITIZEN LEARNS</span>
                          <p>
                            {cognition.decision?.learning ??
                              "The experience is stored as a memory so future decisions can use what happened here."}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}

          {page === "intelligence" && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <span className="eyebrow">CIVILIZATION INTELLIGENCE</span>
                  <h1>THE LEARNING LOOP</h1>
                  <p>One live surface connecting events, knowledge, decisions, learning and strategic reasoning.</p>
                </div>
                <div className="large-counter">
                  <strong>{intelligence.learning?.updates ?? 0}</strong>
                  <span>LEARNING UPDATES</span>
                </div>
              </div>

              <div className="metrics-grid">
                <MetricCard label="KNOWLEDGE" value={intelligence.knowledge?.total_documents ?? knowledge.length} detail="LIVE DOCUMENTS" icon="✦" />
                <MetricCard label="Q TABLE" value={intelligence.learning?.q_table_size ?? 0} detail="LEARNED STATE-ACTIONS" icon="◎" />
                <MetricCard label="DATASET" value={intelligence.learning?.dataset_size ?? 0} detail="EXPERIENCE SAMPLES" icon="◈" />
                <MetricCard label="EVENTS" value={intelligence.events?.length ?? events.length} detail="LIVE SIGNALS" icon="◇" />
              </div>

              <div className="dashboard-grid">
                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <span className="eyebrow">CIVILIZATION LOOP</span>
                      <h2>WORLD → AI → WORLD</h2>
                    </div>
                  </div>
                  <div className="activity-list">
                    <div><span className="activity-number">01</span><span>World state produces events and observations.</span></div>
                    <div><span className="activity-number">02</span><span>Knowledge + memory ground the next decision.</span></div>
                    <div><span className="activity-number">03</span><span>Search, logic, Bayesian inference and planning evaluate options.</span></div>
                    <div><span className="activity-number">04</span><span>Action changes the world; Q-Learning records the outcome.</span></div>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <span className="eyebrow">RECENT LEARNING</span>
                      <h2>EXPERIENCE → POLICY</h2>
                    </div>
                  </div>
                  <div className="signals-panel">
                    {(intelligence.decisions ?? []).slice().reverse().slice(0, 8).map((d, i) => (
                      <div className="signal-row" key={i}>
                        <span className="signal-time">{String(i + 1).padStart(2, "0")}</span>
                        <span className="signal-dot" />
                        <span>{d.citizen ?? "Citizen"} chose <strong>{d.action ?? "unknown"}</strong> — {d.result ?? "observed"}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="panel" style={{ marginTop: "24px" }}>
                <div className="panel-header">
                  <div>
                    <span className="eyebrow">LIVE KNOWLEDGE</span>
                    <h2>WHAT AETHER CURRENTLY KNOWS</h2>
                  </div>
                </div>
                <div className="signals-panel">
                  {(intelligence.events ?? []).slice().reverse().map((event, i) => (
                    <div className="signal-row" key={i}>
                      <span className="signal-time">{String(i + 1).padStart(2, "0")}</span>
                      <span className="signal-dot" />
                      <span>{event.type?.toUpperCase() ?? "EVENT"} — {event.message ?? "No message"}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          {page === "knowledge" && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <span className="eyebrow">AETHER MEMORY SYSTEM</span>
                  <h1>KNOWLEDGE BASE</h1>
                  <p>Retrieved knowledge that gives AETHER citizens context beyond their immediate world state.</p>
                </div>
                <div className="large-counter">
                  <strong>{knowledge.length}</strong>
                  <span>KNOWLEDGE DOCUMENTS</span>
                </div>
              </div>

              <div className="panel knowledge-query-panel" style={{ marginBottom: "24px" }}>
                <div className="panel-header">
                  <div>
                    <span className="eyebrow">AETHER KNOWLEDGE INTELLIGENCE</span>
                    <h2>ASK THE CIVILIZATION</h2>
                  </div>
                  <span className="panel-status">LIVE RAG + LOCAL AI</span>
                </div>
                <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                  <input
                    value={knowledgeQuestion}
                    onChange={(event) => setKnowledgeQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") void askKnowledge();
                    }}
                    placeholder="Ask about the world, citizens, resources, events or decisions..."
                    style={{ flex: 1, minWidth: "260px", padding: "14px 16px", background: "#0b0f0f", border: "1px solid #27302f", color: "inherit" }}
                  />
                  <button className="primary-button" onClick={() => void askKnowledge()} disabled={knowledgeAsking}>
                    {knowledgeAsking ? "THINKING..." : "ASK AETHER"}
                  </button>
                </div>
                {knowledgeAnswer && (
                  <div className="event-card" style={{ marginTop: "18px" }}>
                    <div className="event-index">AI</div>
                    <div className="event-line" />
                    <div className="event-content">
                      <span>GROUNDED IN LIVE CIVILIZATION DATA</span>
                      <h2>ANSWER</h2>
                      <p>{knowledgeAnswer}</p>
                      {knowledgeSources.length > 0 && (
                        <small>SOURCES: {knowledgeSources.join(" · ")}</small>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="events-stream">
                {knowledge.map((document) => (
                  <article className="event-card" key={document.id}>
                    <div className="event-index">KB</div>
                    <div className="event-line" />
                    <div className="event-content">
                      <span>{document.knowledge_type.toUpperCase()} · {document.source.toUpperCase()}</span>
                      <h2>{document.title}</h2>
                      <p>{document.content}</p>
                    </div>
                  </article>
                ))}
                {!knowledge.length && (
                  <div className="empty-state large">No knowledge documents loaded.</div>
                )}
              </div>
            </section>
          )}

          {page === "events" && (
            <section className="page-section">
              <div className="page-heading">
                <div>
                  <span className="eyebrow">
                    CIVILIZATION LOG
                  </span>

                  <h1>
                    EVENT STREAM
                  </h1>

                  <p>
                    Decisions, discoveries,
                    relationships, crises and
                    automation events emerging
                    from AETHER.
                  </p>
                </div>

                <div className="large-counter">
                  <strong>
                    {events.length}
                  </strong>

                  <span>
                    RECORDED EVENTS
                  </span>
                </div>
              </div>

              <div className="events-stream">
                {events
                  .slice()
                  .reverse()
                  .map(
                    (
                      event,
                      index,
                    ) => (
                      <div
                        className="event-card"
                        key={`${event}-${index}`}
                      >
                        <div className="event-index">
                          {String(
                            index + 1,
                          ).padStart(
                            3,
                            "0",
                          )}
                        </div>

                        <div className="event-line" />

                        <div className="event-content">
                          <span>
                            AETHER EVENT
                          </span>

                          <p>
                            {humanizeEvent(event)}
                          </p>
                        </div>
                      </div>
                    ),
                  )}

                {!events.length && (
                  <div className="empty-state large">
                    No events recorded yet.
                    Advance a cognitive tick
                    to generate civilization
                    activity.
                  </div>
                )}
              </div>
            </section>
          )}

          {citizenDetails && (
            <div className="aether-modal-backdrop" onClick={() => setCitizenDetails(null)}>
              <section className="aether-modal citizen-detail-modal" onClick={(event) => event.stopPropagation()}>
                <button className="modal-close" onClick={() => setCitizenDetails(null)}>×</button>
                <div className="detail-hero">
                  <div className="detail-avatar">{citizenDetails.name.split(" ").map((part) => part[0]).join("").slice(0, 2)}</div>
                  <div><span className="eyebrow">CITIZEN PROFILE · {citizenDetails.id.toUpperCase()}</span><h1>{citizenDetails.name}</h1><p>{citizenDetails.age} years · {citizenDetails.occupation.replaceAll("_", " ")} · {citizenDetails.location_id}</p></div>
                  <span className={"detail-status " + (citizenDetails.alive ? "online" : "offline")}>{citizenDetails.alive ? "ALIVE / ACTIVE" : "INACTIVE"}</span>
                </div>
                <div className="detail-grid">
                  <div className="detail-panel"><span className="eyebrow">CURRENT STATE</span><h3>{citizenDetails.current_action.replaceAll("_", " ")}</h3><div className="detail-bars"><NeedBar label="HUNGER" value={citizenDetails.needs.hunger} inverse /><NeedBar label="ENERGY" value={citizenDetails.needs.energy} /><NeedBar label="SOCIAL" value={citizenDetails.needs.social} /><NeedBar label="SAFETY" value={citizenDetails.needs.safety} /></div><div className="detail-money">¤ {formatNumber(citizenDetails.needs.money)} available</div></div>
                  <div className="detail-panel"><span className="eyebrow">PERSONALITY</span><div className="stat-list">{Object.entries(citizenDetails.personality ?? {}).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><strong>{(Number(value) * 100).toFixed(0)}%</strong></div>)}</div></div>
                  <div className="detail-panel"><span className="eyebrow">GOALS</span><ul className="detail-list">{citizenDetails.goals.map((goal) => <li key={goal}>{goal}</li>)}</ul></div>
                  <div className="detail-panel"><span className="eyebrow">BELIEFS</span><ul className="detail-list">{citizenDetails.beliefs.map((belief) => <li key={belief}>{belief}</li>)}</ul></div>
                  <div className="detail-panel detail-wide"><span className="eyebrow">SKILL MATRIX</span><div className="skill-grid">{Object.entries(citizenDetails.skills).map(([skill, value]) => <div key={skill}><span>{skill.replaceAll("_", " ")}</span><div className="skill-track"><i style={{width: (Number(value) * 100) + "%"}} /></div><strong>{(Number(value) * 100).toFixed(0)}</strong></div>)}</div></div>
                  <div className="detail-panel detail-wide"><span className="eyebrow">RELATIONSHIPS</span>{Object.keys(citizenDetails.relationships).length ? <div className="relationship-grid">{Object.entries(citizenDetails.relationships).map(([id, value]) => <span key={id}>{id} · {(Number(value) * 100).toFixed(0)}%</span>)}</div> : <p className="muted-detail">No recorded relationships yet. Social connections will emerge through civilization activity.</p>}</div>
                  <div className="detail-panel detail-wide"><div className="detail-panel-title"><span className="eyebrow">MEMORY STREAM</span><span>{citizenMemories.length} RETRIEVED</span></div>{citizenDetailLoading ? <p className="muted-detail">Retrieving this citizen's episodic memory...</p> : citizenMemories.length ? <div className="memory-stream">{citizenMemories.slice().reverse().map((memory) => <article key={memory.id}><span>{memory.memory_type.toUpperCase()} · {new Date(memory.timestamp).toLocaleString()}</span><p>{memory.content}</p></article>)}</div> : <p className="muted-detail">No episodic memories retrieved yet.</p>}</div>
                </div>
              </section>
            </div>
          )}

          {selectedLocation && (
            <div className="aether-modal-backdrop" onClick={() => setSelectedLocation(null)}>
              <section className="aether-modal location-detail-modal" onClick={(event) => event.stopPropagation()}>
                <button className="modal-close" onClick={() => setSelectedLocation(null)}>×</button>
                <div className="location-hero"><span className="location-symbol">{selectedLocation.location_type === "city" ? "◆" : "◈"}</span><div><span className="eyebrow">AETHER TERRITORY · LIVE NODE</span><h1>{selectedLocation.name}</h1><p>{selectedLocation.location_type.toUpperCase()} · {selectedLocation.population} RESIDENTS</p></div></div>
                <div className="location-profile-grid">
                  <div className="detail-panel location-special"><span className="eyebrow">WHAT MAKES IT SPECIAL</span><h3>{({capital:"Political and cultural heart of AETHER. The main coordination point for civilization-wide decisions.",north:"Residential district and northern population center.",industrial:"Production and engineering hub for infrastructure and machinery.",farmland:"Food-producing region supporting AETHER's food supply.",harbor:"Trade and logistics gateway for movement of goods and resources.",forest:"Natural frontier and ecological zone for exploration and resources.",mountains:"Highland frontier with difficult terrain and strategic resource value."} as Record<string,string>)[selectedLocation.id] ?? "A significant AETHER location with an active role in the civilization."}</h3></div>
                  <div className="detail-panel"><span className="eyebrow">WHO LIVES HERE</span><div className="location-citizen-list">{locationDetails.slice(0, 12).map((citizen) => <button key={citizen.id} onClick={() => { setSelectedLocation(null); openCitizenDetails(citizen); }}><strong>{citizen.name}</strong><span>{citizen.occupation.replaceAll("_", " ")} · {citizen.current_action.replaceAll("_", " ")}</span></button>)}{locationDetails.length > 12 && <small>+ {locationDetails.length - 12} more residents</small>}{!locationDetails.length && <p className="muted-detail">No active residents are currently assigned to this node.</p>}</div></div>
                  <div className="detail-panel"><span className="eyebrow">WHAT IS HAPPENING</span><div className="location-activity">{locationDetails.slice(0, 6).map((citizen) => <div key={citizen.id}><span className="activity-dot" /><strong>{citizen.name}</strong><span>{citizen.current_action.replaceAll("_", " ")}</span></div>)}{events.slice().reverse().slice(0, 3).map((event, index) => <div key={"event-" + index}><span className="activity-dot event" /><strong>WORLD EVENT</strong><span>{event}</span></div>)}</div></div>
                  <div className="detail-panel"><span className="eyebrow">NODE DATA</span><div className="node-data"><div><span>COORDINATES</span><strong>{selectedLocation.x} / {selectedLocation.y}</strong></div><div><span>TYPE</span><strong>{selectedLocation.location_type}</strong></div><div><span>POPULATION</span><strong>{selectedLocation.population}</strong></div><div><span>ACTIVE MINDS</span><strong>{locationDetails.length}</strong></div></div></div>
                </div>
              </section>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;