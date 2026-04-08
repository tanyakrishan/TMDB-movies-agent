# Movie Analysis Agent

A multi-agent data analyst system that performs **Collect → EDA → Hypothesize** on movie data using the TMDB API and MovieLens dataset.

Built with **Google ADK** (Agent Development Kit) + **Vertex AI** (Gemini 2.0 Flash), deployed on **Google Cloud Run**.

---

## Three-Step Data Analysis Pipeline

### Step 1: Collect

The agent retrieves real-world movie data at runtime from **two distinct sources**:

**TMDB API** (primary — runtime HTTP requests):
- `search_movies()` — keyword search (`/search/movie`)
- `discover_movies()` — filter by genre, year, rating, sort order (`/discover/movie`)
- `get_movie_details()` — budget, revenue, runtime, cast, director (`/movie/{id}?append_to_response=credits`)
- `get_trending_movies()` — current trending films (`/trending/movie/{window}`)

**MovieLens via DuckDB SQL** (secondary — SQL composition over bundled CSVs):
- `query_ratings()` — the agent writes and executes SQL queries against 100K+ user ratings, 9.7K movies, user tags, and a bridge table linking MovieLens IDs to TMDB IDs
- Enables analysis of rating distributions, user behavior patterns, and free-text tags that TMDB does not provide

**Files**: `analysis_agent/tools/tmdb_tools.py`, `analysis_agent/tools/duckdb_tools.py`

### Step 2: Explore and Analyze (EDA)

The agent performs statistical analysis on collected data using two tool calls:

- `compute_stats()` — rating distributions, genre frequency, decade profiles, budget/revenue Pearson correlation, scatter data points, key observations
- `detect_anomalies()` — highest/lowest rated, most profitable (ROI), biggest flops, most polarizing films (high vote count + mediocre rating)

Both tools return **Pydantic-validated structured output** (`StatsResult`, `AnomalyResult`).

EDA is dynamic: different questions trigger different data collection and produce different statistical findings. A question about profitability emphasizes budget-revenue correlation and ROI; a question about genre evolution emphasizes decade profiles and genre frequency shifts.

**File**: `analysis_agent/tools/eda_tools.py`

### Step 3: Hypothesize

The hypothesis agent receives EDA findings and produces:
1. A **narrative hypothesis** (3-5 sentences) grounded in specific data points
2. **Evidence bullets** citing exact numbers from the analysis
3. **Chart.js visualizations** (genre bar chart, decade bar chart, scatter plot)
4. A **markdown report artifact** saved to `outputs/`

**Files**: `analysis_agent/tools/hypothesis_tools.py`, `analysis_agent/tools/artifact_tools.py`

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  root_agent                       │
│              (Orchestrator)                       │
│                                                   │
│  Tools:                                           │
│    search_movies      (TMDB API)                  │
│    discover_movies    (TMDB API)                  │
│    get_movie_details  (TMDB API)                  │
│    get_trending_movies (TMDB API)                 │
│    query_ratings      (DuckDB SQL / MovieLens)    │
│    compute_stats      (EDA)                       │
│    detect_anomalies   (EDA)                       │
│                                                   │
│  Sub-agent (AgentTool):                           │
│  ┌────────────────────────────────────────────┐   │
│  │         hypothesis_agent                    │   │
│  │  Tools: build_chart_data, save_report       │   │
│  └────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
         │                         │
    TMDB REST API           DuckDB (in-process)
    (runtime HTTP)          over MovieLens CSVs
```

**Multi-agent pattern**: Agent-as-tool-call. The `root_agent` orchestrates the full pipeline and delegates to `hypothesis_agent` (wrapped as an `AgentTool`) for Step 3. Data-heavy tools remain on the root agent to avoid serializing large payloads through sub-agent text boundaries.

**File**: `analysis_agent/agent.py`

---

## Requirements Checklist

### Core Requirements

| Requirement | Implementation | Location |
|---|---|---|
| Frontend | Cinema-themed dark UI with Chat.js visualizations | `index.html` |
| Agent framework | Google ADK (Agent, Runner, FunctionTool, AgentTool) | `analysis_agent/agent.py` |
| Tool calling | 9 tools across collection, EDA, and hypothesis | `analysis_agent/tools/*.py` |
| Non-trivial dataset | TMDB API (millions of movies) + MovieLens (100K ratings, 9.7K movies) | `tmdb_tools.py`, `duckdb_tools.py`, `data/` |
| Multi-agent pattern | root_agent + hypothesis_agent (agent-as-tool-call) | `analysis_agent/agent.py` |
| Deployed | Google Cloud Run via Dockerfile | `Dockerfile` |
| README | This file | `README.md` |

### Grab-Bag Electives

| Elective | Implementation | Location |
|---|---|---|
| **Data Visualization** | 3 Chart.js charts: genre horizontal bar, decade grouped bar, rating scatter | `hypothesis_tools.py` → `build_chart_data()`, `index.html` → `renderCharts()` |
| **Second data retrieval method** | TMDB REST API + DuckDB SQL over bundled MovieLens CSVs | `tmdb_tools.py`, `duckdb_tools.py` |
| **Structured output** | Pydantic schemas: `StatsResult`, `AnomalyResult`, `GenreCount`, `DecadeProfile`, `MovieHighlight` | `eda_tools.py` |
| **Artifacts** | Timestamped markdown reports saved to `outputs/` | `artifact_tools.py` → `save_report()` |

---

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Google Cloud project with Vertex AI API enabled
- A TMDB account (free)

### 1. Install dependencies

```bash
cd movie-agent
uv venv && source .venv/bin/activate
uv pip install -e .
```

### 2. Create `.env` file

```env
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
TMDB_API_TOKEN=<your-tmdb-v4-read-access-token>
```

**GCP setup**:
1. Go to the [Google Cloud Console](https://console.cloud.google.com)
2. Select or create a project
3. Enable the **Vertex AI API**
4. Authenticate locally: `gcloud auth application-default login`
5. Copy your project ID into `.env`

**TMDB setup**:
1. Create an account at [themoviedb.org](https://www.themoviedb.org/)
2. Go to Settings → API → Create → Developer
3. Copy the **API Read Access Token** (the long one, not the API Key) into `.env`

### 3. Run locally

```bash
uv run python app.py
```

Open http://localhost:8080

### 4. Deploy to Cloud Run

```bash
gcloud run deploy movie-analysis-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "TMDB_API_TOKEN=<token>,GOOGLE_CLOUD_PROJECT=<project>,GOOGLE_CLOUD_LOCATION=us-central1,GOOGLE_GENAI_USE_VERTEXAI=true"
```

---

## Example Questions

- "What makes a movie profitable?"
- "How have horror movies evolved over the past 20 years?"
- "What do the highest-rated films of the 2010s have in common?"
- "How does audience rating relate to box office revenue?"
- "What are the trends in trending movies this week?"

---

## Project Structure

```
movie-agent/
├── analysis_agent/
│   ├── __init__.py
│   ├── agent.py                 # Agent definitions + system prompts
│   └── tools/
│       ├── tmdb_tools.py        # TMDB API collection (4 tools)
│       ├── duckdb_tools.py      # DuckDB SQL over MovieLens (1 tool)
│       ├── eda_tools.py         # Statistical analysis (2 tools) + Pydantic schemas
│       ├── hypothesis_tools.py  # Chart.js spec generation (1 tool)
│       └── artifact_tools.py    # Markdown report saving (1 tool)
├── data/                        # MovieLens Small dataset (bundled)
├── outputs/                     # Generated analysis reports
├── app.py                       # FastAPI server
├── index.html                   # Frontend
├── pyproject.toml               # Dependencies
├── Dockerfile                   # Cloud Run deployment
└── README.md
```
