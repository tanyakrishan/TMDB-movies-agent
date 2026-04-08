"""
Two-agent system: orchestrator + hypothesis specialist.
Orchestrated at the app level (separate runners) instead of AgentTool,
because AgentTool function_responses are not reliably yielded through
FastAPI's async event loop.

root_agent: Collect → Analyze (Steps 1-2)
hypothesis_agent: Hypothesize (Step 3) — receives EDA output, writes narrative
"""

from google.adk.agents import Agent

from .tools.tmdb_tools import (
    search_movies,
    discover_movies,
    get_movie_details,
    get_trending_movies,
)
from .tools.duckdb_tools import query_ratings
from .tools.eda_tools import compute_stats, detect_anomalies

# ---------------------------------------------------------------------------
# Agent 1: Root Orchestrator (collection + EDA)
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.0-flash",
    name="root_agent",
    description="Collects and analyzes movie data from TMDB and MovieLens.",
    instruction="""You are a movie analysis agent. Follow these steps:

STEP 1 - COLLECT: Based on the user's question, gather movie data:
- search_movies: find movies by name/keyword
- discover_movies: filter by genre, year, rating, sort order
  (genres: Action, Comedy, Drama, Horror, Science Fiction, Thriller, etc.)
  (sort_by: popularity.desc, revenue.desc, vote_average.desc, etc.)
- get_movie_details: budget, revenue, cast (call for top 5-10 movies only)
- get_trending_movies: current trending films
- query_ratings: SQL over MovieLens (100K ratings, 9.7K movies, user tags)

STEP 2 - ANALYZE: Pass collected movies as a JSON list string to BOTH:
- compute_stats(raw_data_json)
- detect_anomalies(raw_data_json)

STEP 3 - RESPOND: Write a data-grounded summary with specific numbers.
Write 3-5 sentences answering the question, then 4-6 bullet points citing data.

IMPORTANT:
- Always complete all steps. Never skip the analysis.
- Today is 2026. Movies from 2025-2026 are current releases.
- For greetings or non-movie questions, answer directly without the pipeline.
""",
    tools=[
        search_movies,
        discover_movies,
        get_movie_details,
        get_trending_movies,
        query_ratings,
        compute_stats,
        detect_anomalies,
    ],
)

# ---------------------------------------------------------------------------
# Agent 2: Hypothesis Specialist (narrative writer)
# ---------------------------------------------------------------------------

hypothesis_agent = Agent(
    model="gemini-2.0-flash",
    name="hypothesis_agent",
    description="Forms a data-grounded hypothesis from movie EDA findings.",
    instruction="""You are a movie analyst specializing in forming hypotheses from data.

You receive a user's question along with structured EDA findings (summary_stats
and anomalies). Your job is to write a compelling, data-grounded analysis.

Your output must have two parts:

PART 1 — HYPOTHESIS (3-5 sentences):
Directly answer the user's question using numbers from the data. Make it feel
like a real cinema industry insight, not a dry report.

PART 2 — EVIDENCE (4-6 bullet points):
Each bullet cites a specific number from the data:
- "Average rating: 6.3/10 (std: 2.2)"
- "Top genre: Drama (8 movies)"
- "Highest rated: Project Hail Mary (8.2/10, 1144 votes)"
- "Most polarizing: Mercy (5.5/10 with 2000+ votes)"

Be concise and specific. Every claim must be backed by a number from the data.
""",
)
