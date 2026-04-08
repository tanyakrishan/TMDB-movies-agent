"""
DuckDB SQL tool over MovieLens dataset (Step 1b).

Registers the four MovieLens CSVs as SQL views and exposes a single
FunctionTool that lets the agent write and execute analytical SQL queries.
"""

import json
import os
import re

import duckdb
from google.adk.tools import FunctionTool

# ---------------------------------------------------------------------------
# DuckDB initialisation (module-level singleton)
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_con = duckdb.connect(":memory:")

for _view, _file in [
    ("ratings", "ratings.csv"),
    ("movies", "movies.csv"),
    ("tags", "tags.csv"),
    ("links", "links.csv"),
]:
    _path = os.path.join(_DATA_DIR, _file)
    _con.execute(f"CREATE VIEW {_view} AS SELECT * FROM read_csv_auto('{_path}')")

# Forbidden SQL keywords (prevent writes / schema changes)
_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tool: query_ratings
# ---------------------------------------------------------------------------

def _query_ratings(sql: str) -> str:
    """Execute a read-only SQL query against the MovieLens dataset using DuckDB.

    Available tables and their schemas:

      ratings(userId INT, movieId INT, rating FLOAT, timestamp BIGINT)
        — 100,836 individual user ratings (0.5 to 5.0 scale)

      movies(movieId INT, title VARCHAR, genres VARCHAR)
        — 9,742 movies. The 'genres' column is pipe-delimited,
          e.g. "Action|Adventure|Sci-Fi". Use LIKE or string_split()
          to filter by genre.

      tags(userId INT, movieId INT, tag VARCHAR, timestamp BIGINT)
        — 3,683 free-text tags users applied to movies,
          e.g. "dark comedy", "twist ending", "based on a book".

      links(movieId INT, imdbId INT, tmdbId INT)
        — Bridge table mapping MovieLens movieId to IMDB and TMDB IDs.
          Use this to join MovieLens data with TMDB API results.

    Example queries:
      "SELECT m.title, AVG(r.rating) as avg_rating, COUNT(*) as num_ratings
       FROM ratings r JOIN movies m ON r.movieId = m.movieId
       GROUP BY m.title HAVING COUNT(*) >= 50
       ORDER BY avg_rating DESC LIMIT 20"

      "SELECT l.tmdbId, m.title, AVG(r.rating) as avg_rating
       FROM ratings r JOIN movies m ON r.movieId = m.movieId
       JOIN links l ON m.movieId = l.movieId
       WHERE l.tmdbId IN (550, 680, 155)
       GROUP BY l.tmdbId, m.title"

    Args:
        sql: A SELECT query to execute. Only read operations are allowed.

    Returns:
        JSON string with columns, rows, and row count.
    """
    # Safety: block write/DDL operations
    if _BLOCKED.search(sql):
        return json.dumps({"error": "Only SELECT queries are allowed."})

    # Enforce LIMIT cap
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        sql = sql.rstrip().rstrip(";") + " LIMIT 500"

    try:
        result = _con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = [list(row) for row in result.fetchall()]
        return json.dumps({
            "source": "MovieLens (DuckDB)",
            "sql": sql,
            "row_count": len(rows),
            "columns": columns,
            "rows": rows,
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "sql": sql})


query_ratings = FunctionTool(func=_query_ratings)
