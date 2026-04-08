"""
TMDB API collection tools (Step 1a).

Provides four FunctionTools for retrieving movie data from The Movie Database:
  - search_movies: keyword search
  - discover_movies: filter by genre, year, rating, sort
  - get_movie_details: full details + credits for a single movie
  - get_trending_movies: current trending films
"""

import json
import os

import requests
from google.adk.tools import FunctionTool

_BASE = "https://api.themoviedb.org/3"

# Genre name → TMDB genre ID mapping
_GENRE_IDS = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751,
    "fantasy": 14, "history": 36, "horror": 27, "music": 10402,
    "mystery": 9648, "romance": 10749, "science fiction": 878,
    "sci-fi": 878, "tv movie": 10770, "thriller": 53, "war": 10752,
    "western": 37,
}


def _headers() -> dict:
    token = os.environ.get("TMDB_API_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def _slim_movie(m: dict) -> dict:
    """Extract the fields we care about from a TMDB list-level movie object."""
    return {
        "tmdb_id": m.get("id"),
        "title": m.get("title"),
        "release_date": m.get("release_date", ""),
        "year": int(m["release_date"][:4]) if m.get("release_date") and len(m.get("release_date", "")) >= 4 else None,
        "popularity": m.get("popularity"),
        "vote_average": m.get("vote_average"),
        "vote_count": m.get("vote_count"),
        "genre_ids": m.get("genre_ids", []),
        "original_language": m.get("original_language"),
    }


# ---------------------------------------------------------------------------
# Tool 1: search_movies
# ---------------------------------------------------------------------------

def _search_movies(query: str, limit: int = 20) -> str:
    """Search TMDB for movies matching a keyword or title.

    Args:
        query: Search keywords (e.g. "Inception", "space adventure").
        limit: Max number of results to return (default 20, max 20).

    Returns:
        JSON string with matched movies and their metadata.
    """
    resp = requests.get(
        f"{_BASE}/search/movie",
        headers=_headers(),
        params={"query": query, "page": 1, "include_adult": False},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])[:limit]
    movies = [_slim_movie(m) for m in results]
    return json.dumps({
        "source": "TMDB API",
        "endpoint": "search/movie",
        "query": query,
        "movie_count": len(movies),
        "movies": movies,
    })


search_movies = FunctionTool(func=_search_movies)


# ---------------------------------------------------------------------------
# Tool 2: discover_movies
# ---------------------------------------------------------------------------

def _discover_movies(
    genre: str = "",
    year_min: int = 0,
    year_max: int = 0,
    sort_by: str = "popularity.desc",
    min_vote_count: int = 50,
    min_rating: float = 0,
    limit: int = 20,
    page: int = 1,
) -> str:
    """Discover movies using filters (genre, year range, rating, sort order).

    Args:
        genre: Genre name (e.g. "Horror", "Comedy", "Science Fiction"). Case-insensitive.
        year_min: Earliest release year (inclusive). 0 means no lower bound.
        year_max: Latest release year (inclusive). 0 means no upper bound.
        sort_by: Sort order. Options: popularity.desc, popularity.asc,
                 revenue.desc, revenue.asc, vote_average.desc, vote_average.asc,
                 primary_release_date.desc, primary_release_date.asc.
        min_vote_count: Minimum number of votes (default 50, filters out obscure titles).
        min_rating: Minimum vote_average (0-10).
        limit: Max results (default 20, max 20).
        page: Page number for pagination (default 1).

    Returns:
        JSON string with discovered movies and their metadata.
    """
    params: dict = {
        "sort_by": sort_by,
        "vote_count.gte": min_vote_count,
        "include_adult": False,
        "page": page,
    }
    if genre:
        gid = _GENRE_IDS.get(genre.lower())
        if gid:
            params["with_genres"] = gid
    if year_min:
        params["primary_release_date.gte"] = f"{year_min}-01-01"
    if year_max:
        params["primary_release_date.lte"] = f"{year_max}-12-31"
    if min_rating:
        params["vote_average.gte"] = min_rating

    resp = requests.get(
        f"{_BASE}/discover/movie",
        headers=_headers(),
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])[:limit]
    movies = [_slim_movie(m) for m in results]
    return json.dumps({
        "source": "TMDB API",
        "endpoint": "discover/movie",
        "filters": {"genre": genre, "year_min": year_min, "year_max": year_max,
                     "sort_by": sort_by, "min_vote_count": min_vote_count},
        "movie_count": len(movies),
        "movies": movies,
    })


discover_movies = FunctionTool(func=_discover_movies)


# ---------------------------------------------------------------------------
# Tool 3: get_movie_details
# ---------------------------------------------------------------------------

def _get_movie_details(movie_id: int) -> str:
    """Get full details for a single movie including budget, revenue, runtime, and credits.

    Args:
        movie_id: The TMDB movie ID (integer).

    Returns:
        JSON string with complete movie details including budget, revenue,
        top 5 cast members, and director.
    """
    resp = requests.get(
        f"{_BASE}/movie/{movie_id}",
        headers=_headers(),
        params={"append_to_response": "credits"},
        timeout=15,
    )
    resp.raise_for_status()
    d = resp.json()

    # Extract top cast and director from credits
    credits = d.get("credits", {})
    cast = [
        {"name": c["name"], "character": c.get("character", "")}
        for c in credits.get("cast", [])[:5]
    ]
    director = next(
        (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
        None,
    )

    genres = [g["name"] for g in d.get("genres", [])]

    movie = {
        "tmdb_id": d.get("id"),
        "title": d.get("title"),
        "release_date": d.get("release_date", ""),
        "year": int(d["release_date"][:4]) if d.get("release_date") and len(d.get("release_date", "")) >= 4 else None,
        "genres": genres,
        "runtime": d.get("runtime"),
        "budget": d.get("budget"),
        "revenue": d.get("revenue"),
        "vote_average": d.get("vote_average"),
        "vote_count": d.get("vote_count"),
        "popularity": d.get("popularity"),
        "tagline": d.get("tagline"),
        "original_language": d.get("original_language"),
        "production_companies": [c["name"] for c in d.get("production_companies", [])[:3]],
        "director": director,
        "top_cast": cast,
    }
    return json.dumps({
        "source": "TMDB API",
        "endpoint": f"movie/{movie_id}",
        "movie": movie,
    })


get_movie_details = FunctionTool(func=_get_movie_details)


# ---------------------------------------------------------------------------
# Tool 4: get_trending_movies
# ---------------------------------------------------------------------------

def _get_trending_movies(time_window: str = "week", limit: int = 20) -> str:
    """Get currently trending movies on TMDB.

    Args:
        time_window: "day" for daily trending or "week" for weekly trending.
        limit: Max results (default 20, max 20).

    Returns:
        JSON string with trending movies and their metadata.
    """
    window = time_window if time_window in ("day", "week") else "week"
    resp = requests.get(
        f"{_BASE}/trending/movie/{window}",
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])[:limit]
    movies = [_slim_movie(m) for m in results]
    return json.dumps({
        "source": "TMDB API",
        "endpoint": f"trending/movie/{window}",
        "time_window": window,
        "movie_count": len(movies),
        "movies": movies,
    })


get_trending_movies = FunctionTool(func=_get_trending_movies)
