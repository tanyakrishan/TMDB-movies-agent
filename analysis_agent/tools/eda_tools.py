"""
Exploratory Data Analysis tools (Step 2).

Two FunctionTools backed by Pydantic structured output schemas:
  - compute_stats: distributions, correlations, decade profiles
  - detect_anomalies: outliers, extremes, polarizing films
"""

import json
import math
from collections import Counter

from google.adk.tools import FunctionTool
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic schemas (structured output)
# ---------------------------------------------------------------------------

class GenreCount(BaseModel):
    genre: str
    count: int


class DecadeProfile(BaseModel):
    decade: str
    movie_count: int
    avg_rating: float
    avg_budget: float | None = None
    avg_revenue: float | None = None
    top_genre: str | None = None


class StatsResult(BaseModel):
    total_movies: int
    avg_rating: float
    rating_std: float
    avg_budget: float | None = None
    avg_revenue: float | None = None
    budget_revenue_corr: float | None = None
    top_genres: list[GenreCount]
    decade_profiles: list[DecadeProfile]
    rating_distribution: list[dict]
    scatter_points: list[dict]
    key_observations: list[str]


class MovieHighlight(BaseModel):
    title: str
    value: float
    note: str


class AnomalyResult(BaseModel):
    highest_rated: list[MovieHighlight]
    lowest_rated: list[MovieHighlight]
    most_profitable: list[MovieHighlight]
    biggest_flops: list[MovieHighlight]
    most_polarizing: list[MovieHighlight]
    narrative: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def _extract_genres(movie: dict) -> list[str]:
    """Extract genre list from either TMDB format or MovieLens pipe-delimited."""
    genres = movie.get("genres", [])
    if isinstance(genres, list) and genres and isinstance(genres[0], str):
        return genres  # already a list of strings
    if isinstance(genres, str) and genres:
        return [g.strip() for g in genres.split("|") if g.strip() and g.strip() != "(no genres listed)"]
    return []


# ---------------------------------------------------------------------------
# Tool: compute_stats
# ---------------------------------------------------------------------------

def _compute_stats(raw_data_json: str) -> str:
    """Compute summary statistics over collected movie data.

    Expects a JSON string that is a list of movie objects. Each movie may have:
      title, year, vote_average, vote_count, budget, revenue, genres, popularity.

    Computes: rating distribution, genre frequency, decade profiles,
    budget/revenue correlation, scatter points, and key observations.

    Args:
        raw_data_json: JSON string — a list of movie dicts from the collection step.

    Returns:
        JSON string containing structured StatsResult with all computed statistics.
    """
    movies = json.loads(raw_data_json)
    if not isinstance(movies, list):
        movies = movies.get("movies", [])

    total = len(movies)
    if total == 0:
        return StatsResult(
            total_movies=0, avg_rating=0, rating_std=0,
            top_genres=[], decade_profiles=[], rating_distribution=[],
            scatter_points=[], key_observations=["No movies in dataset"],
        ).model_dump_json()

    # --- Ratings ---
    ratings = [_safe_float(m.get("vote_average")) for m in movies]
    ratings = [r for r in ratings if r is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0
    rating_std = round(
        math.sqrt(sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)), 2
    ) if len(ratings) > 1 else 0

    # Rating distribution buckets
    buckets = {f"{i}-{i+1}": 0 for i in range(0, 10)}
    for r in ratings:
        bucket_idx = min(int(r), 9)
        key = f"{bucket_idx}-{bucket_idx + 1}"
        buckets[key] += 1
    rating_dist = [{"bucket": k, "count": v} for k, v in buckets.items() if v > 0]

    # --- Genres ---
    genre_counter: Counter = Counter()
    for m in movies:
        for g in _extract_genres(m):
            genre_counter[g] += 1
    top_genres = [GenreCount(genre=g, count=c) for g, c in genre_counter.most_common(10)]

    # --- Budget / Revenue ---
    budgets = [_safe_float(m.get("budget")) for m in movies]
    revenues = [_safe_float(m.get("revenue")) for m in movies]
    valid_budgets = [b for b in budgets if b and b > 0]
    valid_revenues = [r for r in revenues if r and r > 0]
    avg_budget = round(sum(valid_budgets) / len(valid_budgets), 0) if valid_budgets else None
    avg_revenue = round(sum(valid_revenues) / len(valid_revenues), 0) if valid_revenues else None

    # Budget-revenue correlation
    br_pairs = [
        (float(m["budget"]), float(m["revenue"]))
        for m in movies
        if _safe_float(m.get("budget")) and m["budget"] > 0
        and _safe_float(m.get("revenue")) and m["revenue"] > 0
    ]
    budget_revenue_corr = _pearson(
        [p[0] for p in br_pairs], [p[1] for p in br_pairs]
    ) if br_pairs else None

    # --- Decade profiles ---
    decade_groups: dict[str, list] = {}
    for m in movies:
        year = m.get("year")
        if not year:
            continue
        decade = f"{(year // 10) * 10}s"
        decade_groups.setdefault(decade, []).append(m)

    decade_profiles = []
    for decade, group in sorted(decade_groups.items()):
        d_ratings = [_safe_float(m.get("vote_average")) for m in group]
        d_ratings = [r for r in d_ratings if r is not None]
        d_budgets = [_safe_float(m.get("budget")) for m in group if _safe_float(m.get("budget")) and m.get("budget", 0) > 0]
        d_revenues = [_safe_float(m.get("revenue")) for m in group if _safe_float(m.get("revenue")) and m.get("revenue", 0) > 0]

        d_genres: Counter = Counter()
        for m in group:
            for g in _extract_genres(m):
                d_genres[g] += 1

        decade_profiles.append(DecadeProfile(
            decade=decade,
            movie_count=len(group),
            avg_rating=round(sum(d_ratings) / len(d_ratings), 2) if d_ratings else 0,
            avg_budget=round(sum(d_budgets) / len(d_budgets), 0) if d_budgets else None,
            avg_revenue=round(sum(d_revenues) / len(d_revenues), 0) if d_revenues else None,
            top_genre=d_genres.most_common(1)[0][0] if d_genres else None,
        ))

    # --- Scatter points ---
    scatter = []
    for m in movies:
        x = m.get("year")
        y = _safe_float(m.get("vote_average"))
        if x and y is not None:
            scatter.append({"x": x, "y": y, "label": m.get("title", "Unknown")})

    # --- Key observations ---
    observations = []
    if ratings:
        observations.append(f"Average rating: {avg_rating}/10 (std: {rating_std})")
    if top_genres:
        observations.append(f"Most common genre: {top_genres[0].genre} ({top_genres[0].count} movies)")
    if avg_budget:
        observations.append(f"Average budget: ${avg_budget:,.0f}")
    if avg_revenue:
        observations.append(f"Average revenue: ${avg_revenue:,.0f}")
    if budget_revenue_corr is not None:
        observations.append(f"Budget-revenue correlation: {budget_revenue_corr}")
    if decade_profiles:
        best = max(decade_profiles, key=lambda d: d.avg_rating)
        observations.append(f"Highest-rated decade: {best.decade} ({best.avg_rating}/10)")

    result = StatsResult(
        total_movies=total,
        avg_rating=avg_rating,
        rating_std=rating_std,
        avg_budget=avg_budget,
        avg_revenue=avg_revenue,
        budget_revenue_corr=budget_revenue_corr,
        top_genres=top_genres,
        decade_profiles=decade_profiles,
        rating_distribution=rating_dist,
        scatter_points=scatter[:100],
        key_observations=observations,
    )
    return result.model_dump_json()


compute_stats = FunctionTool(func=_compute_stats)


# ---------------------------------------------------------------------------
# Tool: detect_anomalies
# ---------------------------------------------------------------------------

def _detect_anomalies(raw_data_json: str) -> str:
    """Detect outliers and notable anomalies in collected movie data.

    Expects a JSON string that is a list of movie objects. Surfaces:
      - highest/lowest rated movies
      - most profitable movies (highest ROI)
      - biggest flops (high budget, low revenue)
      - most polarizing (high vote count but mediocre rating)

    Args:
        raw_data_json: JSON string — a list of movie dicts from the collection step.

    Returns:
        JSON string containing structured AnomalyResult with notable findings.
    """
    movies = json.loads(raw_data_json)
    if not isinstance(movies, list):
        movies = movies.get("movies", [])

    if not movies:
        return AnomalyResult(
            highest_rated=[], lowest_rated=[], most_profitable=[],
            biggest_flops=[], most_polarizing=[],
            narrative="No movies to analyze.",
        ).model_dump_json()

    # Filter to movies with ratings
    rated = [m for m in movies if _safe_float(m.get("vote_average")) is not None]
    rated.sort(key=lambda m: m.get("vote_average", 0), reverse=True)

    highest = [
        MovieHighlight(
            title=m.get("title", "Unknown"),
            value=m["vote_average"],
            note=f"Rating: {m['vote_average']}/10, {m.get('vote_count', 'N/A')} votes",
        )
        for m in rated[:5]
    ]
    lowest = [
        MovieHighlight(
            title=m.get("title", "Unknown"),
            value=m["vote_average"],
            note=f"Rating: {m['vote_average']}/10, {m.get('vote_count', 'N/A')} votes",
        )
        for m in rated[-5:]
    ] if len(rated) >= 5 else []

    # Profitability (ROI = (revenue - budget) / budget)
    with_financials = [
        m for m in movies
        if _safe_float(m.get("budget")) and m["budget"] > 0
        and _safe_float(m.get("revenue")) and m["revenue"] > 0
    ]

    most_profitable = []
    biggest_flops = []
    if with_financials:
        for m in with_financials:
            m["_roi"] = (m["revenue"] - m["budget"]) / m["budget"]

        by_roi = sorted(with_financials, key=lambda m: m["_roi"], reverse=True)
        most_profitable = [
            MovieHighlight(
                title=m.get("title", "Unknown"),
                value=round(m["_roi"] * 100, 1),
                note=f"ROI: {m['_roi'] * 100:.1f}% (budget: ${m['budget']:,.0f}, revenue: ${m['revenue']:,.0f})",
            )
            for m in by_roi[:5]
        ]
        biggest_flops = [
            MovieHighlight(
                title=m.get("title", "Unknown"),
                value=round(m["_roi"] * 100, 1),
                note=f"ROI: {m['_roi'] * 100:.1f}% (budget: ${m['budget']:,.0f}, revenue: ${m['revenue']:,.0f})",
            )
            for m in by_roi[-5:]
        ] if len(by_roi) >= 5 else []

        # Clean up temp field
        for m in with_financials:
            del m["_roi"]

    # Polarizing: high vote count but mediocre rating (4-7 range)
    polarizing_candidates = [
        m for m in rated
        if m.get("vote_count", 0) > 1000 and 4 <= m.get("vote_average", 0) <= 7
    ]
    polarizing_candidates.sort(key=lambda m: m.get("vote_count", 0), reverse=True)
    most_polarizing = [
        MovieHighlight(
            title=m.get("title", "Unknown"),
            value=m["vote_average"],
            note=f"Rating: {m['vote_average']}/10 despite {m['vote_count']:,} votes — widely seen but divisive",
        )
        for m in polarizing_candidates[:5]
    ]

    # Build narrative
    parts = []
    if highest:
        parts.append(f"Top-rated: {highest[0].title} ({highest[0].value}/10)")
    if lowest:
        parts.append(f"Lowest-rated: {lowest[-1].title} ({lowest[-1].value}/10)")
    if most_profitable:
        parts.append(f"Most profitable: {most_profitable[0].title} ({most_profitable[0].value}% ROI)")
    if biggest_flops:
        parts.append(f"Biggest flop: {biggest_flops[-1].title} ({biggest_flops[-1].value}% ROI)")
    if most_polarizing:
        parts.append(f"Most polarizing: {most_polarizing[0].title} ({most_polarizing[0].value}/10 with {polarizing_candidates[0].get('vote_count', 0):,} votes)")

    narrative = " | ".join(parts) if parts else "No clear anomalies detected."

    result = AnomalyResult(
        highest_rated=highest,
        lowest_rated=lowest,
        most_profitable=most_profitable,
        biggest_flops=biggest_flops,
        most_polarizing=most_polarizing,
        narrative=narrative,
    )
    return result.model_dump_json()


detect_anomalies = FunctionTool(func=_detect_anomalies)
