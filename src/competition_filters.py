"""Shared filters for excluding current tournament leakage from training inputs."""

from __future__ import annotations

from datetime import date

CURRENT_WORLD_CUP_YEAR = 2026


def is_current_world_cup_match(match_date: date, competition: str | None) -> bool:
    """Return whether a match belongs to the current World Cup tournament data."""
    return match_date.year == CURRENT_WORLD_CUP_YEAR and _is_world_cup_competition(competition)


def current_world_cup_exclusion_sql(
    *,
    date_expr: str,
    competition_expr: str,
) -> str:
    """DuckDB SQL predicate that is false for current World Cup rows."""
    return (
        "NOT ("
        f"EXTRACT(year FROM {date_expr}) = {CURRENT_WORLD_CUP_YEAR} "
        f"AND {_world_cup_competition_sql(competition_expr)}"
        ")"
    )


def _is_world_cup_competition(competition: str | None) -> bool:
    if competition is None:
        return False
    normalized = competition.casefold()
    return "world cup" in normalized or "copa do mundo" in normalized


def _world_cup_competition_sql(competition_expr: str) -> str:
    lowered = f"lower(coalesce({competition_expr}, ''))"
    return f"({lowered} LIKE '%world cup%' OR {lowered} LIKE '%copa do mundo%')"
