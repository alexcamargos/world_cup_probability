"""Shared filters for excluding current tournament leakage from training inputs."""

from __future__ import annotations

from datetime import date

CURRENT_WORLD_CUP_YEAR = 2026


def is_current_world_cup_match(match_date: date, competition: str | None) -> bool:
    """Return whether a match belongs to the current World Cup tournament data.

    Args:
        match_date: The date of the match.
        competition: The name of the competition.

    Returns:
        True if the match belongs to the current World Cup tournament, False otherwise.
    """
    return match_date.year == CURRENT_WORLD_CUP_YEAR and _is_world_cup_competition(competition)


def current_world_cup_exclusion_sql(
    *,
    date_expr: str,
    competition_expr: str,
) -> str:
    """DuckDB SQL predicate that is false for current World Cup rows.

    Args:
        date_expr: The SQL expression representing the match date.
        competition_expr: The SQL expression representing the competition.

    Returns:
        A SQL predicate string that filters out current World Cup matches.
    """
    return (
        "NOT ("
        f"EXTRACT(year FROM {date_expr}) = {CURRENT_WORLD_CUP_YEAR} "
        f"AND {_world_cup_competition_sql(competition_expr)}"
        ")"
    )


def _is_world_cup_competition(competition: str | None) -> bool:
    """Check if the competition name matches the World Cup (excluding qualifiers).

    Args:
        competition: The name of the competition.

    Returns:
        True if the competition is the World Cup tournament, False otherwise.
    """
    if competition is None:
        return False
    normalized = competition.casefold()
    has_wc = "world cup" in normalized or "copa do mundo" in normalized
    has_qualifier = (
        "qualifier" in normalized
        or "qualifying" in normalized
        or "qualification" in normalized
        or "qualific" in normalized
        or "eliminat" in normalized
        or "preliminar" in normalized
    )
    return has_wc and not has_qualifier


def _world_cup_competition_sql(competition_expr: str) -> str:
    """Generate SQL expression to check for the World Cup (excluding qualifiers).

    Args:
        competition_expr: The SQL expression representing the competition column.

    Returns:
        A SQL predicate string.
    """
    lowered = f"lower(coalesce({competition_expr}, ''))"
    return (
        f"({lowered} LIKE '%world cup%' OR {lowered} LIKE '%copa do mundo%') "
        "AND NOT ("
        f"{lowered} LIKE '%qualifier%' "
        f"OR {lowered} LIKE '%qualifying%' "
        f"OR {lowered} LIKE '%qualification%' "
        f"OR {lowered} LIKE '%qualific%' "
        f"OR {lowered} LIKE '%eliminat%' "
        f"OR {lowered} LIKE '%preliminar%'"
        ")"
    )
