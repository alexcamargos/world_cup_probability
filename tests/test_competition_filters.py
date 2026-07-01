"""Tests for the competition filter functions."""

from datetime import date

import duckdb

from src.competition_filters import (
    current_world_cup_exclusion_sql,
    is_current_world_cup_match,
)


def test_is_current_world_cup_match() -> None:
    """Test Python logic for detecting current World Cup matches.

    Args:
        None

    Returns:
        None
    """
    # Matches that should be recognized as current World Cup matches
    # (year 2026, World Cup tournament)
    assert is_current_world_cup_match(date(2026, 6, 15), "FIFA World Cup") is True
    assert is_current_world_cup_match(date(2026, 7, 10), "Copa do Mundo da FIFA") is True

    # Matches in 2026 that are World Cup Qualifiers
    # (should NOT be classified as current World Cup matches)
    assert is_current_world_cup_match(date(2026, 3, 25), "FIFA World Cup Qualifier") is False
    assert is_current_world_cup_match(date(2026, 3, 26), "FIFA World Cup qualification") is False
    assert is_current_world_cup_match(date(2026, 3, 27), "Copa do Mundo Eliminatórias") is False
    assert is_current_world_cup_match(date(2026, 3, 28), "World Cup Qualifying") is False
    assert (
        is_current_world_cup_match(date(2026, 3, 29), "FIFA World Cup Preliminary Competition")
        is False
    )

    # Matches in other years (should NOT be classified as current World Cup matches)
    assert is_current_world_cup_match(date(2022, 12, 18), "FIFA World Cup") is False
    assert is_current_world_cup_match(date(2026, 6, 15), "Friendly") is False


def test_current_world_cup_exclusion_sql() -> None:
    """Test DuckDB SQL exclusion predicate behaves correctly.

    Args:
        None

    Returns:
        None
    """
    # Create an in-memory database to test the SQL logic
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE test_matches (
            match_date DATE,
            competition VARCHAR
        )
        """
    )

    # Insert test cases
    conn.execute(
        """
        INSERT INTO test_matches VALUES
        ('2026-06-15', 'FIFA World Cup'),
        ('2026-07-10', 'Copa do Mundo da FIFA'),
        ('2026-03-25', 'FIFA World Cup Qualifier'),
        ('2026-03-26', 'FIFA World Cup qualification'),
        ('2026-03-27', 'Copa do Mundo Eliminatórias'),
        ('2026-03-28', 'World Cup Qualifying'),
        ('2026-03-29', 'FIFA World Cup Preliminary Competition'),
        ('2022-12-18', 'FIFA World Cup'),
        ('2026-06-15', 'Friendly')
        """
    )

    exclusion_sql = current_world_cup_exclusion_sql(
        date_expr="match_date", competition_expr="competition"
    )

    # Under this exclusion predicate, we only exclude 2026 matches that are the main World
    # Cup tournament. Therefore, 2026-06-15 'FIFA World Cup' and 2026-07-10 'Copa do
    # Mundo da FIFA' should be excluded. All qualifiers/eliminatórias and other years or
    # friendlies should be included.
    query = f"SELECT match_date, competition FROM test_matches WHERE {exclusion_sql}"
    results = conn.execute(query).fetchall()

    expected_included = {
        (date(2026, 3, 25), "FIFA World Cup Qualifier"),
        (date(2026, 3, 26), "FIFA World Cup qualification"),
        (date(2026, 3, 27), "Copa do Mundo Eliminatórias"),
        (date(2026, 3, 28), "World Cup Qualifying"),
        (date(2026, 3, 29), "FIFA World Cup Preliminary Competition"),
        (date(2022, 12, 18), "FIFA World Cup"),
        (date(2026, 6, 15), "Friendly"),
    }

    assert set(results) == expected_included
