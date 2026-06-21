from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from src.app import (
    _add_played_fixture_result,
    _format_actual_score,
    _format_score,
    _load_round_probabilities,
)
from src.world_cup_2026_schedule import WorldCupFixture


def test_load_round_probabilities_includes_modal_model_score(tmp_path: Path) -> None:
    db_path = tmp_path / "world_cup.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE simulated_results (
                simulation_id INTEGER NOT NULL,
                match_number INTEGER NOT NULL,
                group_name VARCHAR,
                home_team_id VARCHAR NOT NULL,
                home_team_name VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                away_team_name VARCHAR NOT NULL,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL
            )
            """,
        )
        con.executemany(
            """
            INSERT INTO simulated_results (
                simulation_id,
                match_number,
                group_name,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_goals,
                away_goals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 2, 1),
                (2, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 2, 1),
                (3, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 1, 0),
                (4, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 1, 0),
                (5, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 0, 0),
            ],
        )

    rows = _load_round_probabilities(
        str(db_path),
        (1,),
        show_all_matchups=True,
        dynamic_matchups=False,
    )

    assert len(rows) == 1
    assert rows[0]["predicted_home_goals"] == 1
    assert rows[0]["predicted_away_goals"] == 0
    assert rows[0]["score_occurrence_pct"] == 40.0
    assert _format_score(rows[0]) == "1 x 0"


def test_add_played_fixture_result_marks_winner_hit() -> None:
    row = {
        "match_number": 1,
        "home_team_id": "BRA",
        "away_team_id": "ARG",
        "predicted_home_goals": 2,
        "predicted_away_goals": 1,
    }
    fixture = WorldCupFixture(
        match_number=1,
        match_date=datetime(2026, 6, 11, tzinfo=UTC),
        round_name="group_stage",
        group_name="Group A",
        home_slot="BRA",
        away_slot="ARG",
        stadium="Test Stadium",
        city="Test City",
        country="USA",
        played_home_goals=1,
        played_away_goals=0,
    )

    _add_played_fixture_result(row, {1: fixture})

    assert _format_actual_score(row) == "1 x 0"
    assert row["prediction_result"] == "Acerto"


def test_add_played_fixture_result_ignores_non_matching_fixture_teams() -> None:
    row = {
        "match_number": 1,
        "home_team_id": "BRA",
        "away_team_id": "ARG",
        "predicted_home_goals": 2,
        "predicted_away_goals": 1,
    }
    fixture = WorldCupFixture(
        match_number=1,
        match_date=datetime(2026, 6, 11, tzinfo=UTC),
        round_name="group_stage",
        group_name="Group A",
        home_slot="ARG",
        away_slot="BRA",
        stadium="Test Stadium",
        city="Test City",
        country="USA",
        played_home_goals=0,
        played_away_goals=1,
    )

    _add_played_fixture_result(row, {1: fixture})

    assert _format_actual_score(row) == "-"
    assert row["prediction_result"] is None
