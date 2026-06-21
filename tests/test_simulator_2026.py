from __future__ import annotations

from pathlib import Path

import duckdb

from src.simulator import TeamLambda, simulate_world_cup
from src.world_cup_2026_schedule import TEAM_COUNTRIES, TEAM_NAMES, world_cup_2026_fixtures


def test_world_cup_2026_schedule_contains_official_tournament_shape() -> None:
    fixtures = world_cup_2026_fixtures()

    assert len(fixtures) == 104
    assert sum(fixture.round_name == "group_stage" for fixture in fixtures) == 72
    assert sum(fixture.round_name == "round_of_32" for fixture in fixtures) == 16
    assert fixtures[0].match_number == 1
    assert fixtures[0].home_slot == "MEX"
    assert fixtures[0].away_slot == "RSA"
    assert fixtures[-1].round_name == "final"


def test_simulate_world_cup_uses_groups_schedule_and_knockout_path(tmp_path: Path) -> None:
    db_path = tmp_path / "world_cup.duckdb"
    teams = [
        TeamLambda(
            team_id=code,
            team_name=name,
            lambda_goals=1.2,
            country_code=TEAM_COUNTRIES.get(code, code),
        )
        for code, name in TEAM_NAMES.items()
    ]

    simulate_world_cup(teams, iterations=1, batch_size=32, db_path=db_path, seed=7)

    with duckdb.connect(str(db_path), read_only=True) as con:
        total_rows = con.execute("SELECT COUNT(*) FROM simulated_results").fetchone()[0]
        final_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM simulated_results
            WHERE round_name = 'final'
            """
        ).fetchone()[0]
        mexico_opening = con.execute(
            """
            SELECT home_goals, away_goals, stadium, home_advantage
            FROM simulated_results
            WHERE match_number = 1
            """
        ).fetchone()
        rest_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM simulated_results
            WHERE home_rest_days IS NOT NULL OR away_rest_days IS NOT NULL
            """
        ).fetchone()[0]

    assert total_rows == 104
    assert final_rows == 1
    assert mexico_opening == (2, 0, "Mexico City Stadium", True)
    assert rest_rows > 0
