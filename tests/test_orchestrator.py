from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from src.db_init import initialize_database
from src.elo_engine import initialize_elo_history
from src.model import FEATURE_COLUMNS
from src.orchestrator import _build_team_lambdas
from src.world_cup_2026_schedule import TEAM_NAMES


class ConstantGoalModel:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape[1] == len(FEATURE_COLUMNS)
        return np.full(matrix.shape[0], 1.25)


def test_build_team_lambdas_qualifies_match_history_filter_with_match_stats(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "world_cup.duckdb"
    initialize_database(db_path=db_path, load_raw_files=False)
    initialize_elo_history(db_path=db_path)

    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            """
            INSERT INTO d_teams (
                team_id,
                team_name,
                source_file,
                loaded_at
            ) VALUES (?, ?, 'test', current_timestamp)
            """,
            [(team_name, team_name) for team_name in TEAM_NAMES.values()],
        )
        con.execute(
            """
            INSERT INTO f_matches (
                match_id,
                match_date,
                competition,
                season,
                stage,
                home_team_id,
                away_team_id,
                home_team_score,
                away_team_score,
                home_xg,
                away_xg,
                neutral_site,
                source_file,
                loaded_at
            ) VALUES (
                'm1',
                DATE '2025-06-01',
                'Friendly',
                '2025',
                NULL,
                'Mexico',
                'South Africa',
                2,
                1,
                1.8,
                0.9,
                TRUE,
                'test',
                current_timestamp
            )
            """,
        )
        con.executemany(
            """
            INSERT INTO f_match_stats (
                match_id,
                match_date,
                team_id,
                opponent_team_id,
                tournament,
                xg,
                possession_pct,
                shots,
                shots_on_target,
                corners,
                yellow_cards,
                red_cards,
                source,
                source_file,
                loaded_at
            ) VALUES (
                'm1',
                DATE '2025-06-01',
                ?,
                ?,
                'Friendly',
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                'fbref',
                'test',
                current_timestamp
            )
            """,
            [
                ("Mexico", "South Africa", 1.7, 55.0, 12.0, 5.0, 6.0, 1.0, 0.0),
                ("South Africa", "Mexico", 0.8, 45.0, 8.0, 3.0, 4.0, 2.0, 0.0),
            ],
        )
        con.execute(
            """
            INSERT INTO f_elo_history (
                match_id,
                match_date,
                competition,
                season,
                stage,
                home_team_id,
                away_team_id,
                home_rating_before,
                away_rating_before,
                home_rating_after,
                away_rating_after,
                home_expected_score,
                away_expected_score,
                home_actual_score,
                away_actual_score,
                k_factor,
                competition_weight,
                home_advantage_points,
                neutral_site,
                source_file,
                loaded_at,
                updated_at
            ) VALUES (
                'm1',
                DATE '2025-06-01',
                'Friendly',
                '2025',
                NULL,
                'Mexico',
                'South Africa',
                1500.0,
                1500.0,
                1510.0,
                1490.0,
                0.5,
                0.5,
                1.0,
                0.0,
                20.0,
                1.0,
                0.0,
                TRUE,
                'test',
                current_timestamp,
                current_timestamp
            )
            """,
        )

    team_lambdas = _build_team_lambdas(db_path=db_path, model=ConstantGoalModel())  # type: ignore[arg-type]

    assert len(team_lambdas) == 48
    assert {team.team_id for team in team_lambdas} >= {"MEX", "RSA"}
