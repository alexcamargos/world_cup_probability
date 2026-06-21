from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from src.db_init import initialize_database
from src.elo_engine import initialize_elo_history
from src.feature_pipeline import build_feature_frame


def test_build_feature_frame_includes_prior_world_cup_diffs(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"
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
            [("Brazil", "Brazil"), ("Argentina", "Argentina")],
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
                neutral_site,
                source_file,
                loaded_at
            ) VALUES (
                'm1',
                DATE '2014-06-01',
                'Friendly',
                '2014',
                NULL,
                'Brazil',
                'Argentina',
                2,
                1,
                TRUE,
                'test',
                current_timestamp
            )
            """,
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
                DATE '2014-06-01',
                'Friendly',
                '2014',
                NULL,
                'Brazil',
                'Argentina',
                1600.0,
                1550.0,
                1610.0,
                1540.0,
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
        con.executemany(
            """
            INSERT INTO d_world_cup_prior_team_history (
                team_id,
                team_name,
                team_code,
                normalized_team_name,
                as_of_year,
                prior_world_cup_appearances,
                prior_world_cup_points_per_match,
                prior_world_cup_goal_diff_per_match,
                source_file,
                loaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', current_timestamp)
            """,
            [
                ("T-01", "Brazil", "BRA", "brazil", 2014, 19, 2.0, 1.1),
                ("T-02", "Argentina", "ARG", "argentina", 2014, 15, 1.6, 0.7),
            ],
        )
        con.executemany(
            """
            INSERT INTO d_world_cup_prior_discipline_history (
                team_id,
                team_name,
                team_code,
                normalized_team_name,
                as_of_year,
                prior_world_cup_yellow_cards_per_match,
                prior_world_cup_sending_offs_per_match,
                prior_world_cup_fair_play_penalty_per_match,
                source_file,
                loaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test', current_timestamp)
            """,
            [
                ("T-01", "Brazil", "BRA", "brazil", 2014, 1.4, 0.1, 1.8),
                ("T-02", "Argentina", "ARG", "argentina", 2014, 1.1, 0.2, 1.7),
            ],
        )

    frame = build_feature_frame(db_path=db_path)
    row = frame.row(0, named=True)

    assert row["prior_world_cup_appearances_diff"] == 4.0
    assert row["prior_world_cup_points_per_match_diff"] == pytest.approx(0.4)
    assert row["prior_world_cup_goal_diff_per_match_diff"] == pytest.approx(0.4)
    assert row["prior_world_cup_yellow_cards_per_match_diff"] == pytest.approx(0.3)
    assert row["prior_world_cup_sending_offs_per_match_diff"] == pytest.approx(-0.1)
    assert row["prior_world_cup_fair_play_penalty_per_match_diff"] == pytest.approx(0.1)
