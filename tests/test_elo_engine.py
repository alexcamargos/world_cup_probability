from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pytest

from src.db_init import initialize_database
from src.elo_engine import _competition_weight, _goal_margin_multiplier, build_elo_history


def test_build_elo_history_upserts_existing_match(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="src.elo_engine")
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"
    initialize_database(db_path=db_path, load_raw_files=False)

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO d_teams (team_id, team_name, loaded_at)
            VALUES
                ('Brazil', 'Brazil', current_timestamp),
                ('Argentina', 'Argentina', current_timestamp),
                ('Scotland', 'Scotland', current_timestamp),
                ('Morocco', 'Morocco', current_timestamp)
            """
        )
        con.execute(
            """
            INSERT INTO f_matches (
                match_id,
                match_date,
                competition,
                home_team_id,
                away_team_id,
                home_team_score,
                away_team_score,
                neutral_site
            ) VALUES
                ('m1', DATE '2026-06-20', 'Friendly', 'Brazil', 'Argentina', 2, 1, TRUE),
                ('m_wc', DATE '2026-06-20', 'FIFA World Cup', 'Scotland', 'Morocco', 2, 1, TRUE)
            """
        )

    build_elo_history(db_path=db_path)

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO f_elo_history (
                match_id,
                match_date,
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
                updated_at
            ) VALUES (
                'stale',
                DATE '2020-01-01',
                'Old A',
                'Old B',
                1500.0,
                1500.0,
                1500.0,
                1500.0,
                0.5,
                0.5,
                0.5,
                0.5,
                20.0,
                1.0,
                0.0,
                current_timestamp
            )
            """
        )

    build_elo_history(db_path=db_path)

    with duckdb.connect(str(db_path)) as con:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                max(home_rating_before) AS home_rating_before,
                max(away_rating_before) AS away_rating_before,
                max(home_actual_score) AS home_actual_score,
                max(away_actual_score) AS away_actual_score,
                bool_and(updated_at IS NOT NULL) AS has_updated_at
            FROM f_elo_history
            WHERE match_id = 'm1'
            """
        ).fetchone()
        world_cup_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM f_elo_history
            WHERE match_id = 'm_wc'
            """
        ).fetchone()[0]
        stale_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM f_elo_history
            WHERE match_id = 'stale'
            """
        ).fetchone()[0]

    assert row == (1, 1500.0, 1500.0, 1.0, 0.0, True)
    assert world_cup_rows == 0
    assert stale_rows == 0
    assert "Elo progress: match 1/1 processed (100.0% complete)." in caplog.text


def test_dynamic_elo_stores_margin_and_experience_adjusted_k(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"
    initialize_database(db_path=db_path, load_raw_files=False)

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO d_teams (team_id, team_name, loaded_at)
            VALUES
                ('Brazil', 'Brazil', current_timestamp),
                ('Argentina', 'Argentina', current_timestamp)
            """
        )
        con.execute(
            """
            INSERT INTO f_matches (
                match_id,
                match_date,
                competition,
                home_team_id,
                away_team_id,
                home_team_score,
                away_team_score,
                neutral_site
            ) VALUES (
                'm_blowout',
                DATE '2025-06-20',
                'Friendly',
                'Brazil',
                'Argentina',
                4,
                0,
                FALSE
            )
            """
        )

    build_elo_history(db_path=db_path, calibrate_parameters=False)

    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute(
            """
            SELECT
                base_k_factor,
                competition_weight,
                goal_margin,
                goal_margin_multiplier,
                experience_multiplier,
                k_factor,
                home_advantage_points,
                elo_parameter_version
            FROM f_elo_history
            WHERE match_id = 'm_blowout'
            """
        ).fetchone()

    assert row[0] == pytest.approx(20.0)
    assert row[1] == pytest.approx(0.5)
    assert row[2] == 4
    assert row[3] > 1.0
    assert row[4] == pytest.approx(1.35)
    assert row[5] > 20.0 * 0.5
    assert row[6] == pytest.approx(100.0)
    assert row[7] == "dynamic_elo_v1"


def test_world_cup_qualifier_weight_is_more_specific_than_world_cup() -> None:
    assert _competition_weight("FIFA World Cup") == pytest.approx(2.5)
    assert _competition_weight("FIFA World Cup Qualifier") == pytest.approx(2.0)


def test_goal_margin_multiplier_rewards_decisive_upsets_more_than_favorite_wins() -> None:
    favorite_blowout = _goal_margin_multiplier(
        4,
        0,
        home_rating=1800.0,
        away_rating=1400.0,
        home_advantage=100.0,
        exponent=0.6,
    )
    upset_blowout = _goal_margin_multiplier(
        0,
        4,
        home_rating=1800.0,
        away_rating=1400.0,
        home_advantage=100.0,
        exponent=0.6,
    )

    assert upset_blowout > favorite_blowout
    assert _goal_margin_multiplier(
        1,
        0,
        home_rating=1500.0,
        away_rating=1500.0,
        home_advantage=0.0,
        exponent=0.6,
    ) == pytest.approx(1.0)
