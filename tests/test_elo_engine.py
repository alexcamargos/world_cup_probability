from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pytest

from src.db_init import initialize_database
from src.elo_engine import (
    EloParameters,
    MatchRecord,
    _calibrate_elo_parameters,
    _competition_weight,
    _compute_matches_signature,
    _goal_margin_multiplier,
    build_elo_history,
)


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
    """Verify that World Cup qualifiers and qualifications map to the qualifier weight rather than main tournament weight.

    Args:
        None

    Returns:
        None
    """
    assert _competition_weight("FIFA World Cup") == pytest.approx(2.5)
    assert _competition_weight("FIFA World Cup Qualifier") == pytest.approx(2.0)
    assert _competition_weight("FIFA World Cup qualification") == pytest.approx(2.0)
    assert _competition_weight("FIFA World Cup Qualification") == pytest.approx(2.0)
    assert _competition_weight("FIFA World Cup qualifying") == pytest.approx(2.0)


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


def test_elo_calibration_caching(tmp_path: Path) -> None:
    """Test that ELO calibration correctly caches parameters and loads from cache.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # Create sample matches
    matches = [
        MatchRecord(
            match_id=f"m{i}",
            match_date=f"2026-06-{i:02d}",
            competition="Friendly",
            season="2026",
            stage="Group",
            home_team_id="TeamA",
            away_team_id="TeamB",
            home_team_score=1,
            away_team_score=0,
            neutral_site=False,
        )
        for i in range(1, 100)  # 99 matches to satisfy min_matches=80
    ]

    default_params = EloParameters()
    cache_file = tmp_path / "elo_calibration_cache.json"

    # 1. Verify matches signature is computed correctly and is stable
    sig1 = _compute_matches_signature(matches)
    sig2 = _compute_matches_signature(matches)
    assert sig1 == sig2
    assert len(sig1) == 64  # SHA256 hex string length

    # 2. First calibration run (cache miss) should perform grid search and save parameters
    assert not cache_file.exists()
    params_miss = _calibrate_elo_parameters(
        matches,
        initial_world_cup_probability_elo=1500.0,
        default_parameters=default_params,
        validation_fraction=0.2,
        min_matches=80,
        cache_path=cache_file,
    )
    assert cache_file.is_file()
    assert params_miss.validation_error is not None

    # Load cache file and verify its contents
    import json
    with open(cache_file, encoding="utf-8") as f:
        cache_data = json.load(f)
    assert cache_data["matches_signature"] == sig1
    assert cache_data["validation_fraction"] == 0.2
    assert cache_data["min_matches"] == 80
    assert cache_data["initial_world_cup_probability_elo"] == 1500.0
    assert cache_data["calibrated_parameters"]["base_k_factor"] == params_miss.base_k_factor

    # 3. Second run (cache hit) should load from cache instantly (and return identical params)
    params_hit = _calibrate_elo_parameters(
        matches,
        initial_world_cup_probability_elo=1500.0,
        default_parameters=default_params,
        validation_fraction=0.2,
        min_matches=80,
        cache_path=cache_file,
    )
    assert params_hit.base_k_factor == params_miss.base_k_factor
    assert params_hit.home_advantage_points == params_miss.home_advantage_points
    assert params_hit.goal_margin_exponent == params_miss.goal_margin_exponent
    assert params_hit.validation_error == params_miss.validation_error

    # 4. Modifying match records should invalidate the cache (signature mismatch)
    modified_matches = list(matches)
    modified_matches[0] = MatchRecord(
        match_id="m1",
        match_date="2026-06-01",
        competition="Friendly",
        season="2026",
        stage="Group",
        home_team_id="TeamA",
        away_team_id="TeamB",
        home_team_score=10,  # Modified score to change signature
        away_team_score=0,
        neutral_site=False,
    )
    sig_modified = _compute_matches_signature(modified_matches)
    assert sig_modified != sig1

    # Run calibration again with modified matches: should re-calibrate
    params_modified = _calibrate_elo_parameters(
        modified_matches,
        initial_world_cup_probability_elo=1500.0,
        default_parameters=default_params,
        validation_fraction=0.2,
        min_matches=80,
        cache_path=cache_file,
    )
    # The cache file should now be updated with the new signature
    with open(cache_file, encoding="utf-8") as f:
        new_cache_data = json.load(f)
    assert new_cache_data["matches_signature"] == sig_modified

    # 5. Bypassing cache (passing cache_path=None)
    params_no_cache = _calibrate_elo_parameters(
        matches,
        initial_world_cup_probability_elo=1500.0,
        default_parameters=default_params,
        validation_fraction=0.2,
        min_matches=80,
        cache_path=None,
    )
    assert params_no_cache.validation_error is not None
