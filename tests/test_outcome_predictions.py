from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from src.outcome_predictions import (
    OutcomePredictionRow,
    _fixture_feature_vector,
    write_outcome_predictions,
)


def test_fixture_feature_vector_uses_training_feature_order_and_rank_direction() -> None:
    home_features = {
        "world_cup_probability_elo": 1600.0,
        "world_football_elo_ratings": 1620.0,
        "fifa_world_ranking_points": 1700.0,
        "fifa_world_ranking_rank": 2.0,
        "prior_world_cup_appearances": 20.0,
        "prior_world_cup_points_per_match": 2.0,
        "prior_world_cup_goal_diff_per_match": 1.0,
        "prior_world_cup_yellow_cards_per_match": 1.0,
        "prior_world_cup_sending_offs_per_match": 0.1,
        "prior_world_cup_fair_play_penalty_per_match": 1.5,
        "market_value": 900.0,
        "avg_overall": 85.0,
        "avg_pace": 82.0,
        "avg_stamina": 80.0,
        "squad_depth_proxy": 22.0,
        "recent_form": 0.5,
    }
    away_features = {
        "world_cup_probability_elo": 1500.0,
        "world_football_elo_ratings": 1510.0,
        "fifa_world_ranking_points": 1600.0,
        "fifa_world_ranking_rank": 12.0,
        "prior_world_cup_appearances": 10.0,
        "prior_world_cup_points_per_match": 1.5,
        "prior_world_cup_goal_diff_per_match": 0.2,
        "prior_world_cup_yellow_cards_per_match": 1.2,
        "prior_world_cup_sending_offs_per_match": 0.0,
        "prior_world_cup_fair_play_penalty_per_match": 1.2,
        "market_value": 400.0,
        "avg_overall": 80.0,
        "avg_pace": 79.0,
        "avg_stamina": 77.0,
        "squad_depth_proxy": 18.0,
        "recent_form": -0.25,
    }

    vector = _fixture_feature_vector(home_features, away_features)

    assert vector[0] == 100.0
    assert vector[3] == 10.0
    assert vector[10] == 500.0
    assert vector[-1] == pytest.approx(0.75)


def test_write_outcome_predictions_replaces_existing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "world_cup.duckdb"
    created_at = datetime(2026, 6, 22, tzinfo=UTC)
    row = OutcomePredictionRow(
        match_number=1,
        round_name="group_stage",
        group_name="Group A",
        match_date=created_at,
        home_team_id="BRA",
        home_team_name="Brazil",
        away_team_id="ARG",
        away_team_name="Argentina",
        home_win_pct=40.0,
        draw_pct=30.0,
        away_win_pct=30.0,
        predicted_outcome="home_win",
        calibration_temperature=2.0,
        model_path="models/xgb_outcome_model.json",
        created_at=created_at,
    )

    write_outcome_predictions(db_path=db_path, rows=[row])
    write_outcome_predictions(db_path=db_path, rows=[row])

    with duckdb.connect(str(db_path), read_only=True) as con:
        stored = con.execute(
            """
            SELECT COUNT(*), max(home_win_pct), max(predicted_outcome)
            FROM outcome_predictions
            """
        ).fetchone()

    assert stored == (1, 40.0, "home_win")
