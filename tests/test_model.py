from __future__ import annotations

from datetime import date

import polars as pl

from src.feature_pipeline import _build_team_level_features
from src.model import FEATURE_COLUMNS, prepare_matrices


def _model_frame() -> pl.DataFrame:
    rows = []
    for index in range(5):
        row = {
            "match_id": f"m{index}",
            "match_date": date(2020 + index, 1, 1),
            "competition": "Friendly",
            "is_current_world_cup": False,
            "target": float(index),
        }
        row.update(dict.fromkeys(FEATURE_COLUMNS, float(index)))
        rows.append(row)

    holdout_row = {
        "match_id": "wc2026",
        "match_date": date(2026, 6, 20),
        "competition": "FIFA World Cup",
        "is_current_world_cup": True,
        "target": 99.0,
    }
    holdout_row.update(dict.fromkeys(FEATURE_COLUMNS, 99.0))
    rows.append(holdout_row)
    return pl.DataFrame(rows)


def test_prepare_matrices_uses_temporal_split_and_excludes_current_world_cup() -> None:
    X_train, X_valid, y_train, y_valid, feature_names = prepare_matrices(
        _model_frame(),
        validation_fraction=0.4,
    )

    assert feature_names == list(FEATURE_COLUMNS)
    assert X_train.shape == (3, len(FEATURE_COLUMNS))
    assert X_valid.shape == (2, len(FEATURE_COLUMNS))
    assert y_train.tolist() == [0.0, 1.0, 2.0]
    assert y_valid.tolist() == [3.0, 4.0]


def test_current_world_cup_rows_do_not_feed_recent_form_history() -> None:
    frame = pl.DataFrame(
        [
            _match_row("m1", date(2025, 1, 1), "Friendly", False, 2, 0),
            _match_row("m2", date(2026, 6, 20), "FIFA World Cup", True, 7, 0),
            _match_row("m3", date(2026, 6, 24), "FIFA World Cup", True, 0, 0),
        ]
    )

    features = _build_team_level_features(frame, include_metadata=True)
    third_match = features.filter(pl.col("match_id") == "m3").row(0, named=True)

    assert third_match["recent_form_diff"] == 4.0


def _match_row(
    match_id: str,
    match_date: date,
    competition: str,
    is_current_world_cup: bool,
    home_score: int,
    away_score: int,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": match_date,
        "competition": competition,
        "season": str(match_date.year),
        "stage": "group",
        "home_team_id": "Brazil",
        "away_team_id": "Argentina",
        "home_team_score": home_score,
        "away_team_score": away_score,
        "neutral_site": True,
        "is_current_world_cup": is_current_world_cup,
        "home_world_cup_probability_elo_before": 1600.0,
        "away_world_cup_probability_elo_before": 1500.0,
        "home_world_football_elo_ratings": 1600.0,
        "away_world_football_elo_ratings": 1500.0,
        "home_fifa_world_ranking_points": 1600.0,
        "away_fifa_world_ranking_points": 1500.0,
        "home_fifa_world_ranking_rank": 1.0,
        "away_fifa_world_ranking_rank": 2.0,
        "home_avg_overall": 85.0,
        "away_avg_overall": 84.0,
        "home_avg_pace": 82.0,
        "away_avg_pace": 81.0,
        "home_avg_stamina": 80.0,
        "away_avg_stamina": 79.0,
        "home_squad_depth_proxy": 11.0,
        "away_squad_depth_proxy": 10.0,
        "home_market_value_eur": 100.0,
        "away_market_value_eur": 90.0,
    }
