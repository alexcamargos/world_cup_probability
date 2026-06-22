from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.model import FEATURE_COLUMNS
from src.outcome_model import (
    AWAY_WIN_CLASS,
    DRAW_CLASS,
    HOME_WIN_CLASS,
    _binomial_one_sided_p_value,
    evaluate_outcome_model,
    prepare_outcome_matrices,
)


def _outcome_frame() -> pl.DataFrame:
    outcomes = [
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 1),
        (2, 2),
        (1, 3),
    ]
    rows = []
    for index, (home_goals, away_goals) in enumerate(outcomes):
        row = {
            "match_id": f"m{index}",
            "match_date": date(2020 + index, 1, 1),
            "competition": "Friendly",
            "is_current_world_cup": False,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "target": float(home_goals),
        }
        row.update(dict.fromkeys(FEATURE_COLUMNS, float(index)))
        rows.append(row)

    holdout_row = {
        "match_id": "wc2026",
        "match_date": date(2026, 6, 20),
        "competition": "FIFA World Cup",
        "is_current_world_cup": True,
        "home_goals": 9,
        "away_goals": 0,
        "target": 9.0,
    }
    holdout_row.update(dict.fromkeys(FEATURE_COLUMNS, 99.0))
    rows.append(holdout_row)
    return pl.DataFrame(rows)


def test_prepare_outcome_matrices_labels_results_and_excludes_current_world_cup() -> None:
    X_train, X_valid, y_train, y_valid, feature_names = prepare_outcome_matrices(
        _outcome_frame(),
        validation_fraction=0.5,
    )

    assert feature_names == list(FEATURE_COLUMNS)
    assert X_train.shape == (3, len(FEATURE_COLUMNS))
    assert X_valid.shape == (3, len(FEATURE_COLUMNS))
    assert y_train.tolist() == [HOME_WIN_CLASS, DRAW_CLASS, AWAY_WIN_CLASS]
    assert y_valid.tolist() == [HOME_WIN_CLASS, DRAW_CLASS, AWAY_WIN_CLASS]


def test_evaluate_outcome_model_reports_lift_over_uniform_random() -> None:
    y_valid = np.array([HOME_WIN_CLASS, DRAW_CLASS, AWAY_WIN_CLASS])

    class PerfectOutcomeModel:
        def predict(self, X_valid: np.ndarray) -> np.ndarray:
            return y_valid

        def predict_proba(self, X_valid: np.ndarray) -> np.ndarray:
            return np.eye(3)

    metrics = evaluate_outcome_model(
        PerfectOutcomeModel(),  # type: ignore[arg-type]
        np.zeros((3, len(FEATURE_COLUMNS))),
        y_valid,
    )

    assert metrics["rows"] == 3.0
    assert metrics["hits"] == 3.0
    assert metrics["accuracy"] == 1.0
    assert metrics["uniform_random_accuracy"] == 1 / 3
    assert metrics["accuracy_lift_vs_uniform_random"] == pytest.approx(2 / 3)
    assert metrics["binomial_p_value_vs_uniform_random"] == pytest.approx(1 / 27)


def test_binomial_one_sided_p_value_matches_simple_tail() -> None:
    # P[X >= 2], X ~ Binomial(3, 1/3) = C(3,2)(1/3)^2(2/3) + (1/3)^3.
    assert _binomial_one_sided_p_value(successes=2, trials=3, p=1 / 3) == pytest.approx(7 / 27)
