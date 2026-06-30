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
    apply_temperature_scaling,
    build_parser,
    calibrate_temperature,
    evaluate_outcome_model,
    outcome_sample_weights,
    prepare_outcome_matrices,
    prepare_temporal_outcome_splits,
    tune_outcome_hyperparameters,
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


def test_prepare_temporal_outcome_splits_adds_calibration_window() -> None:
    split = prepare_temporal_outcome_splits(
        _outcome_frame(),
        validation_fraction=1 / 3,
        calibration_fraction=1 / 6,
    )

    assert split.X_train.shape == (3, len(FEATURE_COLUMNS))
    assert split.X_calibration.shape == (1, len(FEATURE_COLUMNS))
    assert split.X_valid.shape == (2, len(FEATURE_COLUMNS))
    assert split.y_train.tolist() == [HOME_WIN_CLASS, DRAW_CLASS, AWAY_WIN_CLASS]
    assert split.y_calibration.tolist() == [HOME_WIN_CLASS]
    assert split.y_valid.tolist() == [DRAW_CLASS, AWAY_WIN_CLASS]


def test_outcome_sample_weights_favor_recent_important_matches() -> None:
    frame = pl.DataFrame(
        [
            {"match_date": date(2020, 1, 1), "competition": "Friendly"},
            {"match_date": date(2024, 1, 1), "competition": "FIFA World Cup"},
        ]
    )

    weights = outcome_sample_weights(frame, recency_half_life_days=365.25)

    assert weights.mean() == pytest.approx(1.0)
    assert weights[1] > weights[0]


def test_tune_outcome_hyperparameters_uses_optuna_trial_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeOutcomeModel:
        def predict_proba(self, X_valid: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.8, 0.1, 0.1]]), (X_valid.shape[0], 1))

    def fake_train_outcome_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
        *,
        params: dict[str, object] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> FakeOutcomeModel:
        calls.append(
            {
                "params": params,
                "sample_weight_mean": (
                    None if sample_weight is None else float(sample_weight.mean())
                ),
                "train_rows": X_train.shape[0],
                "valid_rows": 0 if X_valid is None else X_valid.shape[0],
                "valid_labels": [] if y_valid is None else y_valid.tolist(),
            }
        )
        return FakeOutcomeModel()

    monkeypatch.setattr("src.outcome_model.train_outcome_model", fake_train_outcome_model)

    best_params = tune_outcome_hyperparameters(
        _outcome_frame(),
        n_trials=1,
        validation_fraction=1 / 3,
        calibration_fraction=1 / 6,
    )

    assert set(best_params) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
    }
    assert len(calls) == 1
    assert calls[0]["params"] == best_params
    assert calls[0]["sample_weight_mean"] == pytest.approx(1.0)
    assert calls[0]["train_rows"] == 3
    assert calls[0]["valid_rows"] == 1
    assert calls[0]["valid_labels"] == [HOME_WIN_CLASS]


def test_outcome_cli_parser_accepts_optuna_tuning_flags() -> None:
    args = build_parser().parse_args(["--tune", "--trials", "7", "--timeout-seconds", "30"])

    assert args.tune is True
    assert args.trials == 7
    assert args.timeout_seconds == 30


def test_temperature_scaling_can_sharpen_or_smooth_probabilities() -> None:
    probabilities = np.array([[0.8, 0.1, 0.1]])

    sharpened = apply_temperature_scaling(probabilities, temperature=0.5)
    smoothed = apply_temperature_scaling(probabilities, temperature=2.0)

    assert sharpened[0, 0] > probabilities[0, 0]
    assert smoothed[0, 0] < probabilities[0, 0]


def test_calibrate_temperature_returns_bounded_temperature() -> None:
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    labels = np.array([HOME_WIN_CLASS, DRAW_CLASS, AWAY_WIN_CLASS])

    temperature = calibrate_temperature(probabilities, labels)

    assert 0.25 <= temperature <= 4.0


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
