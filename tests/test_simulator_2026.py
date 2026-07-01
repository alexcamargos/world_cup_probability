from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np

from src.simulator import (
    DIXON_COLES_OUTCOME_HYBRID_SCORE_ENGINE,
    DIXON_COLES_POISSON_SCORE_ENGINE,
    POISSON_SCORE_ENGINE,
    GroupStanding,
    OutcomeModelContext,
    TeamLambda,
    _apply_group_result,
    _dixon_coles_tau,
    calibrate_dixon_coles_rho,
    load_dixon_coles_rho,
    save_dixon_coles_calibration,
    simulate_match,
    simulate_world_cup,
)
from src.world_cup_2026_schedule import TEAM_COUNTRIES, TEAM_NAMES, world_cup_2026_fixtures


def test_world_cup_2026_schedule_contains_official_tournament_shape() -> None:
    fixtures = world_cup_2026_fixtures()

    assert len(fixtures) == 104
    assert sum(fixture.round_name == "group_stage" for fixture in fixtures) == 72
    assert sum(fixture.round_name == "round_of_32" for fixture in fixtures) == 16
    assert fixtures[0].match_number == 1
    assert fixtures[0].home_slot == "MEX"
    assert fixtures[0].away_slot == "RSA"
    assert fixtures[0].played_home_goals == 2
    assert fixtures[0].played_away_goals == 0
    assert fixtures[-1].round_name == "final"


def test_simulate_world_cup_uses_real_schedule_without_preserving_real_scores(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="src.simulator")

    db_path = tmp_path / "world_cup.duckdb"
    teams = [
        TeamLambda(
            team_id=code,
            team_name=name,
            lambda_goals=0.0,
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
        opening_match = con.execute(
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
    assert opening_match == (0, 0, "Mexico City Stadium", True)
    assert rest_rows > 0
    assert "Simulation progress: tournament 1/1 completed" in caplog.text


def test_group_result_accumulates_fair_play_penalty_from_team_rate() -> None:
    home = TeamLambda("A", "A", 0.0, fair_play_penalty_rate=0.0)
    away = TeamLambda("B", "B", 0.0, fair_play_penalty_rate=10.0)
    rng = np.random.default_rng(7)
    fixture = world_cup_2026_fixtures()[0]
    result = simulate_match(home, away, require_winner=False, rng=rng)
    standings = {
        "A": {
            "A": GroupStanding(team=home, group_letter="A"),
            "B": GroupStanding(team=away, group_letter="A"),
        }
    }

    _apply_group_result(standings, fixture, result, rng)

    assert standings["A"]["A"].fair_play_points == 0
    assert standings["A"]["B"].fair_play_points > 0


def test_dixon_coles_tau_adjusts_low_score_dependence() -> None:
    home_lambda = 1.4
    away_lambda = 1.1
    rho = -0.10

    assert _dixon_coles_tau(0, 0, home_lambda, away_lambda, rho) > 1.0
    assert _dixon_coles_tau(1, 1, home_lambda, away_lambda, rho) > 1.0
    assert _dixon_coles_tau(0, 1, home_lambda, away_lambda, rho) < 1.0
    assert _dixon_coles_tau(1, 0, home_lambda, away_lambda, rho) < 1.0
    assert _dixon_coles_tau(2, 1, home_lambda, away_lambda, rho) == 1.0


def test_calibrate_dixon_coles_rho_selects_lowest_validation_log_loss() -> None:
    result = calibrate_dixon_coles_rho(
        home_goals=np.asarray([0, 1, 1, 1]),
        away_goals=np.asarray([0, 1, 1, 1]),
        home_lambdas=np.asarray([1.2, 1.2, 1.2, 1.2]),
        away_lambdas=np.asarray([1.1, 1.1, 1.1, 1.1]),
        candidate_rhos=(-0.10, 0.0, 0.10),
        default_rho=0.0,
    )

    assert result.rho == -0.10
    assert result.validation_rows == 4
    assert result.mean_negative_log_likelihood < result.default_mean_negative_log_likelihood


def test_dixon_coles_calibration_artifact_overrides_default_rho(tmp_path: Path) -> None:
    calibration_path = tmp_path / "dixon_coles_calibration.json"

    save_dixon_coles_calibration(
        calibration_path=calibration_path,
        result=calibrate_dixon_coles_rho(
            home_goals=np.asarray([0, 1, 1, 1]),
            away_goals=np.asarray([0, 1, 1, 1]),
            home_lambdas=np.asarray([1.2, 1.2, 1.2, 1.2]),
            away_lambdas=np.asarray([1.1, 1.1, 1.1, 1.1]),
            candidate_rhos=(-0.10, 0.0, 0.10),
        ),
    )

    assert load_dixon_coles_rho(calibration_path=calibration_path) == -0.10
    assert load_dixon_coles_rho(calibration_path=tmp_path / "missing.json") == -0.10


def test_simulate_match_can_use_legacy_independent_poisson_engine() -> None:
    default_result = simulate_match(
        TeamLambda("A", "A", 0.0),
        TeamLambda("B", "B", 0.0),
        require_winner=False,
        rng=np.random.default_rng(7),
    )
    result = simulate_match(
        TeamLambda("A", "A", 0.0),
        TeamLambda("B", "B", 0.0),
        require_winner=False,
        rng=np.random.default_rng(7),
        dixon_coles_rho=0.0,
    )

    assert default_result.score_engine == DIXON_COLES_POISSON_SCORE_ENGINE
    assert result.score_engine == POISSON_SCORE_ENGINE


def test_simulate_match_conditions_score_on_sampled_outcome() -> None:
    result = simulate_match(
        TeamLambda("A", "A", 0.0),
        TeamLambda("B", "B", 0.0),
        require_winner=False,
        rng=np.random.default_rng(7),
        outcome_probabilities=(1.0, 0.0, 0.0),
    )

    assert result.score_engine == DIXON_COLES_OUTCOME_HYBRID_SCORE_ENGINE
    assert result.sampled_outcome == "home"
    assert result.home_goals > result.away_goals
    assert result.outcome_home_win_pct == 100.0


def test_simulate_world_cup_persists_outcome_hybrid_engine(tmp_path: Path) -> None:
    class AlwaysHomeOutcomeModel:
        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[1.0, 0.0, 0.0]]), (matrix.shape[0], 1))

    db_path = tmp_path / "world_cup.duckdb"
    teams = [
        TeamLambda(
            team_id=code,
            team_name=name,
            lambda_goals=0.0,
            country_code=TEAM_COUNTRIES.get(code, code),
        )
        for code, name in TEAM_NAMES.items()
    ]
    feature_defaults = {
        "world_cup_probability_elo": 1500.0,
        "world_football_elo_ratings": 1500.0,
        "fifa_world_ranking_points": 1500.0,
        "fifa_world_ranking_rank": 0.0,
        "prior_world_cup_appearances": 0.0,
        "prior_world_cup_points_per_match": 0.0,
        "prior_world_cup_goal_diff_per_match": 0.0,
        "prior_world_cup_yellow_cards_per_match": 0.0,
        "prior_world_cup_sending_offs_per_match": 0.0,
        "prior_world_cup_fair_play_penalty_per_match": 0.0,
        "market_value": 0.0,
        "avg_overall": 0.0,
        "avg_pace": 0.0,
        "avg_stamina": 0.0,
        "squad_depth_proxy": 0.0,
        "recent_form": 0.0,
    }
    context = OutcomeModelContext(
        model=AlwaysHomeOutcomeModel(),  # type: ignore[arg-type]
        team_features={code: dict(feature_defaults) for code in TEAM_NAMES},
    )

    simulate_world_cup(
        teams,
        iterations=1,
        batch_size=128,
        db_path=db_path,
        seed=7,
        outcome_model_context=context,
    )

    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute(
            """
            SELECT score_engine, sampled_outcome, outcome_home_win_pct, home_goals, away_goals
            FROM simulated_results
            WHERE match_number = 1
            """
        ).fetchone()

    assert row[0] == DIXON_COLES_OUTCOME_HYBRID_SCORE_ENGINE
    assert row[1] == "home"
    assert row[2] == 100.0
    assert row[3] > row[4]
