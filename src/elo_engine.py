"""Iterative World Cup Probability Elo engine persisted in DuckDB.

The engine reads ``f_matches`` in strict chronological order, updates World Cup
Probability Elo ratings match by match, and stores the full rating history in
``f_elo_history``.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import polars as pl

try:
    from .competition_filters import current_world_cup_exclusion_sql
    from .settings import DB_PATH
except ImportError:  # pragma: no cover - supports direct script execution.
    from competition_filters import current_world_cup_exclusion_sql
    from settings import DB_PATH

LOGGER = logging.getLogger(__name__)

INITIAL_WORLD_CUP_PROBABILITY_ELO = 1500.0
BASE_K_FACTOR = 20.0
HOME_ADVANTAGE_POINTS = 100.0
GOAL_MARGIN_EXPONENT = 0.60
DEFAULT_ELO_CALIBRATION_FRACTION = 0.2
MIN_ELO_CALIBRATION_MATCHES = 80
ELO_PARAMETER_VERSION = "dynamic_elo_v1"

COMPETITION_GROUP_WEIGHTS: dict[str, float] = {
    "world_cup": 2.5,
    "world_cup_qualifier": 2.0,
    "continental": 1.8,
    "nations_league": 1.35,
    "qualifier": 1.4,
    "friendly": 0.5,
    "default": 1.0,
}
BASE_K_CANDIDATES: tuple[float, ...] = (12.0, 16.0, 20.0, 24.0, 28.0, 32.0)
HOME_ADVANTAGE_CANDIDATES: tuple[float, ...] = (0.0, 40.0, 70.0, 100.0, 130.0)
GOAL_MARGIN_EXPONENT_CANDIDATES: tuple[float, ...] = (0.0, 0.35, 0.60, 0.85)
COMPETITION_WEIGHT_FACTORS: tuple[float, ...] = (0.75, 0.9, 1.0, 1.1, 1.25)


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """Canonical match row used by the Elo loop."""

    match_id: str
    match_date: object
    competition: str | None
    season: str | None
    stage: str | None
    home_team_id: str
    away_team_id: str
    home_team_score: int | None
    away_team_score: int | None
    neutral_site: bool | None


@dataclass(frozen=True, slots=True)
class EloParameters:
    """Calibrated parameters used by the dynamic Elo loop."""

    base_k_factor: float = BASE_K_FACTOR
    home_advantage_points: float = HOME_ADVANTAGE_POINTS
    goal_margin_exponent: float = GOAL_MARGIN_EXPONENT
    competition_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPETITION_GROUP_WEIGHTS)
    )
    validation_error: float | None = None


def initialize_elo_history(db_path: Path = DB_PATH) -> None:
    """Create the World Cup Probability Elo history table if it does not exist."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS f_elo_history (
                match_id VARCHAR PRIMARY KEY,
                match_date DATE NOT NULL,
                competition VARCHAR,
                season VARCHAR,
                stage VARCHAR,
                home_team_id VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                home_rating_before DOUBLE NOT NULL,
                away_rating_before DOUBLE NOT NULL,
                home_rating_after DOUBLE NOT NULL,
                away_rating_after DOUBLE NOT NULL,
                home_expected_score DOUBLE NOT NULL,
                away_expected_score DOUBLE NOT NULL,
                home_actual_score DOUBLE NOT NULL,
                away_actual_score DOUBLE NOT NULL,
                k_factor DOUBLE NOT NULL,
                base_k_factor DOUBLE,
                competition_weight DOUBLE NOT NULL,
                home_advantage_points DOUBLE NOT NULL,
                goal_margin INTEGER,
                goal_margin_multiplier DOUBLE,
                experience_multiplier DOUBLE,
                elo_parameter_version VARCHAR,
                calibration_validation_error DOUBLE,
                neutral_site BOOLEAN,
                source_file VARCHAR,
                loaded_at TIMESTAMP,
                updated_at TIMESTAMP NOT NULL
            );
            """,
        )
        _migrate_elo_history_schema(con)


def _migrate_elo_history_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Keep existing Elo history tables compatible with the dynamic schema."""
    migration_statements = (
        "ALTER TABLE f_elo_history ADD COLUMN IF NOT EXISTS base_k_factor DOUBLE",
        "ALTER TABLE f_elo_history ADD COLUMN IF NOT EXISTS goal_margin INTEGER",
        "ALTER TABLE f_elo_history ADD COLUMN IF NOT EXISTS goal_margin_multiplier DOUBLE",
        "ALTER TABLE f_elo_history ADD COLUMN IF NOT EXISTS experience_multiplier DOUBLE",
        "ALTER TABLE f_elo_history ADD COLUMN IF NOT EXISTS elo_parameter_version VARCHAR",
        "ALTER TABLE f_elo_history ADD COLUMN IF NOT EXISTS calibration_validation_error DOUBLE",
    )
    for statement in migration_statements:
        con.execute(statement)


def build_elo_history(
    db_path: Path = DB_PATH,
    *,
    initial_world_cup_probability_elo: float = INITIAL_WORLD_CUP_PROBABILITY_ELO,
    base_k_factor: float = BASE_K_FACTOR,
    home_advantage_points: float = HOME_ADVANTAGE_POINTS,
    goal_margin_exponent: float = GOAL_MARGIN_EXPONENT,
    calibrate_parameters: bool = True,
    calibration_validation_fraction: float = DEFAULT_ELO_CALIBRATION_FRACTION,
    min_calibration_matches: int = MIN_ELO_CALIBRATION_MATCHES,
) -> None:
    """Iteratively compute World Cup Probability Elo and upsert it into DuckDB.

    Args:
        db_path: Path to the local DuckDB warehouse.
        initial_world_cup_probability_elo: Starting rating for unseen teams.
        base_k_factor: Initial/default learning rate for the Elo update.
        home_advantage_points: Initial/default home-field advantage in Elo points.
        goal_margin_exponent: Initial/default exponent for goal-margin scaling.
        calibrate_parameters: Whether to optimize Elo parameters on a temporal holdout.
        calibration_validation_fraction: Most recent historical fraction used for calibration.
        min_calibration_matches: Minimum rows required before calibration is attempted.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    initialize_elo_history(db_path)

    with duckdb.connect(str(db_path)) as con:
        _ensure_matches_exist(con)
        _delete_stale_elo_history(con)
        _delete_current_world_cup_history(con)
        matches = _load_matches(con)
        ratings: dict[str, float] = {}
        match_counts: dict[str, int] = {}
        default_parameters = EloParameters(
            base_k_factor=base_k_factor,
            home_advantage_points=home_advantage_points,
            goal_margin_exponent=goal_margin_exponent,
            competition_weights=dict(COMPETITION_GROUP_WEIGHTS),
        )
        parameters = (
            _calibrate_elo_parameters(
                matches,
                initial_world_cup_probability_elo=initial_world_cup_probability_elo,
                default_parameters=default_parameters,
                validation_fraction=calibration_validation_fraction,
                min_matches=min_calibration_matches,
            )
            if calibrate_parameters
            else default_parameters
        )
        _log_elo_parameters(parameters, calibrated=calibrate_parameters)

        history_records: list[dict] = []
        progress_interval = _elo_progress_interval(len(matches))
        LOGGER.info("Processing %d matches in strict chronological order.", len(matches))
        for match_number, match in enumerate(matches, start=1):
            home_before = ratings.get(match.home_team_id, initial_world_cup_probability_elo)
            away_before = ratings.get(match.away_team_id, initial_world_cup_probability_elo)

            competition_weight = _competition_weight(
                match.competition,
                weights=parameters.competition_weights,
            )
            neutral_site = bool(match.neutral_site) if match.neutral_site is not None else False
            home_advantage = 0.0 if neutral_site else parameters.home_advantage_points

            home_expected = _expected_score(home_before + home_advantage, away_before)
            away_expected = 1.0 - home_expected
            home_actual, away_actual = _actual_scores(
                match.home_team_score,
                match.away_team_score,
            )
            goal_margin = _goal_margin(match.home_team_score, match.away_team_score)
            goal_margin_multiplier = _goal_margin_multiplier(
                match.home_team_score,
                match.away_team_score,
                home_rating=home_before,
                away_rating=away_before,
                home_advantage=home_advantage,
                exponent=parameters.goal_margin_exponent,
            )
            experience_multiplier = _experience_multiplier(
                match_counts.get(match.home_team_id, 0),
                match_counts.get(match.away_team_id, 0),
            )
            k_factor = (
                parameters.base_k_factor
                * competition_weight
                * goal_margin_multiplier
                * experience_multiplier
            )

            home_after = home_before + k_factor * (home_actual - home_expected)
            away_after = away_before + k_factor * (away_actual - away_expected)

            ratings[match.home_team_id] = home_after
            ratings[match.away_team_id] = away_after
            match_counts[match.home_team_id] = match_counts.get(match.home_team_id, 0) + 1
            match_counts[match.away_team_id] = match_counts.get(match.away_team_id, 0) + 1

            history_records.append({
                "match_id": match.match_id,
                "match_date": match.match_date,
                "competition": match.competition,
                "season": match.season,
                "stage": match.stage,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_rating_before": home_before,
                "away_rating_before": away_before,
                "home_rating_after": home_after,
                "away_rating_after": away_after,
                "home_expected_score": home_expected,
                "away_expected_score": away_expected,
                "home_actual_score": home_actual,
                "away_actual_score": away_actual,
                "k_factor": k_factor,
                "base_k_factor": parameters.base_k_factor,
                "competition_weight": competition_weight,
                "home_advantage_points": home_advantage if not neutral_site else 0.0,
                "goal_margin": goal_margin,
                "goal_margin_multiplier": goal_margin_multiplier,
                "experience_multiplier": experience_multiplier,
                "neutral_site": neutral_site,
            })

            if _should_log_elo_progress(match_number, len(matches), progress_interval):
                LOGGER.info(
                    "Elo progress: match %d/%d processed (%.1f%% complete).",
                    match_number,
                    len(matches),
                    100 * match_number / len(matches),
                )

        if history_records:
            LOGGER.info("Performing bulk upsert of ELO history into database...")
            df_history = pl.DataFrame(history_records)
            con.register("df_history", df_history)
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
                    base_k_factor,
                    competition_weight,
                    home_advantage_points,
                    goal_margin,
                    goal_margin_multiplier,
                    experience_multiplier,
                    elo_parameter_version,
                    calibration_validation_error,
                    neutral_site,
                    source_file,
                    loaded_at,
                    updated_at
                )
                SELECT
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
                    base_k_factor,
                    competition_weight,
                    home_advantage_points,
                    goal_margin,
                    goal_margin_multiplier,
                    experience_multiplier,
                    ? AS elo_parameter_version,
                    ? AS calibration_validation_error,
                    neutral_site,
                    NULL AS source_file,
                    NULL AS loaded_at,
                    current_timestamp AS updated_at
                FROM df_history
                ON CONFLICT (match_id) DO UPDATE SET
                    match_date = excluded.match_date,
                    competition = excluded.competition,
                    season = excluded.season,
                    stage = excluded.stage,
                    home_team_id = excluded.home_team_id,
                    away_team_id = excluded.away_team_id,
                    home_rating_before = excluded.home_rating_before,
                    away_rating_before = excluded.away_rating_before,
                    home_rating_after = excluded.home_rating_after,
                    away_rating_after = excluded.away_rating_after,
                    home_expected_score = excluded.home_expected_score,
                    away_expected_score = excluded.away_expected_score,
                    home_actual_score = excluded.home_actual_score,
                    away_actual_score = excluded.away_actual_score,
                    k_factor = excluded.k_factor,
                    base_k_factor = excluded.base_k_factor,
                    competition_weight = excluded.competition_weight,
                    home_advantage_points = excluded.home_advantage_points,
                    goal_margin = excluded.goal_margin,
                    goal_margin_multiplier = excluded.goal_margin_multiplier,
                    experience_multiplier = excluded.experience_multiplier,
                    elo_parameter_version = excluded.elo_parameter_version,
                    calibration_validation_error = excluded.calibration_validation_error,
                    neutral_site = excluded.neutral_site,
                    updated_at = now();
                """,
                [ELO_PARAMETER_VERSION, parameters.validation_error],
            )

    LOGGER.info("World Cup Probability Elo history updated successfully.")


def _elo_progress_interval(match_count: int) -> int:
    return max(1, min(1_000, match_count // 100 or 1))


def _should_log_elo_progress(
    match_number: int,
    match_count: int,
    progress_interval: int,
) -> bool:
    return (
        match_count > 0
        and (
            match_number == 1
            or match_number == match_count
            or match_number % progress_interval == 0
        )
    )


def _calibrate_elo_parameters(
    matches: list[MatchRecord],
    *,
    initial_world_cup_probability_elo: float,
    default_parameters: EloParameters,
    validation_fraction: float,
    min_matches: int,
) -> EloParameters:
    """Tune Elo parameters by minimizing temporal holdout squared error."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("calibration_validation_fraction must be greater than 0 and less than 1.")
    if min_matches <= 0:
        raise ValueError("min_calibration_matches must be greater than zero.")
    if len(matches) < min_matches:
        LOGGER.info(
            "Skipping Elo calibration: %d matches available, minimum is %d.",
            len(matches),
            min_matches,
        )
        return default_parameters

    best_parameters = _grid_search_global_elo_parameters(
        matches,
        initial_world_cup_probability_elo=initial_world_cup_probability_elo,
        default_parameters=default_parameters,
        validation_fraction=validation_fraction,
    )

    for _ in range(2):
        best_parameters = _coordinate_search_competition_weights(
            matches,
            initial_world_cup_probability_elo=initial_world_cup_probability_elo,
            parameters=best_parameters,
            validation_fraction=validation_fraction,
        )

    return best_parameters


def _grid_search_global_elo_parameters(
    matches: list[MatchRecord],
    *,
    initial_world_cup_probability_elo: float,
    default_parameters: EloParameters,
    validation_fraction: float,
) -> EloParameters:
    best_parameters = default_parameters
    best_error = _elo_validation_error(
        matches,
        initial_world_cup_probability_elo=initial_world_cup_probability_elo,
        parameters=default_parameters,
        validation_fraction=validation_fraction,
    )
    for base_k_factor in BASE_K_CANDIDATES:
        for home_advantage_points in HOME_ADVANTAGE_CANDIDATES:
            for goal_margin_exponent in GOAL_MARGIN_EXPONENT_CANDIDATES:
                candidate = EloParameters(
                    base_k_factor=base_k_factor,
                    home_advantage_points=home_advantage_points,
                    goal_margin_exponent=goal_margin_exponent,
                    competition_weights=dict(default_parameters.competition_weights),
                )
                error = _elo_validation_error(
                    matches,
                    initial_world_cup_probability_elo=initial_world_cup_probability_elo,
                    parameters=candidate,
                    validation_fraction=validation_fraction,
                )
                if error < best_error:
                    best_error = error
                    best_parameters = candidate

    return EloParameters(
        base_k_factor=best_parameters.base_k_factor,
        home_advantage_points=best_parameters.home_advantage_points,
        goal_margin_exponent=best_parameters.goal_margin_exponent,
        competition_weights=dict(best_parameters.competition_weights),
        validation_error=best_error,
    )


def _coordinate_search_competition_weights(
    matches: list[MatchRecord],
    *,
    initial_world_cup_probability_elo: float,
    parameters: EloParameters,
    validation_fraction: float,
) -> EloParameters:
    best_parameters = parameters
    best_error = parameters.validation_error
    if best_error is None:
        best_error = _elo_validation_error(
            matches,
            initial_world_cup_probability_elo=initial_world_cup_probability_elo,
            parameters=parameters,
            validation_fraction=validation_fraction,
        )

    for group, default_weight in COMPETITION_GROUP_WEIGHTS.items():
        group_best_parameters = best_parameters
        group_best_error = best_error
        for factor in COMPETITION_WEIGHT_FACTORS:
            weights = dict(best_parameters.competition_weights)
            weights[group] = max(0.2, min(4.0, default_weight * factor))
            candidate = EloParameters(
                base_k_factor=best_parameters.base_k_factor,
                home_advantage_points=best_parameters.home_advantage_points,
                goal_margin_exponent=best_parameters.goal_margin_exponent,
                competition_weights=weights,
            )
            error = _elo_validation_error(
                matches,
                initial_world_cup_probability_elo=initial_world_cup_probability_elo,
                parameters=candidate,
                validation_fraction=validation_fraction,
            )
            if error < group_best_error:
                group_best_error = error
                group_best_parameters = candidate

        best_parameters = EloParameters(
            base_k_factor=group_best_parameters.base_k_factor,
            home_advantage_points=group_best_parameters.home_advantage_points,
            goal_margin_exponent=group_best_parameters.goal_margin_exponent,
            competition_weights=dict(group_best_parameters.competition_weights),
            validation_error=group_best_error,
        )
        best_error = group_best_error

    return best_parameters


def _elo_validation_error(
    matches: list[MatchRecord],
    *,
    initial_world_cup_probability_elo: float,
    parameters: EloParameters,
    validation_fraction: float,
) -> float:
    validation_start = int(len(matches) * (1.0 - validation_fraction))
    validation_start = min(max(validation_start, 1), len(matches) - 1)
    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}
    squared_errors: list[float] = []

    for index, match in enumerate(matches):
        home_before = ratings.get(match.home_team_id, initial_world_cup_probability_elo)
        away_before = ratings.get(match.away_team_id, initial_world_cup_probability_elo)
        neutral_site = bool(match.neutral_site) if match.neutral_site is not None else False
        home_advantage = 0.0 if neutral_site else parameters.home_advantage_points
        home_expected = _expected_score(home_before + home_advantage, away_before)
        home_actual, away_actual = _actual_scores(match.home_team_score, match.away_team_score)

        if index >= validation_start:
            squared_errors.append((home_actual - home_expected) ** 2)

        k_factor = (
            parameters.base_k_factor
            * _competition_weight(match.competition, weights=parameters.competition_weights)
            * _goal_margin_multiplier(
                match.home_team_score,
                match.away_team_score,
                home_rating=home_before,
                away_rating=away_before,
                home_advantage=home_advantage,
                exponent=parameters.goal_margin_exponent,
            )
            * _experience_multiplier(
                match_counts.get(match.home_team_id, 0),
                match_counts.get(match.away_team_id, 0),
            )
        )
        away_expected = 1.0 - home_expected
        ratings[match.home_team_id] = home_before + k_factor * (home_actual - home_expected)
        ratings[match.away_team_id] = away_before + k_factor * (away_actual - away_expected)
        match_counts[match.home_team_id] = match_counts.get(match.home_team_id, 0) + 1
        match_counts[match.away_team_id] = match_counts.get(match.away_team_id, 0) + 1

    if not squared_errors:
        return math.inf
    return sum(squared_errors) / len(squared_errors)


def _ensure_matches_exist(con: duckdb.DuckDBPyConnection) -> None:
    count = con.execute("SELECT COUNT(*) FROM f_matches").fetchone()[0]
    if int(count) == 0:
        LOGGER.warning("f_matches is empty. World Cup Probability Elo will not be generated.")


def _delete_current_world_cup_history(con: duckdb.DuckDBPyConnection) -> None:
    current_world_cup_exclusion = current_world_cup_exclusion_sql(
        date_expr="match_date",
        competition_expr="competition",
    )
    con.execute(
        f"""
        DELETE FROM f_elo_history
        WHERE match_id IN (
            SELECT match_id
            FROM f_matches
            WHERE NOT {current_world_cup_exclusion}
        )
        """,
    )


def _delete_stale_elo_history(con: duckdb.DuckDBPyConnection) -> None:
    """Remove Elo rows for match IDs no longer present in the match fact table."""
    con.execute(
        """
        DELETE FROM f_elo_history
        WHERE match_id NOT IN (
            SELECT match_id
            FROM f_matches
        )
        """,
    )


def _load_matches(con: duckdb.DuckDBPyConnection) -> list[MatchRecord]:
    null_dates = con.execute(
        "SELECT COUNT(*) FROM f_matches WHERE match_date IS NULL",
    ).fetchone()[0]
    if int(null_dates) > 0:
        raise ValueError("f_matches contains rows with NULL match_date.")

    current_world_cup_exclusion = current_world_cup_exclusion_sql(
        date_expr="match_date",
        competition_expr="competition",
    )
    query = f"""
        SELECT
            match_id,
            match_date,
            competition,
            season,
            stage,
            home_team_id,
            away_team_id,
            home_team_score,
            away_team_score,
            neutral_site
        FROM f_matches
        WHERE
            home_team_score IS NOT NULL
            AND away_team_score IS NOT NULL
            AND {current_world_cup_exclusion}
        ORDER BY match_date ASC, match_id ASC, home_team_id ASC, away_team_id ASC
    """
    rows = con.execute(query).fetchall()
    return [
        MatchRecord(
            match_id=str(row[0]),
            match_date=row[1],
            competition=row[2],
            season=row[3],
            stage=row[4],
            home_team_id=str(row[5]),
            away_team_id=str(row[6]),
            home_team_score=_optional_int(row[7]),
            away_team_score=_optional_int(row[8]),
            neutral_site=_optional_bool(row[9]),
        )
        for row in rows
    ]


def _expected_score(rating_for: float, rating_against: float) -> float:
    """Calculate the expected score for a rating pair."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_against - rating_for) / 400.0))


def _actual_scores(home_score: int | None, away_score: int | None) -> tuple[float, float]:
    """Convert the final score into World Cup Probability Elo actual scores."""
    if home_score is None or away_score is None:
        return 0.5, 0.5

    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def _goal_margin(home_score: int | None, away_score: int | None) -> int:
    """Return the absolute goal margin for a completed match."""
    if home_score is None or away_score is None:
        return 0
    return abs(int(home_score) - int(away_score))


def _goal_margin_multiplier(
    home_score: int | None,
    away_score: int | None,
    *,
    home_rating: float,
    away_rating: float,
    home_advantage: float,
    exponent: float,
) -> float:
    """Scale updates for decisive wins while limiting favorite blowout inflation."""
    margin = _goal_margin(home_score, away_score)
    if margin <= 1 or exponent <= 0.0 or home_score == away_score:
        return 1.0

    if home_score is not None and away_score is not None and home_score > away_score:
        winner_rating_diff = home_rating + home_advantage - away_rating
    else:
        winner_rating_diff = away_rating - (home_rating + home_advantage)

    margin_boost = math.pow(math.log1p(float(margin)), exponent)
    expectation_correction = 2.2 / max(1.0, (winner_rating_diff * 0.001) + 2.2)
    return max(1.0, min(3.0, margin_boost * expectation_correction))


def _experience_multiplier(home_match_count: int, away_match_count: int) -> float:
    """Raise K for teams with short observed histories, then decay toward 1."""
    return (
        _single_team_experience_multiplier(home_match_count)
        + _single_team_experience_multiplier(away_match_count)
    ) / 2.0


def _single_team_experience_multiplier(match_count: int) -> float:
    if match_count < 10:
        return 1.35
    if match_count < 25:
        return 1.15
    return 1.0


def _competition_weight(
    competition: str | None,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return the competition multiplier for the given match label."""
    active_weights = weights or COMPETITION_GROUP_WEIGHTS
    return float(active_weights.get(_competition_group(competition), active_weights["default"]))


def _competition_group(competition: str | None) -> str:
    """Map noisy competition labels to tunable Elo competition buckets.

    Args:
        competition: The competition name or label to group.

    Returns:
        The matched competition category string as defined in Elo weights.
    """
    if competition is None:
        return "default"
    normalized = competition.casefold()
    if (
        "world cup qualifier" in normalized
        or "world cup qualifying" in normalized
        or "world cup qualification" in normalized
        or "copa do mundo qualific" in normalized
    ):
        return "world_cup_qualifier"
    if "world cup" in normalized or "copa do mundo" in normalized:
        return "world_cup"
    if "friendly" in normalized or "amistoso" in normalized:
        return "friendly"
    if "nations league" in normalized:
        return "nations_league"
    if "continental championship" in normalized or "continental cup" in normalized:
        return "continental"
    if "qualifier" in normalized or "qualifying" in normalized or "qualificat" in normalized:
        return "qualifier"
    return "default"


def _log_elo_parameters(parameters: EloParameters, *, calibrated: bool) -> None:
    status = "calibrated" if calibrated and parameters.validation_error is not None else "default"
    LOGGER.info(
        (
            "Using %s Elo parameters: base_k=%.2f | home_advantage=%.1f | "
            "goal_margin_exponent=%.2f | validation_error=%s | competition_weights=%s"
        ),
        status,
        parameters.base_k_factor,
        parameters.home_advantage_points,
        parameters.goal_margin_exponent,
        (
            f"{parameters.validation_error:.6f}"
            if parameters.validation_error is not None
            else "n/a"
        ),
        dict(parameters.competition_weights),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the dynamic Elo engine."""
    parser = argparse.ArgumentParser(description="Build calibrated World Cup Probability Elo.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="DuckDB warehouse path.")
    parser.add_argument(
        "--base-k-factor",
        type=float,
        default=BASE_K_FACTOR,
        help="Default base K used directly or as the calibration reference.",
    )
    parser.add_argument(
        "--home-advantage-points",
        type=float,
        default=HOME_ADVANTAGE_POINTS,
        help="Default home-field advantage used directly or as the calibration reference.",
    )
    parser.add_argument(
        "--goal-margin-exponent",
        type=float,
        default=GOAL_MARGIN_EXPONENT,
        help="Default exponent used for goal-margin Elo scaling.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable temporal calibration and use the provided/default Elo parameters.",
    )
    parser.add_argument(
        "--calibration-validation-fraction",
        type=float,
        default=DEFAULT_ELO_CALIBRATION_FRACTION,
        help="Most recent historical fraction used to calibrate Elo parameters.",
    )
    parser.add_argument(
        "--min-calibration-matches",
        type=int,
        default=MIN_ELO_CALIBRATION_MATCHES,
        help="Minimum historical rows required before Elo calibration runs.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    build_elo_history(
        db_path=args.db_path,
        base_k_factor=args.base_k_factor,
        home_advantage_points=args.home_advantage_points,
        goal_margin_exponent=args.goal_margin_exponent,
        calibrate_parameters=not args.no_calibration,
        calibration_validation_fraction=args.calibration_validation_fraction,
        min_calibration_matches=args.min_calibration_matches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
