"""Iterative World Cup Probability Elo engine persisted in DuckDB.

The engine reads ``f_matches`` in strict chronological order, updates World Cup
Probability Elo ratings match by match, and stores the full rating history in
``f_elo_history``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb

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

COMPETITION_WEIGHT_MULTIPLIERS: dict[str, float] = {
    "World Cup": 2.5,
    "World Cup Qualifier": 2.0,
    "World Cup Qualifying": 2.0,
    "Copa do Mundo": 2.5,
    "Copa do Mundo Qualificatória": 2.0,
    "Continental Championship": 1.8,
    "Continental Cup": 1.8,
    "Nations League": 1.35,
    "Qualifier": 1.4,
    "Friendly": 0.5,
    "International Friendly": 0.5,
    "Amistoso": 0.5,
}


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
                competition_weight DOUBLE NOT NULL,
                home_advantage_points DOUBLE NOT NULL,
                neutral_site BOOLEAN,
                source_file VARCHAR,
                loaded_at TIMESTAMP,
                updated_at TIMESTAMP NOT NULL
            );
            """,
        )


def build_elo_history(
    db_path: Path = DB_PATH,
    *,
    initial_world_cup_probability_elo: float = INITIAL_WORLD_CUP_PROBABILITY_ELO,
    base_k_factor: float = BASE_K_FACTOR,
    home_advantage_points: float = HOME_ADVANTAGE_POINTS,
) -> None:
    """Iteratively compute World Cup Probability Elo and upsert it into DuckDB.

    Args:
        db_path: Path to the local DuckDB warehouse.
        initial_world_cup_probability_elo: Starting rating for unseen teams.
        base_k_factor: Base learning rate for the World Cup Probability Elo update.
        home_advantage_points: Home-field advantage in Elo points.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    initialize_elo_history(db_path)

    with duckdb.connect(str(db_path)) as con:
        _ensure_matches_exist(con)
        _delete_current_world_cup_history(con)
        matches = _load_matches(con)
        ratings: dict[str, float] = {}

        LOGGER.info("Processing %d matches in strict chronological order.", len(matches))
        for match in matches:
            home_before = ratings.get(match.home_team_id, initial_world_cup_probability_elo)
            away_before = ratings.get(match.away_team_id, initial_world_cup_probability_elo)

            competition_weight = _competition_weight(match.competition)
            k_factor = base_k_factor * competition_weight
            neutral_site = bool(match.neutral_site) if match.neutral_site is not None else False
            home_advantage = 0.0 if neutral_site else home_advantage_points

            home_expected = _expected_score(home_before + home_advantage, away_before)
            away_expected = 1.0 - home_expected
            home_actual, away_actual = _actual_scores(
                match.home_team_score,
                match.away_team_score,
            )

            home_after = home_before + k_factor * (home_actual - home_expected)
            away_after = away_before + k_factor * (away_actual - away_expected)

            ratings[match.home_team_id] = home_after
            ratings[match.away_team_id] = away_after

            _upsert_history(
                con=con,
                match=match,
                home_before=home_before,
                away_before=away_before,
                home_after=home_after,
                away_after=away_after,
                home_expected=home_expected,
                away_expected=away_expected,
                home_actual=home_actual,
                away_actual=away_actual,
                k_factor=k_factor,
                competition_weight=competition_weight,
                home_advantage_points=home_advantage if not neutral_site else 0.0,
                neutral_site=neutral_site,
            )

        LOGGER.info("World Cup Probability Elo history updated successfully.")


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


def _upsert_history(
    *,
    con: duckdb.DuckDBPyConnection,
    match: MatchRecord,
    home_before: float,
    away_before: float,
    home_after: float,
    away_after: float,
    home_expected: float,
    away_expected: float,
    home_actual: float,
    away_actual: float,
    k_factor: float,
    competition_weight: float,
    home_advantage_points: float,
    neutral_site: bool,
) -> None:
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
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            NULL,
            NULL,
            current_timestamp
        )
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
            competition_weight = excluded.competition_weight,
            home_advantage_points = excluded.home_advantage_points,
            neutral_site = excluded.neutral_site,
            updated_at = now();
        """,
        [
            match.match_id,
            match.match_date,
            match.competition,
            match.season,
            match.stage,
            match.home_team_id,
            match.away_team_id,
            home_before,
            away_before,
            home_after,
            away_after,
            home_expected,
            away_expected,
            home_actual,
            away_actual,
            k_factor,
            competition_weight,
            home_advantage_points,
            neutral_site,
        ],
    )


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


def _competition_weight(competition: str | None) -> float:
    """Return the competition multiplier for the given match label."""
    if competition is None:
        return 1.0

    normalized = competition.casefold()
    for label, weight in COMPETITION_WEIGHT_MULTIPLIERS.items():
        if label.casefold() in normalized:
            return weight

    return 1.0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def main() -> int:
    """CLI entrypoint."""
    build_elo_history()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
