"""Batch scoring for dashboard-ready win/draw/loss probabilities."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import xgboost as xgb

try:
    from .model import FEATURE_COLUMNS
    from .outcome_model import (
        OUTCOME_CALIBRATION_PATH,
        OUTCOME_CLASS_LABELS,
        OUTCOME_MODEL_PATH,
        apply_temperature_scaling,
    )
    from .settings import DB_PATH
    from .world_cup_2026_schedule import TEAM_NAMES, WorldCupFixture, world_cup_2026_fixtures
except ImportError:  # pragma: no cover - supports direct script execution.
    from model import FEATURE_COLUMNS
    from outcome_model import (
        OUTCOME_CALIBRATION_PATH,
        OUTCOME_CLASS_LABELS,
        OUTCOME_MODEL_PATH,
        apply_temperature_scaling,
    )
    from settings import DB_PATH
    from world_cup_2026_schedule import TEAM_NAMES, WorldCupFixture, world_cup_2026_fixtures

LOGGER = logging.getLogger(__name__)

OUTCOME_PREDICTIONS_TABLE = "outcome_predictions"

RAW_FEATURE_DEFAULTS: dict[str, float] = {
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


@dataclass(frozen=True, slots=True)
class OutcomePredictionRow:
    """One dashboard-ready predicted fixture row."""

    match_number: int
    round_name: str
    group_name: str | None
    match_date: datetime
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    home_win_pct: float
    draw_pct: float
    away_win_pct: float
    predicted_outcome: str
    calibration_temperature: float | None
    model_path: str
    created_at: datetime


def build_outcome_predictions(
    *,
    db_path: Path = DB_PATH,
    model_path: Path = OUTCOME_MODEL_PATH,
    calibration_path: Path = OUTCOME_CALIBRATION_PATH,
) -> int:
    """Score fixed World Cup group-stage fixtures and persist them to DuckDB."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB warehouse not found: {db_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Outcome model not found: {model_path}")

    model = load_outcome_model_artifact(model_path)
    calibration_temperature = load_calibration_temperature(calibration_path)
    team_features = build_current_world_cup_team_features(db_path)
    fixtures = tuple(
        fixture for fixture in world_cup_2026_fixtures() if fixture.round_name == "group_stage"
    )
    rows = score_fixtures(
        fixtures,
        team_features=team_features,
        model=model,
        model_path=model_path,
        calibration_temperature=calibration_temperature,
    )
    write_outcome_predictions(db_path=db_path, rows=rows)
    LOGGER.info("Wrote %d outcome predictions to %s.", len(rows), db_path)
    return len(rows)


def build_current_world_cup_team_features(db_path: Path = DB_PATH) -> dict[str, dict[str, float]]:
    """Build raw current team features keyed by official World Cup team code."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        feature_values_by_key: dict[str, dict[str, list[float]]] = {}
        _load_team_market_values(con, feature_values_by_key)
        _load_latest_world_cup_probability_elo(con, feature_values_by_key)
        _load_recent_form(con, feature_values_by_key)
        _load_world_football_elo_ratings(con, feature_values_by_key)
        _load_fifa_world_ranking(con, feature_values_by_key)
        _load_squad_attributes(con, feature_values_by_key)
        _load_prior_world_cup_history(con, feature_values_by_key)
        _load_prior_world_cup_discipline(con, feature_values_by_key)

    team_features: dict[str, dict[str, float]] = {}
    for code, official_name in TEAM_NAMES.items():
        keys = _official_team_lookup_keys(code, official_name)
        team_features[code] = {
            feature: _aggregate_feature(feature_values_by_key, keys, feature)
            for feature in RAW_FEATURE_DEFAULTS
        }
    return team_features


def score_fixtures(
    fixtures: tuple[WorldCupFixture, ...],
    *,
    team_features: dict[str, dict[str, float]],
    model: xgb.XGBClassifier,
    model_path: Path,
    calibration_temperature: float | None,
) -> list[OutcomePredictionRow]:
    """Score fixtures with the outcome classifier."""
    probabilities = np.asarray(
        [
            predict_match_probabilities(
                home_team_id=fixture.home_slot,
                away_team_id=fixture.away_slot,
                team_features=team_features,
                model=model,
                calibration_temperature=calibration_temperature,
            )
            for fixture in fixtures
        ],
        dtype=float,
    )

    created_at = datetime.now(UTC)
    rows: list[OutcomePredictionRow] = []
    for fixture, probability_row in zip(fixtures, probabilities, strict=True):
        predicted_class = int(np.argmax(probability_row))
        rows.append(
            OutcomePredictionRow(
                match_number=fixture.match_number,
                round_name=fixture.round_name,
                group_name=fixture.group_name,
                match_date=fixture.match_date,
                home_team_id=fixture.home_slot,
                home_team_name=TEAM_NAMES[fixture.home_slot],
                away_team_id=fixture.away_slot,
                away_team_name=TEAM_NAMES[fixture.away_slot],
                home_win_pct=round(100.0 * float(probability_row[0]), 4),
                draw_pct=round(100.0 * float(probability_row[1]), 4),
                away_win_pct=round(100.0 * float(probability_row[2]), 4),
                predicted_outcome=OUTCOME_CLASS_LABELS[predicted_class],
                calibration_temperature=calibration_temperature,
                model_path=str(model_path),
                created_at=created_at,
            )
        )
    return rows


def predict_match_probabilities(
    *,
    home_team_id: str,
    away_team_id: str,
    team_features: dict[str, dict[str, float]],
    model: xgb.XGBClassifier,
    calibration_temperature: float | None,
) -> np.ndarray:
    """Predict V/E/D probabilities for a home/away team pair."""
    if home_team_id not in team_features or away_team_id not in team_features:
        missing = sorted({home_team_id, away_team_id} - set(team_features))
        raise KeyError(f"Missing team features for: {', '.join(missing)}")

    vector = _fixture_feature_vector(team_features[home_team_id], team_features[away_team_id])
    probabilities = model.predict_proba(np.asarray([vector], dtype=float))
    if calibration_temperature is not None:
        probabilities = apply_temperature_scaling(probabilities, calibration_temperature)
    return np.asarray(probabilities[0], dtype=float)


def write_outcome_predictions(*, db_path: Path, rows: list[OutcomePredictionRow]) -> None:
    """Replace the dashboard outcome prediction table."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {OUTCOME_PREDICTIONS_TABLE} (
                match_number INTEGER PRIMARY KEY,
                round_name VARCHAR NOT NULL,
                group_name VARCHAR,
                match_date TIMESTAMP,
                home_team_id VARCHAR NOT NULL,
                home_team_name VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                away_team_name VARCHAR NOT NULL,
                home_win_pct DOUBLE NOT NULL,
                draw_pct DOUBLE NOT NULL,
                away_win_pct DOUBLE NOT NULL,
                predicted_outcome VARCHAR NOT NULL,
                calibration_temperature DOUBLE,
                model_path VARCHAR,
                created_at TIMESTAMP NOT NULL
            )
            """,
        )
        con.execute(f"DELETE FROM {OUTCOME_PREDICTIONS_TABLE}")
        con.executemany(
            f"""
            INSERT INTO {OUTCOME_PREDICTIONS_TABLE} (
                match_number,
                round_name,
                group_name,
                match_date,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_win_pct,
                draw_pct,
                away_win_pct,
                predicted_outcome,
                calibration_temperature,
                model_path,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_row_tuple(row) for row in rows],
        )


def _fixture_feature_vector(
    home_features: dict[str, float],
    away_features: dict[str, float],
) -> list[float]:
    values = {
        "world_cup_probability_elo_diff": (
            home_features["world_cup_probability_elo"] - away_features["world_cup_probability_elo"]
        ),
        "world_football_elo_ratings_diff": (
            home_features["world_football_elo_ratings"]
            - away_features["world_football_elo_ratings"]
        ),
        "fifa_world_ranking_points_diff": (
            home_features["fifa_world_ranking_points"] - away_features["fifa_world_ranking_points"]
        ),
        "fifa_world_ranking_rank_diff": (
            away_features["fifa_world_ranking_rank"] - home_features["fifa_world_ranking_rank"]
        ),
        "prior_world_cup_appearances_diff": (
            home_features["prior_world_cup_appearances"]
            - away_features["prior_world_cup_appearances"]
        ),
        "prior_world_cup_points_per_match_diff": (
            home_features["prior_world_cup_points_per_match"]
            - away_features["prior_world_cup_points_per_match"]
        ),
        "prior_world_cup_goal_diff_per_match_diff": (
            home_features["prior_world_cup_goal_diff_per_match"]
            - away_features["prior_world_cup_goal_diff_per_match"]
        ),
        "prior_world_cup_yellow_cards_per_match_diff": (
            home_features["prior_world_cup_yellow_cards_per_match"]
            - away_features["prior_world_cup_yellow_cards_per_match"]
        ),
        "prior_world_cup_sending_offs_per_match_diff": (
            home_features["prior_world_cup_sending_offs_per_match"]
            - away_features["prior_world_cup_sending_offs_per_match"]
        ),
        "prior_world_cup_fair_play_penalty_per_match_diff": (
            home_features["prior_world_cup_fair_play_penalty_per_match"]
            - away_features["prior_world_cup_fair_play_penalty_per_match"]
        ),
        "market_value_diff": home_features["market_value"] - away_features["market_value"],
        "avg_overall_diff": home_features["avg_overall"] - away_features["avg_overall"],
        "avg_pace_diff": home_features["avg_pace"] - away_features["avg_pace"],
        "avg_stamina_diff": home_features["avg_stamina"] - away_features["avg_stamina"],
        "squad_depth_proxy": (
            home_features["squad_depth_proxy"] - away_features["squad_depth_proxy"]
        ),
        "recent_form_diff": home_features["recent_form"] - away_features["recent_form"],
    }
    return [float(values[column]) for column in FEATURE_COLUMNS]


def _load_team_market_values(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "d_teams"):
        return
    rows = con.execute(
        """
        SELECT
            team_id,
            team_name,
            COALESCE(market_value_eur, total_market_value_eur, 0.0) AS market_value
        FROM d_teams
        """,
    ).fetchall()
    for team_id, team_name, market_value in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_id, team_name),
            {"market_value": float(market_value or 0.0)},
        )


def _load_latest_world_cup_probability_elo(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "f_elo_history"):
        return
    rows = con.execute(
        """
        WITH elo_union AS (
            SELECT match_id, match_date, home_team_id AS team_id, home_rating_after AS elo
            FROM f_elo_history
            UNION ALL
            SELECT match_id, match_date, away_team_id AS team_id, away_rating_after AS elo
            FROM f_elo_history
        ),
        ranked AS (
            SELECT
                e.team_id,
                t.team_name,
                e.elo,
                ROW_NUMBER() OVER (
                    PARTITION BY e.team_id
                    ORDER BY e.match_date DESC, e.match_id DESC
                ) AS rn
            FROM elo_union AS e
            LEFT JOIN d_teams AS t
                ON t.team_id = e.team_id
        )
        SELECT team_id, team_name, elo
        FROM ranked
        WHERE rn = 1
        """,
    ).fetchall()
    for team_id, team_name, elo in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_id, team_name),
            {"world_cup_probability_elo": float(elo or 1500.0)},
        )


def _load_recent_form(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "f_matches"):
        return
    rows = con.execute(
        """
        WITH team_match_history AS (
            SELECT
                match_id,
                match_date,
                home_team_id AS team_id,
                home_team_score AS goals_for,
                away_team_score AS goals_against
            FROM f_matches
            WHERE home_team_score IS NOT NULL AND away_team_score IS NOT NULL
            UNION ALL
            SELECT
                match_id,
                match_date,
                away_team_id AS team_id,
                away_team_score AS goals_for,
                home_team_score AS goals_against
            FROM f_matches
            WHERE home_team_score IS NOT NULL AND away_team_score IS NOT NULL
        ),
        recent_form AS (
            SELECT
                team_id,
                COALESCE(
                    AVG(goals_for) OVER (
                        PARTITION BY team_id
                        ORDER BY match_date, match_id
                        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                    ),
                    0.0
                ) - COALESCE(
                    AVG(goals_against) OVER (
                        PARTITION BY team_id
                        ORDER BY match_date, match_id
                        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                    ),
                    0.0
                ) AS recent_form,
                ROW_NUMBER() OVER (
                    PARTITION BY team_id
                    ORDER BY match_date DESC, match_id DESC
                ) AS rn
            FROM team_match_history
        )
        SELECT rf.team_id, t.team_name, rf.recent_form
        FROM recent_form AS rf
        LEFT JOIN d_teams AS t
            ON t.team_id = rf.team_id
        WHERE rf.rn = 1
        """,
    ).fetchall()
    for team_id, team_name, recent_form in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_id, team_name),
            {"recent_form": float(recent_form or 0.0)},
        )


def _load_world_football_elo_ratings(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "d_world_football_elo_ratings") or not _table_exists(
        con, "d_world_football_elo_team_aliases"
    ):
        return
    rows = con.execute(
        """
        SELECT a.team_alias, r.elo_rating
        FROM d_world_football_elo_team_aliases AS a
        INNER JOIN d_world_football_elo_ratings AS r
            ON r.world_football_team_code = a.world_football_team_code
        """,
    ).fetchall()
    for team_alias, elo_rating in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_alias,),
            {"world_football_elo_ratings": float(elo_rating or 1500.0)},
        )


def _load_fifa_world_ranking(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "d_fifa_world_ranking") or not _table_exists(
        con, "d_fifa_world_ranking_team_aliases"
    ):
        return
    rows = con.execute(
        """
        SELECT a.team_alias, r.ranking_points, CAST(r.fifa_rank AS DOUBLE) AS fifa_rank
        FROM d_fifa_world_ranking_team_aliases AS a
        INNER JOIN d_fifa_world_ranking AS r
            ON r.fifa_country_code = a.fifa_country_code
        """,
    ).fetchall()
    for team_alias, ranking_points, fifa_rank in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_alias,),
            {
                "fifa_world_ranking_points": float(ranking_points or 1500.0),
                "fifa_world_ranking_rank": float(fifa_rank or 0.0),
            },
        )


def _load_squad_attributes(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "d_squad_attributes"):
        return
    rows = con.execute(
        """
        SELECT
            team_id,
            avg_overall,
            avg_pace,
            avg_stamina,
            CAST(sampled_player_count AS DOUBLE) AS squad_depth_proxy
        FROM (
            SELECT
                team_id,
                avg_overall,
                avg_pace,
                avg_stamina,
                sampled_player_count,
                ROW_NUMBER() OVER (
                    PARTITION BY team_id
                    ORDER BY source_season DESC NULLS LAST, loaded_at DESC NULLS LAST
                ) AS rn
            FROM d_squad_attributes
        ) AS ranked
        WHERE rn = 1
        """,
    ).fetchall()
    for team_id, avg_overall, avg_pace, avg_stamina, squad_depth_proxy in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_id,),
            {
                "avg_overall": float(avg_overall or 0.0),
                "avg_pace": float(avg_pace or 0.0),
                "avg_stamina": float(avg_stamina or 0.0),
                "squad_depth_proxy": float(squad_depth_proxy or 0.0),
            },
        )


def _load_prior_world_cup_history(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "d_world_cup_prior_team_history"):
        return
    rows = con.execute(
        """
        SELECT
            team_name,
            team_code,
            prior_world_cup_appearances,
            prior_world_cup_points_per_match,
            prior_world_cup_goal_diff_per_match
        FROM d_world_cup_prior_team_history
        WHERE as_of_year = 2026
        """,
    ).fetchall()
    for team_name, team_code, appearances, points_per_match, goal_diff_per_match in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_name, team_code),
            {
                "prior_world_cup_appearances": float(appearances or 0.0),
                "prior_world_cup_points_per_match": float(points_per_match or 0.0),
                "prior_world_cup_goal_diff_per_match": float(goal_diff_per_match or 0.0),
            },
        )


def _load_prior_world_cup_discipline(
    con: duckdb.DuckDBPyConnection,
    feature_values_by_key: dict[str, dict[str, list[float]]],
) -> None:
    if not _table_exists(con, "d_world_cup_prior_discipline_history"):
        return
    rows = con.execute(
        """
        SELECT
            team_name,
            team_code,
            prior_world_cup_yellow_cards_per_match,
            prior_world_cup_sending_offs_per_match,
            prior_world_cup_fair_play_penalty_per_match
        FROM d_world_cup_prior_discipline_history
        WHERE as_of_year = 2026
        """,
    ).fetchall()
    for team_name, team_code, yellow_cards, sending_offs, fair_play_penalty in rows:
        _add_feature_values(
            feature_values_by_key,
            (team_name, team_code),
            {
                "prior_world_cup_yellow_cards_per_match": float(yellow_cards or 0.0),
                "prior_world_cup_sending_offs_per_match": float(sending_offs or 0.0),
                "prior_world_cup_fair_play_penalty_per_match": float(fair_play_penalty or 0.0),
            },
        )


def _add_feature_values(
    feature_values_by_key: dict[str, dict[str, list[float]]],
    keys: tuple[object, ...],
    values: dict[str, float],
) -> None:
    for key in keys:
        if key is None:
            continue
        normalized_key = _normalize_team_key(str(key))
        if not normalized_key:
            continue
        target = feature_values_by_key.setdefault(normalized_key, {})
        for feature, value in values.items():
            if math.isnan(float(value)):
                continue
            target.setdefault(feature, []).append(float(value))


def _aggregate_feature(
    feature_values_by_key: dict[str, dict[str, list[float]]],
    keys: tuple[str, ...],
    feature: str,
) -> float:
    values = [
        value
        for key in keys
        for value in feature_values_by_key.get(key, {}).get(feature, [])
        if not math.isnan(float(value))
    ]
    if not values:
        return RAW_FEATURE_DEFAULTS[feature]
    if feature == "fifa_world_ranking_rank":
        positive_values = [value for value in values if value > 0.0]
        return min(positive_values) if positive_values else RAW_FEATURE_DEFAULTS[feature]
    if feature == "recent_form":
        return max(values, key=lambda value: abs(value))
    return max(values)


def load_outcome_model_artifact(model_path: Path) -> xgb.XGBClassifier:
    """Load the trained V/E/D outcome model."""
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def load_calibration_temperature(calibration_path: Path) -> float | None:
    """Load the optional scalar calibration temperature."""
    if not calibration_path.exists():
        return None
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    temperature = payload.get("temperature")
    return None if temperature is None else float(temperature)


def _row_tuple(row: OutcomePredictionRow) -> tuple[object, ...]:
    return (
        row.match_number,
        row.round_name,
        row.group_name,
        row.match_date,
        row.home_team_id,
        row.home_team_name,
        row.away_team_id,
        row.away_team_name,
        row.home_win_pct,
        row.draw_pct,
        row.away_win_pct,
        row.predicted_outcome,
        row.calibration_temperature,
        row.model_path,
        row.created_at,
    )


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return (
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
        > 0
    )


def _official_team_lookup_keys(code: str, official_name: str) -> tuple[str, ...]:
    aliases = {
        "BIH": ("Bosnia", "Bosnia-Herzegovina", "Bosnia and Herzegovina"),
        "CIV": ("Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire"),
        "COD": ("DR Congo", "Congo DR", "Congo Democratic Republic"),
        "CPV": ("Cape Verde", "Cabo Verde"),
        "CZE": ("Czech Republic", "Czechia"),
        "IRN": ("Iran", "IR Iran"),
        "KOR": ("South Korea", "Korea Republic"),
        "RSA": ("South Africa",),
        "SUI": ("Switzerland", "Swiss"),
        "TUR": ("Turkey", "Türkiye"),
        "USA": ("United States", "USMNT"),
    }
    names = (code, official_name, *aliases.get(code, ()))
    return tuple(_normalize_team_key(name) for name in names)


def _normalize_team_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for dashboard outcome prediction scoring."""
    parser = argparse.ArgumentParser(description="Write dashboard V/E/D outcome predictions.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="DuckDB warehouse path.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=OUTCOME_MODEL_PATH,
        help="Trained outcome model path.",
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=OUTCOME_CALIBRATION_PATH,
        help="Outcome calibration JSON path.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    build_outcome_predictions(
        db_path=args.db_path,
        model_path=args.model_path,
        calibration_path=args.calibration_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
