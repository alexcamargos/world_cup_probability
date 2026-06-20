"""Project orchestrator for the full World Cup pipeline.

The execution order is:
db_init -> elo_engine -> feature_pipeline -> model -> simulator
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import polars as pl
import xgboost as xgb

from db_init import initialize_database
from elo_engine import DB_PATH as WAREHOUSE_DB_PATH
from elo_engine import build_elo_history
from feature_pipeline import build_feature_frame
from model import BEESWARM_PATH, FEATURE_COLUMNS, MODEL_PATH, save_model, explain_model, prepare_matrices, train_poisson_model
from simulator import DB_PATH as SIMULATOR_DB_PATH
from simulator import TeamLambda, simulate_world_cup

LOGGER = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 100_000
DEFAULT_BATCH_SIZE = 2_500
DEFAULT_SEED = 42
MIN_WORLD_CUP_TEAMS = 32


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the orchestrator."""
    parser = argparse.ArgumentParser(description="Run the full World Cup pipeline.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of Monte Carlo tournament simulations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used when persisting simulated results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=WAREHOUSE_DB_PATH,
        help="DuckDB warehouse path.",
    )
    return parser


def run_pipeline(
    *,
    db_path: Path,
    iterations: int,
    batch_size: int,
    seed: int | None,
) -> None:
    """Run the project end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    LOGGER.info("Step 1/5: initializing DuckDB warehouse.")
    initialize_database(db_path=db_path)

    LOGGER.info("Step 2/5: building Elo history.")
    build_elo_history(db_path=db_path)

    LOGGER.info("Step 3/5: building feature frame.")
    feature_frame = build_feature_frame(db_path=db_path)

    LOGGER.info("Step 4/5: training Poisson XGBoost model.")
    X_train, X_valid, y_train, y_valid, feature_names = prepare_matrices(feature_frame)
    model = train_poisson_model(X_train, y_train, X_valid, y_valid)
    save_model(model, MODEL_PATH)
    explain_model(model, X_valid, feature_names, BEESWARM_PATH)

    LOGGER.info("Step 5/5: building team lambdas and running Monte Carlo simulation.")
    team_lambdas = _build_team_lambdas(db_path=db_path, model=model)
    simulate_world_cup(
        team_lambdas,
        iterations=iterations,
        batch_size=batch_size,
        db_path=SIMULATOR_DB_PATH,
        seed=seed,
    )

    LOGGER.info("Pipeline completed successfully.")


def _build_team_lambdas(
    *,
    db_path: Path,
    model: xgb.XGBRegressor,
) -> list[TeamLambda]:
    """Build team-level lambdas from the trained model.

    The XGBoost model was trained on difference-based features, so we predict
    each team's attack intensity against the league-average reference point.
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        query = """
            WITH team_match_history AS (
                SELECT
                    match_id,
                    match_date,
                    home_team_id AS team_id,
                    home_team_score AS goals_for,
                    away_team_score AS goals_against
                FROM f_matches
                UNION ALL
                SELECT
                    match_id,
                    match_date,
                    away_team_id AS team_id,
                    away_team_score AS goals_for,
                    home_team_score AS goals_against
                FROM f_matches
            ),
            latest_elo AS (
                SELECT team_id, elo_now
                FROM (
                    SELECT
                        team_id,
                        elo_now,
                        ROW_NUMBER() OVER (
                            PARTITION BY team_id
                            ORDER BY match_date DESC, match_id DESC
                        ) AS rn
                    FROM (
                        SELECT
                            match_id,
                            match_date,
                            home_team_id AS team_id,
                            home_rating_after AS elo_now
                        FROM f_elo_history
                        UNION ALL
                        SELECT
                            match_id,
                            match_date,
                            away_team_id AS team_id,
                            away_rating_after AS elo_now
                        FROM f_elo_history
                    ) AS elo_union
                ) AS ranked
                WHERE rn = 1
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
                    ) AS goals_for_last5,
                    COALESCE(
                        AVG(goals_against) OVER (
                            PARTITION BY team_id
                            ORDER BY match_date, match_id
                            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                        ),
                        0.0
                    ) AS goals_against_last5,
                    ROW_NUMBER() OVER (
                        PARTITION BY team_id
                        ORDER BY match_date DESC, match_id DESC
                    ) AS rn
                FROM team_match_history
            ),
            latest_form AS (
                SELECT
                    team_id,
                    goals_for_last5 - goals_against_last5 AS recent_form
                FROM recent_form
                WHERE rn = 1
            )
            SELECT
                t.team_id,
                t.team_name,
                COALESCE(le.elo_now, 1500.0) AS elo_now,
                COALESCE(t.market_value_eur, 0.0) AS market_value_eur,
                COALESCE(lf.recent_form, 0.0) AS recent_form
            FROM d_teams AS t
            LEFT JOIN latest_elo AS le
                ON le.team_id = t.team_id
            LEFT JOIN latest_form AS lf
                ON lf.team_id = t.team_id
            ORDER BY elo_now DESC, market_value_eur DESC, team_name ASC
        """
        team_frame = con.sql(query).pl()

    if team_frame.height < MIN_WORLD_CUP_TEAMS:
        raise RuntimeError(
            f"Need at least {MIN_WORLD_CUP_TEAMS} teams for the World Cup bracket, got {team_frame.height}.",
        )

    league_means = team_frame.select(
        [
            pl.mean("elo_now").alias("elo_mean"),
            pl.mean("market_value_eur").alias("market_value_mean"),
            pl.mean("recent_form").alias("recent_form_mean"),
        ]
    ).row(0)
    elo_mean, market_value_mean, recent_form_mean = map(float, league_means)

    scored_frame = team_frame.with_columns(
        [
            (pl.col("elo_now") - pl.lit(elo_mean)).alias("elo_diff"),
            (pl.col("market_value_eur") - pl.lit(market_value_mean)).alias("market_value_diff"),
            (pl.col("recent_form") - pl.lit(recent_form_mean)).alias("recent_form_diff"),
        ]
    )

    predictions = model.predict(scored_frame.select(list(FEATURE_COLUMNS)).to_numpy())
    scored_frame = scored_frame.with_columns(pl.Series("lambda_goals", predictions))

    top_teams = scored_frame.sort("lambda_goals", descending=True).head(MIN_WORLD_CUP_TEAMS)
    seeded = _seed_bracket(top_teams)

    return [
        TeamLambda(
            team_id=row["team_id"],
            team_name=row["team_name"],
            lambda_goals=float(row["lambda_goals"]),
        )
        for row in seeded.to_dicts()
    ]


def _seed_bracket(team_frame: pl.DataFrame) -> pl.DataFrame:
    """Seed bracket order as strongest vs weakest, second strongest vs second weakest."""
    ordered = team_frame.sort("lambda_goals", descending=True)
    rows = ordered.to_dicts()

    bracket_rows: list[dict[str, object]] = []
    left = 0
    right = len(rows) - 1
    while left <= right:
        bracket_rows.append(rows[left])
        if left != right:
            bracket_rows.append(rows[right])
        left += 1
        right -= 1

    return pl.DataFrame(bracket_rows)


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    run_pipeline(
        db_path=args.db_path,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
