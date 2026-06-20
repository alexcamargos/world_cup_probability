"""Project orchestrator for the full World Cup pipeline.

The execution order is:
db_init -> World Cup Probability Elo -> World Football Elo Ratings -> features -> model -> simulator
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl
import xgboost as xgb

try:
    from .analytics import export_analytics
    from .db_init import initialize_database
    from .elo_engine import DB_PATH as WAREHOUSE_DB_PATH
    from .elo_engine import build_elo_history
    from .feature_pipeline import build_feature_frame
    from .model import (
        BEESWARM_PATH,
        FEATURE_COLUMNS,
        MODEL_PATH,
        explain_model,
        prepare_matrices,
        save_model,
        train_poisson_model,
    )
    from .simulator import TeamLambda, simulate_world_cup
    from .world_football_elo_ratings import load_world_football_elo_ratings
except ImportError:  # pragma: no cover - supports direct script execution.
    from analytics import export_analytics
    from db_init import initialize_database
    from elo_engine import DB_PATH as WAREHOUSE_DB_PATH
    from elo_engine import build_elo_history
    from feature_pipeline import build_feature_frame
    from model import (
        BEESWARM_PATH,
        FEATURE_COLUMNS,
        MODEL_PATH,
        explain_model,
        prepare_matrices,
        save_model,
        train_poisson_model,
    )
    from simulator import TeamLambda, simulate_world_cup
    from world_football_elo_ratings import load_world_football_elo_ratings

LOGGER = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 100_000
DEFAULT_BATCH_SIZE = 2_500
DEFAULT_SEED = 42
MIN_WORLD_CUP_TEAMS = 32


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime configuration for the end-to-end pipeline."""

    db_path: Path
    iterations: int
    batch_size: int
    seed: int | None

    def validate(self) -> None:
        """Validate user-provided runtime values before work starts."""
        if self.iterations <= 0:
            raise ValueError("iterations must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")


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
    config = PipelineConfig(
        db_path=db_path,
        iterations=iterations,
        batch_size=batch_size,
        seed=seed,
    )
    config.validate()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    LOGGER.info("Step 1/7: initializing DuckDB warehouse.")
    initialize_database(db_path=config.db_path)

    LOGGER.info("Step 2/7: building World Cup Probability Elo history.")
    build_elo_history(db_path=config.db_path)

    LOGGER.info("Step 3/7: loading World Football Elo Ratings snapshot.")
    load_world_football_elo_ratings(db_path=config.db_path)

    LOGGER.info("Step 4/7: building feature frame.")
    feature_frame = build_feature_frame(db_path=config.db_path)

    LOGGER.info("Step 5/7: training Poisson XGBoost model.")
    X_train, X_valid, y_train, y_valid, feature_names = prepare_matrices(feature_frame)
    model = train_poisson_model(X_train, y_train, X_valid, y_valid)
    save_model(model, MODEL_PATH)
    explain_model(model, X_valid, feature_names, BEESWARM_PATH)

    LOGGER.info("Step 6/7: building team lambdas and running Monte Carlo simulation.")
    team_lambdas = _build_team_lambdas(db_path=config.db_path, model=model)
    simulate_world_cup(
        team_lambdas,
        iterations=config.iterations,
        batch_size=config.batch_size,
        db_path=config.db_path,
        seed=config.seed,
    )

    LOGGER.info("Step 7/7: exporting analytical summaries.")
    export_analytics(db_path=config.db_path)

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
        has_world_football_elo_ratings = _world_football_elo_ratings_tables_available(con)
        world_football_elo_ratings_cte = (
            """
            world_football_elo_ratings AS (
                SELECT team_id, world_football_elo_ratings_now
                FROM (
                    SELECT
                        a.team_alias AS team_id,
                        r.elo_rating AS world_football_elo_ratings_now,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.team_alias_key
                            ORDER BY r.rating_date DESC NULLS LAST, r.elo_rank ASC
                        ) AS rn
                    FROM d_world_football_elo_team_aliases AS a
                    INNER JOIN d_world_football_elo_ratings AS r
                        ON r.world_football_team_code = a.world_football_team_code
                ) AS ranked_world_football_elo_ratings
                WHERE rn = 1
            ),
            """
            if has_world_football_elo_ratings
            else ""
        )
        world_football_elo_ratings_select = (
            "COALESCE(ee.world_football_elo_ratings_now, "
            "le.world_cup_probability_elo_now, 1500.0) AS world_football_elo_ratings_now"
            if has_world_football_elo_ratings
            else (
                "COALESCE(le.world_cup_probability_elo_now, 1500.0) "
                "AS world_football_elo_ratings_now"
            )
        )
        world_football_elo_ratings_join = (
            """
            LEFT JOIN world_football_elo_ratings AS ee
                ON lower(ee.team_id) = lower(t.team_id)
                OR regexp_replace(lower(ee.team_id), '[^a-z0-9]+', '', 'g')
                    = regexp_replace(lower(t.team_id), '[^a-z0-9]+', '', 'g')
            """
            if has_world_football_elo_ratings
            else ""
        )
        query = f"""
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
            latest_world_cup_probability_elo AS (
                SELECT team_id, world_cup_probability_elo_now
                FROM (
                    SELECT
                        team_id,
                        world_cup_probability_elo_now,
                        ROW_NUMBER() OVER (
                            PARTITION BY team_id
                            ORDER BY match_date DESC, match_id DESC
                        ) AS rn
                    FROM (
                        SELECT
                            match_id,
                            match_date,
                            home_team_id AS team_id,
                            home_rating_after AS world_cup_probability_elo_now
                        FROM f_elo_history
                        UNION ALL
                        SELECT
                            match_id,
                            match_date,
                            away_team_id AS team_id,
                            away_rating_after AS world_cup_probability_elo_now
                        FROM f_elo_history
                    ) AS elo_union
                ) AS ranked
                WHERE rn = 1
            ),
            {world_football_elo_ratings_cte}
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
                COALESCE(le.world_cup_probability_elo_now, 1500.0) AS world_cup_probability_elo_now,
                {world_football_elo_ratings_select},
                COALESCE(t.market_value_eur, 0.0) AS market_value_eur,
                COALESCE(lf.recent_form, 0.0) AS recent_form
            FROM d_teams AS t
            LEFT JOIN latest_world_cup_probability_elo AS le
                ON le.team_id = t.team_id
            {world_football_elo_ratings_join}
            LEFT JOIN latest_form AS lf
                ON lf.team_id = t.team_id
            ORDER BY world_cup_probability_elo_now DESC, market_value_eur DESC, team_name ASC
        """
        team_frame = con.sql(query).pl()

    if team_frame.height < MIN_WORLD_CUP_TEAMS:
        raise RuntimeError(
            "Need at least "
            f"{MIN_WORLD_CUP_TEAMS} teams for the World Cup bracket, got {team_frame.height}.",
        )

    league_means = team_frame.select(
        [
            pl.mean("world_cup_probability_elo_now").alias("world_cup_probability_elo_mean"),
            pl.mean("world_football_elo_ratings_now").alias("world_football_elo_ratings_mean"),
            pl.mean("market_value_eur").alias("market_value_mean"),
            pl.mean("recent_form").alias("recent_form_mean"),
        ]
    ).row(0)
    (
        world_cup_probability_elo_mean,
        world_football_elo_ratings_mean,
        market_value_mean,
        recent_form_mean,
    ) = map(float, league_means)

    scored_frame = team_frame.with_columns(
        [
            (
                pl.col("world_cup_probability_elo_now") - pl.lit(world_cup_probability_elo_mean)
            ).alias("world_cup_probability_elo_diff"),
            (
                pl.col("world_football_elo_ratings_now") - pl.lit(world_football_elo_ratings_mean)
            ).alias("world_football_elo_ratings_diff"),
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


def _world_football_elo_ratings_tables_available(con: duckdb.DuckDBPyConnection) -> bool:
    tables = ("d_world_football_elo_ratings", "d_world_football_elo_team_aliases")
    return all(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table],
        ).fetchone()[0]
        > 0
        for table in tables
    )


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
