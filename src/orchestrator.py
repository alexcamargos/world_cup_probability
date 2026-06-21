"""Project orchestrator for the full World Cup pipeline.

The execution order is:
db_init -> World Cup Probability Elo -> World Football Elo Ratings -> FIFA World Ranking
-> features -> model -> simulator
"""

from __future__ import annotations

import argparse
import logging
import re
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
    from .fifa_world_ranking import load_fifa_world_ranking
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
    from .world_cup_2026_schedule import TEAM_COUNTRIES, TEAM_NAMES
    from .world_football_elo_ratings import load_world_football_elo_ratings
except ImportError:  # pragma: no cover - supports direct script execution.
    from analytics import export_analytics
    from db_init import initialize_database
    from elo_engine import DB_PATH as WAREHOUSE_DB_PATH
    from elo_engine import build_elo_history
    from feature_pipeline import build_feature_frame
    from fifa_world_ranking import load_fifa_world_ranking
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
    from world_cup_2026_schedule import TEAM_COUNTRIES, TEAM_NAMES
    from world_football_elo_ratings import load_world_football_elo_ratings

LOGGER = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 100_000
DEFAULT_BATCH_SIZE = 2_500
DEFAULT_SEED = 42
MIN_WORLD_CUP_TEAMS = 48


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

    LOGGER.info("Step 1/8: initializing DuckDB warehouse.")
    initialize_database(db_path=config.db_path)

    LOGGER.info("Step 2/8: building World Cup Probability Elo history.")
    build_elo_history(db_path=config.db_path)

    LOGGER.info("Step 3/8: loading World Football Elo Ratings snapshot.")
    load_world_football_elo_ratings(db_path=config.db_path)

    LOGGER.info("Step 4/8: loading FIFA World Ranking snapshot.")
    load_fifa_world_ranking(db_path=config.db_path)

    LOGGER.info("Step 5/8: building feature frame.")
    feature_frame = build_feature_frame(db_path=config.db_path)

    LOGGER.info("Step 6/8: training Poisson XGBoost model.")
    X_train, X_valid, y_train, y_valid, feature_names = prepare_matrices(feature_frame)
    model = train_poisson_model(X_train, y_train, X_valid, y_valid)
    save_model(model, MODEL_PATH)
    explain_model(model, X_valid, feature_names, BEESWARM_PATH)

    LOGGER.info("Step 7/8: building team lambdas and running Monte Carlo simulation.")
    team_lambdas = _build_team_lambdas(db_path=config.db_path, model=model)
    simulate_world_cup(
        team_lambdas,
        iterations=config.iterations,
        batch_size=config.batch_size,
        db_path=config.db_path,
        seed=config.seed,
    )

    LOGGER.info("Step 8/8: exporting analytical summaries.")
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
        has_fifa_world_ranking = _fifa_world_ranking_tables_available(con)
        has_squad_attributes = _squad_attributes_table_available(con)
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
        fifa_world_ranking_cte = (
            """
            fifa_world_ranking AS (
                SELECT team_id, fifa_world_ranking_points, fifa_world_ranking_rank
                FROM (
                    SELECT
                        a.team_alias AS team_id,
                        r.ranking_points AS fifa_world_ranking_points,
                        CAST(r.fifa_rank AS DOUBLE) AS fifa_world_ranking_rank,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.team_alias_key
                            ORDER BY r.ranking_date DESC, r.fifa_rank ASC
                        ) AS rn
                    FROM d_fifa_world_ranking_team_aliases AS a
                    INNER JOIN d_fifa_world_ranking AS r
                        ON r.fifa_country_code = a.fifa_country_code
                ) AS ranked_fifa_world_ranking
                WHERE rn = 1
            ),
            """
            if has_fifa_world_ranking
            else ""
        )
        squad_attributes_ctes = (
            """
            latest_squad_attributes AS (
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
                ) AS ranked_squad_attributes
                WHERE rn = 1
            ),
            squad_attribute_defaults AS (
                SELECT
                    COALESCE(AVG(avg_overall), 0.0) AS avg_overall,
                    COALESCE(AVG(avg_pace), 0.0) AS avg_pace,
                    COALESCE(AVG(avg_stamina), 0.0) AS avg_stamina,
                    COALESCE(AVG(squad_depth_proxy), 0.0) AS squad_depth_proxy
                FROM latest_squad_attributes
            ),
            """
            if has_squad_attributes
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
        fifa_world_ranking_points_select = (
            "COALESCE(fr.fifa_world_ranking_points, le.world_cup_probability_elo_now, 1500.0) "
            "AS fifa_world_ranking_points"
            if has_fifa_world_ranking
            else "COALESCE(le.world_cup_probability_elo_now, 1500.0) AS fifa_world_ranking_points"
        )
        fifa_world_ranking_rank_select = (
            "COALESCE(fr.fifa_world_ranking_rank, 0.0) AS fifa_world_ranking_rank"
            if has_fifa_world_ranking
            else "0.0 AS fifa_world_ranking_rank"
        )
        fifa_world_ranking_join = (
            """
            LEFT JOIN fifa_world_ranking AS fr
                ON lower(fr.team_id) = lower(t.team_id)
                OR regexp_replace(lower(fr.team_id), '[^a-z0-9]+', '', 'g')
                    = regexp_replace(lower(t.team_id), '[^a-z0-9]+', '', 'g')
            """
            if has_fifa_world_ranking
            else ""
        )
        squad_attributes_select = (
            """
                COALESCE(sa.avg_overall, sad.avg_overall) AS avg_overall,
                COALESCE(sa.avg_pace, sad.avg_pace) AS avg_pace,
                COALESCE(sa.avg_stamina, sad.avg_stamina) AS avg_stamina,
                COALESCE(sa.squad_depth_proxy, sad.squad_depth_proxy) AS squad_depth_proxy,
            """
            if has_squad_attributes
            else """
                0.0 AS avg_overall,
                0.0 AS avg_pace,
                0.0 AS avg_stamina,
                0.0 AS squad_depth_proxy,
            """
        )
        squad_attributes_join = (
            """
            CROSS JOIN squad_attribute_defaults AS sad
            LEFT JOIN latest_squad_attributes AS sa
                ON lower(sa.team_id) = lower(t.team_id)
                OR regexp_replace(lower(sa.team_id), '[^a-z0-9]+', '', 'g')
                    = regexp_replace(lower(t.team_id), '[^a-z0-9]+', '', 'g')
            """
            if has_squad_attributes
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
            {fifa_world_ranking_cte}
            {squad_attributes_ctes}
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
                {fifa_world_ranking_points_select},
                {fifa_world_ranking_rank_select},
                COALESCE(t.market_value_eur, 0.0) AS market_value_eur,
                {squad_attributes_select}
                COALESCE(lf.recent_form, 0.0) AS recent_form
            FROM d_teams AS t
            LEFT JOIN latest_world_cup_probability_elo AS le
                ON le.team_id = t.team_id
            {world_football_elo_ratings_join}
            {fifa_world_ranking_join}
            {squad_attributes_join}
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
            pl.mean("fifa_world_ranking_points").alias("fifa_world_ranking_points_mean"),
            pl.mean("fifa_world_ranking_rank").alias("fifa_world_ranking_rank_mean"),
            pl.mean("market_value_eur").alias("market_value_mean"),
            pl.mean("avg_overall").alias("avg_overall_mean"),
            pl.mean("avg_pace").alias("avg_pace_mean"),
            pl.mean("avg_stamina").alias("avg_stamina_mean"),
            pl.mean("squad_depth_proxy").alias("squad_depth_proxy_mean"),
            pl.mean("recent_form").alias("recent_form_mean"),
        ]
    ).row(0)
    (
        world_cup_probability_elo_mean,
        world_football_elo_ratings_mean,
        fifa_world_ranking_points_mean,
        fifa_world_ranking_rank_mean,
        market_value_mean,
        avg_overall_mean,
        avg_pace_mean,
        avg_stamina_mean,
        squad_depth_proxy_mean,
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
            (pl.col("fifa_world_ranking_points") - pl.lit(fifa_world_ranking_points_mean)).alias(
                "fifa_world_ranking_points_diff"
            ),
            (pl.lit(fifa_world_ranking_rank_mean) - pl.col("fifa_world_ranking_rank")).alias(
                "fifa_world_ranking_rank_diff"
            ),
            (pl.col("market_value_eur") - pl.lit(market_value_mean)).alias("market_value_diff"),
            (pl.col("avg_overall") - pl.lit(avg_overall_mean)).alias("avg_overall_diff"),
            (pl.col("avg_pace") - pl.lit(avg_pace_mean)).alias("avg_pace_diff"),
            (pl.col("avg_stamina") - pl.lit(avg_stamina_mean)).alias("avg_stamina_diff"),
            (pl.col("squad_depth_proxy") - pl.lit(squad_depth_proxy_mean)).alias(
                "squad_depth_proxy"
            ),
            (pl.col("recent_form") - pl.lit(recent_form_mean)).alias("recent_form_diff"),
        ]
    )

    predictions = model.predict(scored_frame.select(list(FEATURE_COLUMNS)).to_numpy())
    scored_frame = scored_frame.with_columns(pl.Series("lambda_goals", predictions))

    return _world_cup_2026_team_lambdas(scored_frame)


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


def _world_cup_2026_team_lambdas(scored_frame: pl.DataFrame) -> list[TeamLambda]:
    """Return lambdas for the 48 official FIFA World Cup 2026 participants."""
    rows = scored_frame.to_dicts()
    lookup: dict[str, dict[str, object]] = {}
    for row in rows:
        lookup[_normalize_team_key(str(row["team_id"]))] = row
        lookup[_normalize_team_key(str(row["team_name"]))] = row

    average_lambda = float(scored_frame.get_column("lambda_goals").mean() or 1.0)
    team_lambdas: list[TeamLambda] = []
    missing_codes: list[str] = []

    for code, official_name in sorted(TEAM_NAMES.items()):
        row = next(
            (
                lookup[key]
                for key in _official_team_lookup_keys(code, official_name)
                if key in lookup
            ),
            None,
        )
        if row is None:
            lambda_goals = average_lambda
            missing_codes.append(code)
        else:
            lambda_goals = float(row["lambda_goals"])
        team_lambdas.append(
            TeamLambda(
                team_id=code,
                team_name=official_name,
                lambda_goals=lambda_goals,
                country_code=TEAM_COUNTRIES.get(code, code),
            )
        )

    if missing_codes:
        LOGGER.warning(
            "Using average lambda for World Cup 2026 teams missing from warehouse: %s",
            ", ".join(f"{code} ({TEAM_NAMES[code]})" for code in missing_codes),
        )

    return team_lambdas


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
        "TUR": ("Turkey", "Türkiye"),
        "USA": ("United States", "USA", "USMNT"),
    }
    names = (code, official_name, *aliases.get(code, ()))
    return tuple(_normalize_team_key(name) for name in names)


def _normalize_team_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


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


def _fifa_world_ranking_tables_available(con: duckdb.DuckDBPyConnection) -> bool:
    tables = ("d_fifa_world_ranking", "d_fifa_world_ranking_team_aliases")
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


def _squad_attributes_table_available(con: duckdb.DuckDBPyConnection) -> bool:
    return (
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'd_squad_attributes'
            """,
        ).fetchone()[0]
        > 0
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
