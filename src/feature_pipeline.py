"""Feature engineering pipeline for the match-level regression dataset.

The pipeline reads a consolidated DuckDB view into Polars without materializing
intermediate CSV/Parquet files, then computes recent-form features with window
functions before returning a compact regression frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "world_cup.duckdb"


def build_feature_frame(db_path: Path = DB_PATH) -> pl.DataFrame:
    """Build the final training dataframe.

    Args:
        db_path: Path to the DuckDB warehouse.

    Returns:
        A Polars DataFrame containing only predictive features and the target.

    Raises:
        FileNotFoundError: If the DuckDB warehouse does not exist.
        RuntimeError: If the required source tables are missing or empty.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB warehouse not found: {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as con:
        _validate_sources(con)
        base_df = _load_consolidated_matches(con)

    feature_df = _build_team_level_features(base_df)
    LOGGER.info("Feature frame built with %d rows and %d columns.", *feature_df.shape)
    return feature_df


def _validate_sources(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the minimum warehouse state exists before feature generation."""
    required_tables = ("f_matches", "f_elo_history", "d_teams")
    missing_tables = [
        table
        for table in required_tables
        if con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table],
        ).fetchone()[0]
        == 0
    ]
    if missing_tables:
        raise RuntimeError(f"Missing required DuckDB tables: {', '.join(missing_tables)}")

    match_count = con.execute("SELECT COUNT(*) FROM f_matches").fetchone()[0]
    elo_count = con.execute("SELECT COUNT(*) FROM f_elo_history").fetchone()[0]
    if int(match_count) == 0:
        raise RuntimeError(
            "f_matches is empty. Run the ingestion and World Cup Probability Elo first."
        )
    if int(elo_count) == 0:
        raise RuntimeError("f_elo_history is empty. Run the World Cup Probability Elo step first.")


def _load_consolidated_matches(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Read the consolidated match-level relation from DuckDB into Polars."""
    has_world_football_elo_ratings = _world_football_elo_ratings_tables_available(con)
    home_team_key = "regexp_replace(lower(m.home_team_id), '[^a-z0-9]+', '', 'g')"
    away_team_key = "regexp_replace(lower(m.away_team_id), '[^a-z0-9]+', '', 'g')"
    home_world_football_elo_ratings_expr = (
        "COALESCE(her.elo_rating, e.home_rating_before)"
        if has_world_football_elo_ratings
        else "e.home_rating_before"
    )
    away_world_football_elo_ratings_expr = (
        "COALESCE(aer.elo_rating, e.away_rating_before)"
        if has_world_football_elo_ratings
        else "e.away_rating_before"
    )
    world_football_elo_ratings_joins = (
        f"""
        LEFT JOIN d_world_football_elo_team_aliases AS hea
            ON hea.team_alias_key = lower(m.home_team_id)
            OR hea.normalized_team_alias = {home_team_key}
        LEFT JOIN d_world_football_elo_ratings AS her
            ON her.world_football_team_code = hea.world_football_team_code
        LEFT JOIN d_world_football_elo_team_aliases AS aea
            ON aea.team_alias_key = lower(m.away_team_id)
            OR aea.normalized_team_alias = {away_team_key}
        LEFT JOIN d_world_football_elo_ratings AS aer
            ON aer.world_football_team_code = aea.world_football_team_code
        """
        if has_world_football_elo_ratings
        else ""
    )

    query = f"""
        SELECT
            m.match_id,
            m.match_date,
            m.competition,
            m.season,
            m.stage,
            m.home_team_id,
            m.away_team_id,
            m.home_team_score,
            m.away_team_score,
            m.neutral_site,
            e.home_rating_before AS home_world_cup_probability_elo_before,
            e.away_rating_before AS away_world_cup_probability_elo_before,
            {home_world_football_elo_ratings_expr} AS home_world_football_elo_ratings,
            {away_world_football_elo_ratings_expr} AS away_world_football_elo_ratings,
            COALESCE(ht.market_value_eur, 0.0) AS home_market_value_eur,
            COALESCE(awt.market_value_eur, 0.0) AS away_market_value_eur
        FROM f_matches AS m
        INNER JOIN f_elo_history AS e
            ON e.match_id = m.match_id
        {world_football_elo_ratings_joins}
        LEFT JOIN d_teams AS ht
            ON ht.team_id = m.home_team_id
        LEFT JOIN d_teams AS awt
            ON awt.team_id = m.away_team_id
        WHERE m.match_date IS NOT NULL
          AND m.home_team_score IS NOT NULL
          AND m.away_team_score IS NOT NULL
        ORDER BY m.match_date ASC, m.match_id ASC
    """
    relation = con.sql(query)
    return relation.pl()


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


def _build_team_level_features(match_df: pl.DataFrame) -> pl.DataFrame:
    """Derive recent-form features and collapse the dataset back to match level."""
    home_team = _build_side_frame(match_df, side="home")
    away_team = _build_side_frame(match_df, side="away")

    home_features = _recent_form(home_team)
    away_features = _recent_form(away_team)

    matched = home_features.join(
        away_features,
        on="match_id",
        how="inner",
        suffix="_away",
    )

    feature_df = matched.select(
        [
            (
                pl.col("team_world_cup_probability_elo_before")
                - pl.col("team_world_cup_probability_elo_before_away")
            ).alias("world_cup_probability_elo_diff"),
            (
                pl.col("team_world_football_elo_ratings")
                - pl.col("team_world_football_elo_ratings_away")
            ).alias("world_football_elo_ratings_diff"),
            (pl.col("team_market_value_eur") - pl.col("team_market_value_eur_away")).alias(
                "market_value_diff"
            ),
            (pl.col("recent_form") - pl.col("recent_form_away")).alias("recent_form_diff"),
            pl.col("target"),
        ]
    )

    return feature_df


def _build_side_frame(match_df: pl.DataFrame, *, side: str) -> pl.DataFrame:
    """Convert the match table into one row per team per match."""
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")

    if side == "home":
        return match_df.select(
            [
                pl.col("match_id"),
                pl.col("match_date"),
                pl.lit("home").alias("side"),
                pl.col("home_team_id").alias("team"),
                pl.col("away_team_id").alias("opponent"),
                pl.col("home_world_cup_probability_elo_before").alias(
                    "team_world_cup_probability_elo_before"
                ),
                pl.col("away_world_cup_probability_elo_before").alias(
                    "opponent_world_cup_probability_elo_before"
                ),
                pl.col("home_world_football_elo_ratings").alias("team_world_football_elo_ratings"),
                pl.col("away_world_football_elo_ratings").alias(
                    "opponent_world_football_elo_ratings"
                ),
                pl.col("home_market_value_eur").alias("team_market_value_eur"),
                pl.col("away_market_value_eur").alias("opponent_market_value_eur"),
                pl.col("home_team_score").alias("goals_for"),
                pl.col("away_team_score").alias("goals_against"),
                pl.col("home_team_score").alias("target"),
            ]
        )

    return match_df.select(
        [
            pl.col("match_id"),
            pl.col("match_date"),
            pl.lit("away").alias("side"),
            pl.col("away_team_id").alias("team"),
            pl.col("home_team_id").alias("opponent"),
            pl.col("away_world_cup_probability_elo_before").alias(
                "team_world_cup_probability_elo_before"
            ),
            pl.col("home_world_cup_probability_elo_before").alias(
                "opponent_world_cup_probability_elo_before"
            ),
            pl.col("away_world_football_elo_ratings").alias("team_world_football_elo_ratings"),
            pl.col("home_world_football_elo_ratings").alias("opponent_world_football_elo_ratings"),
            pl.col("away_market_value_eur").alias("team_market_value_eur"),
            pl.col("home_market_value_eur").alias("opponent_market_value_eur"),
            pl.col("away_team_score").alias("goals_for"),
            pl.col("home_team_score").alias("goals_against"),
            pl.col("away_team_score").alias("target"),
        ]
    )


def _recent_form(team_df: pl.DataFrame) -> pl.DataFrame:
    """Compute recent form using rolling means over the team partition.

    The current match is excluded from the window by shifting one row before the
    rolling average is computed.
    """
    sorted_df = team_df.sort(["team", "match_date", "match_id"])

    enriched = sorted_df.with_columns(
        [
            pl.col("goals_for")
            .shift(1)
            .rolling_mean(window_size=5, min_periods=1)
            .over("team")
            .fill_null(0.0)
            .alias("goals_for_last5"),
            pl.col("goals_against")
            .shift(1)
            .rolling_mean(window_size=5, min_periods=1)
            .over("team")
            .fill_null(0.0)
            .alias("goals_against_last5"),
        ]
    ).with_columns((pl.col("goals_for_last5") - pl.col("goals_against_last5")).alias("recent_form"))

    return enriched.select(
        [
            pl.col("match_id"),
            pl.col("team_world_cup_probability_elo_before"),
            pl.col("opponent_world_cup_probability_elo_before"),
            pl.col("team_world_football_elo_ratings"),
            pl.col("opponent_world_football_elo_ratings"),
            pl.col("team_market_value_eur"),
            pl.col("opponent_market_value_eur"),
            pl.col("recent_form"),
            pl.col("target"),
        ]
    )


def main() -> int:
    """CLI entrypoint."""
    frame = build_feature_frame()
    LOGGER.info("Generated feature frame with shape %s", frame.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
