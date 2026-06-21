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
    has_fifa_world_ranking = _fifa_world_ranking_tables_available(con)
    has_squad_attributes = _squad_attributes_table_available(con)
    home_team_key = "regexp_replace(lower(m.home_team_id), '[^a-z0-9]+', '', 'g')"
    away_team_key = "regexp_replace(lower(m.away_team_id), '[^a-z0-9]+', '', 'g')"
    squad_attributes_ctes = (
        """
        WITH latest_squad_attributes AS (
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
        )
        """
        if has_squad_attributes
        else ""
    )
    home_squad_attribute_selects = (
        """
            COALESCE(hsa.avg_overall, sad.avg_overall) AS home_avg_overall,
            COALESCE(hsa.avg_pace, sad.avg_pace) AS home_avg_pace,
            COALESCE(hsa.avg_stamina, sad.avg_stamina) AS home_avg_stamina,
            COALESCE(hsa.squad_depth_proxy, sad.squad_depth_proxy) AS home_squad_depth_proxy,
        """
        if has_squad_attributes
        else """
            0.0 AS home_avg_overall,
            0.0 AS home_avg_pace,
            0.0 AS home_avg_stamina,
            0.0 AS home_squad_depth_proxy,
        """
    )
    away_squad_attribute_selects = (
        """
            COALESCE(asa.avg_overall, sad.avg_overall) AS away_avg_overall,
            COALESCE(asa.avg_pace, sad.avg_pace) AS away_avg_pace,
            COALESCE(asa.avg_stamina, sad.avg_stamina) AS away_avg_stamina,
            COALESCE(asa.squad_depth_proxy, sad.squad_depth_proxy) AS away_squad_depth_proxy,
        """
        if has_squad_attributes
        else """
            0.0 AS away_avg_overall,
            0.0 AS away_avg_pace,
            0.0 AS away_avg_stamina,
            0.0 AS away_squad_depth_proxy,
        """
    )
    squad_attribute_joins = (
        f"""
        CROSS JOIN squad_attribute_defaults AS sad
        LEFT JOIN latest_squad_attributes AS hsa
            ON lower(hsa.team_id) = lower(m.home_team_id)
            OR regexp_replace(lower(hsa.team_id), '[^a-z0-9]+', '', 'g') = {home_team_key}
        LEFT JOIN latest_squad_attributes AS asa
            ON lower(asa.team_id) = lower(m.away_team_id)
            OR regexp_replace(lower(asa.team_id), '[^a-z0-9]+', '', 'g') = {away_team_key}
        """
        if has_squad_attributes
        else ""
    )
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
    home_fifa_points_expr = (
        "COALESCE(hfr.ranking_points, e.home_rating_before)"
        if has_fifa_world_ranking
        else "e.home_rating_before"
    )
    away_fifa_points_expr = (
        "COALESCE(afr.ranking_points, e.away_rating_before)"
        if has_fifa_world_ranking
        else "e.away_rating_before"
    )
    home_fifa_rank_expr = (
        "COALESCE(CAST(hfr.fifa_rank AS DOUBLE), 0.0)" if has_fifa_world_ranking else "0.0"
    )
    away_fifa_rank_expr = (
        "COALESCE(CAST(afr.fifa_rank AS DOUBLE), 0.0)" if has_fifa_world_ranking else "0.0"
    )
    fifa_world_ranking_joins = (
        f"""
        LEFT JOIN d_fifa_world_ranking_team_aliases AS hfa
            ON hfa.team_alias_key = lower(m.home_team_id)
            OR hfa.normalized_team_alias = {home_team_key}
        LEFT JOIN d_fifa_world_ranking AS hfr
            ON hfr.fifa_country_code = hfa.fifa_country_code
        LEFT JOIN d_fifa_world_ranking_team_aliases AS afa
            ON afa.team_alias_key = lower(m.away_team_id)
            OR afa.normalized_team_alias = {away_team_key}
        LEFT JOIN d_fifa_world_ranking AS afr
            ON afr.fifa_country_code = afa.fifa_country_code
        """
        if has_fifa_world_ranking
        else ""
    )

    query = f"""
        {squad_attributes_ctes}
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
            {home_fifa_points_expr} AS home_fifa_world_ranking_points,
            {away_fifa_points_expr} AS away_fifa_world_ranking_points,
            {home_fifa_rank_expr} AS home_fifa_world_ranking_rank,
            {away_fifa_rank_expr} AS away_fifa_world_ranking_rank,
            {home_squad_attribute_selects}
            {away_squad_attribute_selects}
            COALESCE(ht.market_value_eur, 0.0) AS home_market_value_eur,
            COALESCE(awt.market_value_eur, 0.0) AS away_market_value_eur
        FROM f_matches AS m
        INNER JOIN f_elo_history AS e
            ON e.match_id = m.match_id
        {world_football_elo_ratings_joins}
        {fifa_world_ranking_joins}
        {squad_attribute_joins}
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
            (
                pl.col("team_fifa_world_ranking_points")
                - pl.col("team_fifa_world_ranking_points_away")
            ).alias("fifa_world_ranking_points_diff"),
            (
                pl.col("team_fifa_world_ranking_rank_away") - pl.col("team_fifa_world_ranking_rank")
            ).alias("fifa_world_ranking_rank_diff"),
            (pl.col("team_market_value_eur") - pl.col("team_market_value_eur_away")).alias(
                "market_value_diff"
            ),
            (pl.col("team_avg_overall") - pl.col("team_avg_overall_away")).alias(
                "avg_overall_diff"
            ),
            (pl.col("team_avg_pace") - pl.col("team_avg_pace_away")).alias("avg_pace_diff"),
            (pl.col("team_avg_stamina") - pl.col("team_avg_stamina_away")).alias(
                "avg_stamina_diff"
            ),
            (pl.col("team_squad_depth_proxy") - pl.col("team_squad_depth_proxy_away")).alias(
                "squad_depth_proxy"
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
                pl.col("home_fifa_world_ranking_points").alias("team_fifa_world_ranking_points"),
                pl.col("away_fifa_world_ranking_points").alias(
                    "opponent_fifa_world_ranking_points"
                ),
                pl.col("home_fifa_world_ranking_rank").alias("team_fifa_world_ranking_rank"),
                pl.col("away_fifa_world_ranking_rank").alias("opponent_fifa_world_ranking_rank"),
                pl.col("home_market_value_eur").alias("team_market_value_eur"),
                pl.col("away_market_value_eur").alias("opponent_market_value_eur"),
                pl.col("home_avg_overall").alias("team_avg_overall"),
                pl.col("away_avg_overall").alias("opponent_avg_overall"),
                pl.col("home_avg_pace").alias("team_avg_pace"),
                pl.col("away_avg_pace").alias("opponent_avg_pace"),
                pl.col("home_avg_stamina").alias("team_avg_stamina"),
                pl.col("away_avg_stamina").alias("opponent_avg_stamina"),
                pl.col("home_squad_depth_proxy").alias("team_squad_depth_proxy"),
                pl.col("away_squad_depth_proxy").alias("opponent_squad_depth_proxy"),
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
            pl.col("away_fifa_world_ranking_points").alias("team_fifa_world_ranking_points"),
            pl.col("home_fifa_world_ranking_points").alias("opponent_fifa_world_ranking_points"),
            pl.col("away_fifa_world_ranking_rank").alias("team_fifa_world_ranking_rank"),
            pl.col("home_fifa_world_ranking_rank").alias("opponent_fifa_world_ranking_rank"),
            pl.col("away_market_value_eur").alias("team_market_value_eur"),
            pl.col("home_market_value_eur").alias("opponent_market_value_eur"),
            pl.col("away_avg_overall").alias("team_avg_overall"),
            pl.col("home_avg_overall").alias("opponent_avg_overall"),
            pl.col("away_avg_pace").alias("team_avg_pace"),
            pl.col("home_avg_pace").alias("opponent_avg_pace"),
            pl.col("away_avg_stamina").alias("team_avg_stamina"),
            pl.col("home_avg_stamina").alias("opponent_avg_stamina"),
            pl.col("away_squad_depth_proxy").alias("team_squad_depth_proxy"),
            pl.col("home_squad_depth_proxy").alias("opponent_squad_depth_proxy"),
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
            pl.col("team_fifa_world_ranking_points"),
            pl.col("opponent_fifa_world_ranking_points"),
            pl.col("team_fifa_world_ranking_rank"),
            pl.col("opponent_fifa_world_ranking_rank"),
            pl.col("team_market_value_eur"),
            pl.col("opponent_market_value_eur"),
            pl.col("team_avg_overall"),
            pl.col("opponent_avg_overall"),
            pl.col("team_avg_pace"),
            pl.col("opponent_avg_pace"),
            pl.col("team_avg_stamina"),
            pl.col("opponent_avg_stamina"),
            pl.col("team_squad_depth_proxy"),
            pl.col("opponent_squad_depth_proxy"),
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
