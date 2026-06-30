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

try:
    from .competition_filters import current_world_cup_exclusion_sql
    from .settings import DB_PATH
    from .sql_loader import render_sql_template
except ImportError:  # pragma: no cover - supports direct script execution.
    from competition_filters import current_world_cup_exclusion_sql
    from settings import DB_PATH
    from sql_loader import render_sql_template

LOGGER = logging.getLogger(__name__)

SQUAD_POSITION_FEATURES: tuple[str, ...] = (
    "avg_goalkeeper_overall",
    "avg_defense_overall",
    "avg_midfield_overall",
    "avg_attack_overall",
)
MATCH_STAT_FEATURES: tuple[str, ...] = (
    "xg",
    "possession_pct",
    "shots",
    "shots_on_target",
    "corners",
    "yellow_cards",
    "red_cards",
)
ROLLING_FEATURE_DEFAULTS: dict[str, float] = {
    "recent_xg_for": 0.0,
    "recent_xg_against": 0.0,
    "recent_possession_pct": 0.0,
    "recent_shots_for": 0.0,
    "recent_shots_against": 0.0,
    "recent_shots_on_target_for": 0.0,
    "recent_shots_on_target_against": 0.0,
    "recent_corners_for": 0.0,
    "recent_corners_against": 0.0,
    "recent_yellow_cards": 0.0,
    "recent_red_cards": 0.0,
    "opponent_adjusted_recent_form": 0.0,
    "recent_official_form": 0.0,
    "recent_friendly_form": 0.0,
}


def build_feature_frame(
    db_path: Path = DB_PATH,
    *,
    include_current_world_cup: bool = False,
    include_metadata: bool = False,
) -> pl.DataFrame:
    """Build the final training dataframe.

    Args:
        db_path: Path to the DuckDB warehouse.
        include_current_world_cup: Whether to include scored 2026 World Cup rows.
            Keep this false for model training.
        include_metadata: Whether to keep match metadata needed for temporal
            validation and holdout evaluation.

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
        base_df = _load_consolidated_matches(
            con,
            include_current_world_cup=include_current_world_cup,
        )

    feature_df = _build_team_level_features(base_df, include_metadata=include_metadata)
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


def _load_consolidated_matches(
    con: duckdb.DuckDBPyConnection,
    *,
    include_current_world_cup: bool = False,
) -> pl.DataFrame:
    """Read the consolidated match-level relation from DuckDB into Polars."""
    has_world_football_elo_ratings = _world_football_elo_ratings_tables_available(con)
    has_fifa_world_ranking = _fifa_world_ranking_tables_available(con)
    has_squad_attributes = _squad_attributes_table_available(con)
    has_match_stats = _match_stats_table_available(con)
    match_stats_columns = _table_columns(con, "f_match_stats") if has_match_stats else set()
    has_world_cup_prior_history = _world_cup_prior_history_table_available(con)
    has_world_cup_prior_discipline = _world_cup_prior_discipline_table_available(con)
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
                avg_goalkeeper_overall,
                avg_defense_overall,
                avg_midfield_overall,
                avg_attack_overall,
                CAST(sampled_player_count AS DOUBLE) AS squad_depth_proxy
            FROM (
                SELECT
                    team_id,
                    avg_overall,
                    avg_pace,
                    avg_stamina,
                    COALESCE(avg_goalkeeper_overall, avg_overall) AS avg_goalkeeper_overall,
                    COALESCE(avg_defense_overall, avg_overall) AS avg_defense_overall,
                    COALESCE(avg_midfield_overall, avg_overall) AS avg_midfield_overall,
                    COALESCE(avg_attack_overall, avg_overall) AS avg_attack_overall,
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
                COALESCE(AVG(avg_goalkeeper_overall), 0.0) AS avg_goalkeeper_overall,
                COALESCE(AVG(avg_defense_overall), 0.0) AS avg_defense_overall,
                COALESCE(AVG(avg_midfield_overall), 0.0) AS avg_midfield_overall,
                COALESCE(AVG(avg_attack_overall), 0.0) AS avg_attack_overall,
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
            COALESCE(hsa.avg_goalkeeper_overall, sad.avg_goalkeeper_overall)
                AS home_avg_goalkeeper_overall,
            COALESCE(hsa.avg_defense_overall, sad.avg_defense_overall)
                AS home_avg_defense_overall,
            COALESCE(hsa.avg_midfield_overall, sad.avg_midfield_overall)
                AS home_avg_midfield_overall,
            COALESCE(hsa.avg_attack_overall, sad.avg_attack_overall)
                AS home_avg_attack_overall,
            COALESCE(hsa.squad_depth_proxy, sad.squad_depth_proxy) AS home_squad_depth_proxy,
        """
        if has_squad_attributes
        else """
            0.0 AS home_avg_overall,
            0.0 AS home_avg_pace,
            0.0 AS home_avg_stamina,
            0.0 AS home_avg_goalkeeper_overall,
            0.0 AS home_avg_defense_overall,
            0.0 AS home_avg_midfield_overall,
            0.0 AS home_avg_attack_overall,
            0.0 AS home_squad_depth_proxy,
        """
    )
    away_squad_attribute_selects = (
        """
            COALESCE(asa.avg_overall, sad.avg_overall) AS away_avg_overall,
            COALESCE(asa.avg_pace, sad.avg_pace) AS away_avg_pace,
            COALESCE(asa.avg_stamina, sad.avg_stamina) AS away_avg_stamina,
            COALESCE(asa.avg_goalkeeper_overall, sad.avg_goalkeeper_overall)
                AS away_avg_goalkeeper_overall,
            COALESCE(asa.avg_defense_overall, sad.avg_defense_overall)
                AS away_avg_defense_overall,
            COALESCE(asa.avg_midfield_overall, sad.avg_midfield_overall)
                AS away_avg_midfield_overall,
            COALESCE(asa.avg_attack_overall, sad.avg_attack_overall)
                AS away_avg_attack_overall,
            COALESCE(asa.squad_depth_proxy, sad.squad_depth_proxy) AS away_squad_depth_proxy,
        """
        if has_squad_attributes
        else """
            0.0 AS away_avg_overall,
            0.0 AS away_avg_pace,
            0.0 AS away_avg_stamina,
            0.0 AS away_avg_goalkeeper_overall,
            0.0 AS away_avg_defense_overall,
            0.0 AS away_avg_midfield_overall,
            0.0 AS away_avg_attack_overall,
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
    home_prior_world_cup_appearances_expr = (
        "COALESCE(CAST(hwch.prior_world_cup_appearances AS DOUBLE), 0.0)"
        if has_world_cup_prior_history
        else "0.0"
    )
    away_prior_world_cup_appearances_expr = (
        "COALESCE(CAST(awch.prior_world_cup_appearances AS DOUBLE), 0.0)"
        if has_world_cup_prior_history
        else "0.0"
    )
    home_prior_world_cup_points_per_match_expr = (
        "COALESCE(hwch.prior_world_cup_points_per_match, 0.0)"
        if has_world_cup_prior_history
        else "0.0"
    )
    away_prior_world_cup_points_per_match_expr = (
        "COALESCE(awch.prior_world_cup_points_per_match, 0.0)"
        if has_world_cup_prior_history
        else "0.0"
    )
    home_prior_world_cup_goal_diff_per_match_expr = (
        "COALESCE(hwch.prior_world_cup_goal_diff_per_match, 0.0)"
        if has_world_cup_prior_history
        else "0.0"
    )
    away_prior_world_cup_goal_diff_per_match_expr = (
        "COALESCE(awch.prior_world_cup_goal_diff_per_match, 0.0)"
        if has_world_cup_prior_history
        else "0.0"
    )
    world_cup_prior_history_joins = (
        f"""
        LEFT JOIN d_world_cup_prior_team_history AS hwch
            ON hwch.as_of_year = CAST(EXTRACT(year FROM m.match_date) AS INTEGER)
            AND (
                lower(hwch.team_name) = lower(m.home_team_id)
                OR hwch.normalized_team_name = {home_team_key}
                OR lower(hwch.team_code) = lower(m.home_team_id)
                OR regexp_replace(lower(COALESCE(hwch.team_code, '')), '[^a-z0-9]+', '', 'g')
                    = {home_team_key}
            )
        LEFT JOIN d_world_cup_prior_team_history AS awch
            ON awch.as_of_year = CAST(EXTRACT(year FROM m.match_date) AS INTEGER)
            AND (
                lower(awch.team_name) = lower(m.away_team_id)
                OR awch.normalized_team_name = {away_team_key}
                OR lower(awch.team_code) = lower(m.away_team_id)
                OR regexp_replace(lower(COALESCE(awch.team_code, '')), '[^a-z0-9]+', '', 'g')
                    = {away_team_key}
            )
        """
        if has_world_cup_prior_history
        else ""
    )
    home_prior_world_cup_yellow_cards_per_match_expr = (
        "COALESCE(hwdh.prior_world_cup_yellow_cards_per_match, 0.0)"
        if has_world_cup_prior_discipline
        else "0.0"
    )
    away_prior_world_cup_yellow_cards_per_match_expr = (
        "COALESCE(awdh.prior_world_cup_yellow_cards_per_match, 0.0)"
        if has_world_cup_prior_discipline
        else "0.0"
    )
    home_prior_world_cup_sending_offs_per_match_expr = (
        "COALESCE(hwdh.prior_world_cup_sending_offs_per_match, 0.0)"
        if has_world_cup_prior_discipline
        else "0.0"
    )
    away_prior_world_cup_sending_offs_per_match_expr = (
        "COALESCE(awdh.prior_world_cup_sending_offs_per_match, 0.0)"
        if has_world_cup_prior_discipline
        else "0.0"
    )
    home_prior_world_cup_fair_play_penalty_per_match_expr = (
        "COALESCE(hwdh.prior_world_cup_fair_play_penalty_per_match, 0.0)"
        if has_world_cup_prior_discipline
        else "0.0"
    )
    away_prior_world_cup_fair_play_penalty_per_match_expr = (
        "COALESCE(awdh.prior_world_cup_fair_play_penalty_per_match, 0.0)"
        if has_world_cup_prior_discipline
        else "0.0"
    )
    world_cup_prior_discipline_joins = (
        f"""
        LEFT JOIN d_world_cup_prior_discipline_history AS hwdh
            ON hwdh.as_of_year = CAST(EXTRACT(year FROM m.match_date) AS INTEGER)
            AND (
                lower(hwdh.team_name) = lower(m.home_team_id)
                OR hwdh.normalized_team_name = {home_team_key}
                OR lower(hwdh.team_code) = lower(m.home_team_id)
                OR regexp_replace(lower(COALESCE(hwdh.team_code, '')), '[^a-z0-9]+', '', 'g')
                    = {home_team_key}
            )
        LEFT JOIN d_world_cup_prior_discipline_history AS awdh
            ON awdh.as_of_year = CAST(EXTRACT(year FROM m.match_date) AS INTEGER)
            AND (
                lower(awdh.team_name) = lower(m.away_team_id)
                OR awdh.normalized_team_name = {away_team_key}
                OR lower(awdh.team_code) = lower(m.away_team_id)
                OR regexp_replace(lower(COALESCE(awdh.team_code, '')), '[^a-z0-9]+', '', 'g')
                    = {away_team_key}
            )
        """
        if has_world_cup_prior_discipline
        else ""
    )
    current_world_cup_exclusion = current_world_cup_exclusion_sql(
        date_expr="m.match_date",
        competition_expr="m.competition",
    )
    current_world_cup_flag = f"NOT ({current_world_cup_exclusion})"
    training_scope_filter = "TRUE" if include_current_world_cup else current_world_cup_exclusion
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
    home_match_stats_selects = _match_stats_selects(
        table_alias="hms",
        side="home",
        available_columns=match_stats_columns,
        has_match_stats=has_match_stats,
    )
    away_match_stats_selects = _match_stats_selects(
        table_alias="ams",
        side="away",
        available_columns=match_stats_columns,
        has_match_stats=has_match_stats,
    )
    match_stats_joins = (
        f"""
        LEFT JOIN f_match_stats AS hms
            ON hms.source = 'fbref'
            AND hms.match_id = m.match_id
            AND (
                lower(hms.team_id) = lower(m.home_team_id)
                OR regexp_replace(lower(hms.team_id), '[^a-z0-9]+', '', 'g') = {home_team_key}
            )
        LEFT JOIN f_match_stats AS ams
            ON ams.source = 'fbref'
            AND ams.match_id = m.match_id
            AND (
                lower(ams.team_id) = lower(m.away_team_id)
                OR regexp_replace(lower(ams.team_id), '[^a-z0-9]+', '', 'g') = {away_team_key}
            )
        """
        if has_match_stats
        else ""
    )

    query = render_sql_template(
        "feature_pipeline/consolidated_matches.sql.j2",
        squad_attributes_ctes=squad_attributes_ctes,
        home_world_football_elo_ratings_expr=home_world_football_elo_ratings_expr,
        away_world_football_elo_ratings_expr=away_world_football_elo_ratings_expr,
        home_fifa_points_expr=home_fifa_points_expr,
        away_fifa_points_expr=away_fifa_points_expr,
        home_fifa_rank_expr=home_fifa_rank_expr,
        away_fifa_rank_expr=away_fifa_rank_expr,
        home_prior_world_cup_appearances_expr=home_prior_world_cup_appearances_expr,
        away_prior_world_cup_appearances_expr=away_prior_world_cup_appearances_expr,
        home_prior_world_cup_points_per_match_expr=home_prior_world_cup_points_per_match_expr,
        away_prior_world_cup_points_per_match_expr=away_prior_world_cup_points_per_match_expr,
        home_prior_world_cup_goal_diff_per_match_expr=(
            home_prior_world_cup_goal_diff_per_match_expr
        ),
        away_prior_world_cup_goal_diff_per_match_expr=(
            away_prior_world_cup_goal_diff_per_match_expr
        ),
        home_prior_world_cup_yellow_cards_per_match_expr=(
            home_prior_world_cup_yellow_cards_per_match_expr
        ),
        away_prior_world_cup_yellow_cards_per_match_expr=(
            away_prior_world_cup_yellow_cards_per_match_expr
        ),
        home_prior_world_cup_sending_offs_per_match_expr=(
            home_prior_world_cup_sending_offs_per_match_expr
        ),
        away_prior_world_cup_sending_offs_per_match_expr=(
            away_prior_world_cup_sending_offs_per_match_expr
        ),
        home_prior_world_cup_fair_play_penalty_per_match_expr=(
            home_prior_world_cup_fair_play_penalty_per_match_expr
        ),
        away_prior_world_cup_fair_play_penalty_per_match_expr=(
            away_prior_world_cup_fair_play_penalty_per_match_expr
        ),
        home_squad_attribute_selects=home_squad_attribute_selects,
        away_squad_attribute_selects=away_squad_attribute_selects,
        home_match_stats_selects=home_match_stats_selects,
        away_match_stats_selects=away_match_stats_selects,
        world_football_elo_ratings_joins=world_football_elo_ratings_joins,
        fifa_world_ranking_joins=fifa_world_ranking_joins,
        world_cup_prior_history_joins=world_cup_prior_history_joins,
        world_cup_prior_discipline_joins=world_cup_prior_discipline_joins,
        squad_attribute_joins=squad_attribute_joins,
        match_stats_joins=match_stats_joins,
        current_world_cup_flag=current_world_cup_flag,
        current_world_cup_exclusion=training_scope_filter,
    )
    relation = con.sql(query)
    return relation.pl()


def _match_stats_selects(
    *,
    table_alias: str,
    side: str,
    available_columns: set[str],
    has_match_stats: bool,
) -> str:
    """Return nullable FBref stat projections for a home or away side."""
    xg_expr = _match_stats_column_expr(
        table_alias,
        "xg",
        available_columns=available_columns,
        has_match_stats=has_match_stats,
    )
    projections = [f"COALESCE({xg_expr}, m.{side}_xg) AS {side}_xg"]
    for column in MATCH_STAT_FEATURES:
        if column == "xg":
            continue
        column_expr = _match_stats_column_expr(
            table_alias,
            column,
            available_columns=available_columns,
            has_match_stats=has_match_stats,
        )
        projections.append(f"{column_expr} AS {side}_{column}")
    return ",\n    ".join(projections) + ","


def _match_stats_column_expr(
    table_alias: str,
    column: str,
    *,
    available_columns: set[str],
    has_match_stats: bool,
) -> str:
    if has_match_stats and column in available_columns:
        return f"{table_alias}.{column}"
    return "NULL::DOUBLE"


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    if not _table_exists(con, table_name):
        return set()
    rows = con.execute(f"DESCRIBE {table_name}").fetchall()
    return {str(row[0]).casefold() for row in rows}


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


def _world_football_elo_ratings_tables_available(con: duckdb.DuckDBPyConnection) -> bool:
    tables = ("d_world_football_elo_ratings", "d_world_football_elo_team_aliases")
    return all(_table_exists(con, table) for table in tables)


def _fifa_world_ranking_tables_available(con: duckdb.DuckDBPyConnection) -> bool:
    tables = ("d_fifa_world_ranking", "d_fifa_world_ranking_team_aliases")
    return all(_table_exists(con, table) for table in tables)


def _squad_attributes_table_available(con: duckdb.DuckDBPyConnection) -> bool:
    return _table_exists(con, "d_squad_attributes")


def _match_stats_table_available(con: duckdb.DuckDBPyConnection) -> bool:
    return _table_exists(con, "f_match_stats")


def _world_cup_prior_history_table_available(con: duckdb.DuckDBPyConnection) -> bool:
    return _table_exists(con, "d_world_cup_prior_team_history")


def _world_cup_prior_discipline_table_available(con: duckdb.DuckDBPyConnection) -> bool:
    return _table_exists(con, "d_world_cup_prior_discipline_history")


def _build_team_level_features(
    match_df: pl.DataFrame,
    *,
    include_metadata: bool = False,
) -> pl.DataFrame:
    """Derive recent-form features and collapse the dataset back to match level."""
    if match_df.height == 0:
        return _empty_feature_frame(include_metadata=include_metadata)

    match_df = _ensure_prior_world_cup_columns(match_df)
    match_df = _ensure_prior_world_cup_discipline_columns(match_df)
    match_df = _ensure_squad_position_columns(match_df)
    match_df = _ensure_match_stat_columns(match_df)
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

    feature_columns = [
        (
            pl.col("team_world_cup_probability_elo_before")
            - pl.col("team_world_cup_probability_elo_before_away")
        ).alias("world_cup_probability_elo_diff"),
        (
            pl.col("team_world_football_elo_ratings")
            - pl.col("team_world_football_elo_ratings_away")
        ).alias("world_football_elo_ratings_diff"),
        (
            pl.col("team_fifa_world_ranking_points") - pl.col("team_fifa_world_ranking_points_away")
        ).alias("fifa_world_ranking_points_diff"),
        (
            pl.col("team_fifa_world_ranking_rank_away") - pl.col("team_fifa_world_ranking_rank")
        ).alias("fifa_world_ranking_rank_diff"),
        (
            pl.col("team_prior_world_cup_appearances")
            - pl.col("team_prior_world_cup_appearances_away")
        ).alias("prior_world_cup_appearances_diff"),
        (
            pl.col("team_prior_world_cup_points_per_match")
            - pl.col("team_prior_world_cup_points_per_match_away")
        ).alias("prior_world_cup_points_per_match_diff"),
        (
            pl.col("team_prior_world_cup_goal_diff_per_match")
            - pl.col("team_prior_world_cup_goal_diff_per_match_away")
        ).alias("prior_world_cup_goal_diff_per_match_diff"),
        (
            pl.col("team_prior_world_cup_yellow_cards_per_match")
            - pl.col("team_prior_world_cup_yellow_cards_per_match_away")
        ).alias("prior_world_cup_yellow_cards_per_match_diff"),
        (
            pl.col("team_prior_world_cup_sending_offs_per_match")
            - pl.col("team_prior_world_cup_sending_offs_per_match_away")
        ).alias("prior_world_cup_sending_offs_per_match_diff"),
        (
            pl.col("team_prior_world_cup_fair_play_penalty_per_match")
            - pl.col("team_prior_world_cup_fair_play_penalty_per_match_away")
        ).alias("prior_world_cup_fair_play_penalty_per_match_diff"),
        (pl.col("team_market_value_eur") - pl.col("team_market_value_eur_away")).alias(
            "market_value_diff"
        ),
        (pl.col("team_avg_overall") - pl.col("team_avg_overall_away")).alias("avg_overall_diff"),
        (pl.col("team_avg_pace") - pl.col("team_avg_pace_away")).alias("avg_pace_diff"),
        (pl.col("team_avg_stamina") - pl.col("team_avg_stamina_away")).alias("avg_stamina_diff"),
        (pl.col("team_avg_goalkeeper_overall") - pl.col("team_avg_goalkeeper_overall_away")).alias(
            "avg_goalkeeper_overall_diff"
        ),
        (pl.col("team_avg_defense_overall") - pl.col("team_avg_defense_overall_away")).alias(
            "avg_defense_overall_diff"
        ),
        (pl.col("team_avg_midfield_overall") - pl.col("team_avg_midfield_overall_away")).alias(
            "avg_midfield_overall_diff"
        ),
        (pl.col("team_avg_attack_overall") - pl.col("team_avg_attack_overall_away")).alias(
            "avg_attack_overall_diff"
        ),
        (pl.col("team_squad_depth_proxy") - pl.col("team_squad_depth_proxy_away")).alias(
            "squad_depth_proxy"
        ),
        (pl.col("recent_form") - pl.col("recent_form_away")).alias("recent_form_diff"),
        (pl.col("recent_xg_for") - pl.col("recent_xg_for_away")).alias("recent_xg_for_diff"),
        (pl.col("recent_xg_against") - pl.col("recent_xg_against_away")).alias(
            "recent_xg_against_diff"
        ),
        (pl.col("recent_possession_pct") - pl.col("recent_possession_pct_away")).alias(
            "recent_possession_pct_diff"
        ),
        (pl.col("recent_shots_for") - pl.col("recent_shots_for_away")).alias(
            "recent_shots_for_diff"
        ),
        (pl.col("recent_shots_against") - pl.col("recent_shots_against_away")).alias(
            "recent_shots_against_diff"
        ),
        (pl.col("recent_shots_on_target_for") - pl.col("recent_shots_on_target_for_away")).alias(
            "recent_shots_on_target_for_diff"
        ),
        (
            pl.col("recent_shots_on_target_against") - pl.col("recent_shots_on_target_against_away")
        ).alias("recent_shots_on_target_against_diff"),
        (pl.col("recent_corners_for") - pl.col("recent_corners_for_away")).alias(
            "recent_corners_for_diff"
        ),
        (pl.col("recent_corners_against") - pl.col("recent_corners_against_away")).alias(
            "recent_corners_against_diff"
        ),
        (pl.col("recent_yellow_cards") - pl.col("recent_yellow_cards_away")).alias(
            "recent_yellow_cards_diff"
        ),
        (pl.col("recent_red_cards") - pl.col("recent_red_cards_away")).alias(
            "recent_red_cards_diff"
        ),
        (
            pl.col("opponent_adjusted_recent_form") - pl.col("opponent_adjusted_recent_form_away")
        ).alias("opponent_adjusted_recent_form_diff"),
        (pl.col("recent_official_form") - pl.col("recent_official_form_away")).alias(
            "recent_official_form_diff"
        ),
        (pl.col("recent_friendly_form") - pl.col("recent_friendly_form_away")).alias(
            "recent_friendly_form_diff"
        ),
        pl.col("is_friendly_match"),
        pl.col("target"),
    ]
    if include_metadata:
        feature_columns = [
            pl.col("match_id"),
            pl.col("match_date"),
            pl.col("competition"),
            pl.col("is_current_world_cup"),
            pl.col("target").alias("home_goals"),
            pl.col("target_away").alias("away_goals"),
            *feature_columns,
        ]

    feature_df = matched.select(feature_columns)

    return feature_df


def _empty_feature_frame(*, include_metadata: bool) -> pl.DataFrame:
    """Return an empty frame with the public feature schema."""
    schema: dict[str, pl.DataType] = {}
    if include_metadata:
        schema.update(
            {
                "match_id": pl.Utf8,
                "match_date": pl.Date,
                "competition": pl.Utf8,
                "is_current_world_cup": pl.Boolean,
                "home_goals": pl.Float64,
                "away_goals": pl.Float64,
            }
        )
    schema.update(
        {
            "world_cup_probability_elo_diff": pl.Float64,
            "world_football_elo_ratings_diff": pl.Float64,
            "fifa_world_ranking_points_diff": pl.Float64,
            "fifa_world_ranking_rank_diff": pl.Float64,
            "prior_world_cup_appearances_diff": pl.Float64,
            "prior_world_cup_points_per_match_diff": pl.Float64,
            "prior_world_cup_goal_diff_per_match_diff": pl.Float64,
            "prior_world_cup_yellow_cards_per_match_diff": pl.Float64,
            "prior_world_cup_sending_offs_per_match_diff": pl.Float64,
            "prior_world_cup_fair_play_penalty_per_match_diff": pl.Float64,
            "market_value_diff": pl.Float64,
            "avg_overall_diff": pl.Float64,
            "avg_pace_diff": pl.Float64,
            "avg_stamina_diff": pl.Float64,
            "avg_goalkeeper_overall_diff": pl.Float64,
            "avg_defense_overall_diff": pl.Float64,
            "avg_midfield_overall_diff": pl.Float64,
            "avg_attack_overall_diff": pl.Float64,
            "squad_depth_proxy": pl.Float64,
            "recent_form_diff": pl.Float64,
            "recent_xg_for_diff": pl.Float64,
            "recent_xg_against_diff": pl.Float64,
            "recent_possession_pct_diff": pl.Float64,
            "recent_shots_for_diff": pl.Float64,
            "recent_shots_against_diff": pl.Float64,
            "recent_shots_on_target_for_diff": pl.Float64,
            "recent_shots_on_target_against_diff": pl.Float64,
            "recent_corners_for_diff": pl.Float64,
            "recent_corners_against_diff": pl.Float64,
            "recent_yellow_cards_diff": pl.Float64,
            "recent_red_cards_diff": pl.Float64,
            "opponent_adjusted_recent_form_diff": pl.Float64,
            "recent_official_form_diff": pl.Float64,
            "recent_friendly_form_diff": pl.Float64,
            "is_friendly_match": pl.Float64,
            "target": pl.Float64,
        }
    )
    return pl.DataFrame(schema=schema)


def _ensure_prior_world_cup_columns(match_df: pl.DataFrame) -> pl.DataFrame:
    """Backfill optional Fjelstul feature columns for direct unit-test frames."""
    required_columns = (
        "home_prior_world_cup_appearances",
        "away_prior_world_cup_appearances",
        "home_prior_world_cup_points_per_match",
        "away_prior_world_cup_points_per_match",
        "home_prior_world_cup_goal_diff_per_match",
        "away_prior_world_cup_goal_diff_per_match",
    )
    missing_columns = [column for column in required_columns if column not in match_df.columns]
    if not missing_columns:
        return match_df
    return match_df.with_columns(pl.lit(0.0).alias(column) for column in missing_columns)


def _ensure_prior_world_cup_discipline_columns(match_df: pl.DataFrame) -> pl.DataFrame:
    """Backfill optional Fjelstul discipline feature columns for direct unit-test frames."""
    required_columns = (
        "home_prior_world_cup_yellow_cards_per_match",
        "away_prior_world_cup_yellow_cards_per_match",
        "home_prior_world_cup_sending_offs_per_match",
        "away_prior_world_cup_sending_offs_per_match",
        "home_prior_world_cup_fair_play_penalty_per_match",
        "away_prior_world_cup_fair_play_penalty_per_match",
    )
    missing_columns = [column for column in required_columns if column not in match_df.columns]
    if not missing_columns:
        return match_df
    return match_df.with_columns(pl.lit(0.0).alias(column) for column in missing_columns)


def _ensure_squad_position_columns(match_df: pl.DataFrame) -> pl.DataFrame:
    """Backfill optional positional squad attributes for direct unit-test frames."""
    required_columns = tuple(
        f"{side}_{feature}" for side in ("home", "away") for feature in SQUAD_POSITION_FEATURES
    )
    missing_columns = [column for column in required_columns if column not in match_df.columns]
    if not missing_columns:
        return match_df

    fallback_expressions = []
    for column in missing_columns:
        side = column.split("_", 1)[0]
        fallback_column = f"{side}_avg_overall"
        fallback = pl.col(fallback_column) if fallback_column in match_df.columns else pl.lit(0.0)
        fallback_expressions.append(fallback.alias(column))
    return match_df.with_columns(fallback_expressions)


def _ensure_match_stat_columns(match_df: pl.DataFrame) -> pl.DataFrame:
    """Backfill nullable optional match stat columns for direct unit-test frames."""
    required_columns = tuple(
        f"{side}_{feature}" for side in ("home", "away") for feature in MATCH_STAT_FEATURES
    )
    missing_columns = [column for column in required_columns if column not in match_df.columns]
    if not missing_columns:
        return match_df
    return match_df.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(column) for column in missing_columns
    )


def _build_side_frame(match_df: pl.DataFrame, *, side: str) -> pl.DataFrame:
    """Convert the match table into one row per team per match."""
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")

    if side == "home":
        return match_df.select(
            [
                pl.col("match_id"),
                pl.col("match_date"),
                pl.col("competition"),
                pl.col("is_current_world_cup"),
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
                pl.col("home_prior_world_cup_appearances").alias(
                    "team_prior_world_cup_appearances"
                ),
                pl.col("away_prior_world_cup_appearances").alias(
                    "opponent_prior_world_cup_appearances"
                ),
                pl.col("home_prior_world_cup_points_per_match").alias(
                    "team_prior_world_cup_points_per_match"
                ),
                pl.col("away_prior_world_cup_points_per_match").alias(
                    "opponent_prior_world_cup_points_per_match"
                ),
                pl.col("home_prior_world_cup_goal_diff_per_match").alias(
                    "team_prior_world_cup_goal_diff_per_match"
                ),
                pl.col("away_prior_world_cup_goal_diff_per_match").alias(
                    "opponent_prior_world_cup_goal_diff_per_match"
                ),
                pl.col("home_prior_world_cup_yellow_cards_per_match").alias(
                    "team_prior_world_cup_yellow_cards_per_match"
                ),
                pl.col("away_prior_world_cup_yellow_cards_per_match").alias(
                    "opponent_prior_world_cup_yellow_cards_per_match"
                ),
                pl.col("home_prior_world_cup_sending_offs_per_match").alias(
                    "team_prior_world_cup_sending_offs_per_match"
                ),
                pl.col("away_prior_world_cup_sending_offs_per_match").alias(
                    "opponent_prior_world_cup_sending_offs_per_match"
                ),
                pl.col("home_prior_world_cup_fair_play_penalty_per_match").alias(
                    "team_prior_world_cup_fair_play_penalty_per_match"
                ),
                pl.col("away_prior_world_cup_fair_play_penalty_per_match").alias(
                    "opponent_prior_world_cup_fair_play_penalty_per_match"
                ),
                pl.col("home_market_value_eur").alias("team_market_value_eur"),
                pl.col("away_market_value_eur").alias("opponent_market_value_eur"),
                pl.col("home_avg_overall").alias("team_avg_overall"),
                pl.col("away_avg_overall").alias("opponent_avg_overall"),
                pl.col("home_avg_pace").alias("team_avg_pace"),
                pl.col("away_avg_pace").alias("opponent_avg_pace"),
                pl.col("home_avg_stamina").alias("team_avg_stamina"),
                pl.col("away_avg_stamina").alias("opponent_avg_stamina"),
                pl.col("home_avg_goalkeeper_overall").alias("team_avg_goalkeeper_overall"),
                pl.col("away_avg_goalkeeper_overall").alias("opponent_avg_goalkeeper_overall"),
                pl.col("home_avg_defense_overall").alias("team_avg_defense_overall"),
                pl.col("away_avg_defense_overall").alias("opponent_avg_defense_overall"),
                pl.col("home_avg_midfield_overall").alias("team_avg_midfield_overall"),
                pl.col("away_avg_midfield_overall").alias("opponent_avg_midfield_overall"),
                pl.col("home_avg_attack_overall").alias("team_avg_attack_overall"),
                pl.col("away_avg_attack_overall").alias("opponent_avg_attack_overall"),
                pl.col("home_squad_depth_proxy").alias("team_squad_depth_proxy"),
                pl.col("away_squad_depth_proxy").alias("opponent_squad_depth_proxy"),
                pl.col("home_xg").alias("xg_for"),
                pl.col("away_xg").alias("xg_against"),
                pl.col("home_possession_pct").alias("possession_pct"),
                pl.col("home_shots").alias("shots_for"),
                pl.col("away_shots").alias("shots_against"),
                pl.col("home_shots_on_target").alias("shots_on_target_for"),
                pl.col("away_shots_on_target").alias("shots_on_target_against"),
                pl.col("home_corners").alias("corners_for"),
                pl.col("away_corners").alias("corners_against"),
                pl.col("home_yellow_cards").alias("yellow_cards"),
                pl.col("home_red_cards").alias("red_cards"),
                pl.col("home_team_score").alias("goals_for"),
                pl.col("away_team_score").alias("goals_against"),
                pl.col("home_team_score").alias("target"),
            ]
        )

    return match_df.select(
        [
            pl.col("match_id"),
            pl.col("match_date"),
            pl.col("competition"),
            pl.col("is_current_world_cup"),
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
            pl.col("away_prior_world_cup_appearances").alias("team_prior_world_cup_appearances"),
            pl.col("home_prior_world_cup_appearances").alias(
                "opponent_prior_world_cup_appearances"
            ),
            pl.col("away_prior_world_cup_points_per_match").alias(
                "team_prior_world_cup_points_per_match"
            ),
            pl.col("home_prior_world_cup_points_per_match").alias(
                "opponent_prior_world_cup_points_per_match"
            ),
            pl.col("away_prior_world_cup_goal_diff_per_match").alias(
                "team_prior_world_cup_goal_diff_per_match"
            ),
            pl.col("home_prior_world_cup_goal_diff_per_match").alias(
                "opponent_prior_world_cup_goal_diff_per_match"
            ),
            pl.col("away_prior_world_cup_yellow_cards_per_match").alias(
                "team_prior_world_cup_yellow_cards_per_match"
            ),
            pl.col("home_prior_world_cup_yellow_cards_per_match").alias(
                "opponent_prior_world_cup_yellow_cards_per_match"
            ),
            pl.col("away_prior_world_cup_sending_offs_per_match").alias(
                "team_prior_world_cup_sending_offs_per_match"
            ),
            pl.col("home_prior_world_cup_sending_offs_per_match").alias(
                "opponent_prior_world_cup_sending_offs_per_match"
            ),
            pl.col("away_prior_world_cup_fair_play_penalty_per_match").alias(
                "team_prior_world_cup_fair_play_penalty_per_match"
            ),
            pl.col("home_prior_world_cup_fair_play_penalty_per_match").alias(
                "opponent_prior_world_cup_fair_play_penalty_per_match"
            ),
            pl.col("away_market_value_eur").alias("team_market_value_eur"),
            pl.col("home_market_value_eur").alias("opponent_market_value_eur"),
            pl.col("away_avg_overall").alias("team_avg_overall"),
            pl.col("home_avg_overall").alias("opponent_avg_overall"),
            pl.col("away_avg_pace").alias("team_avg_pace"),
            pl.col("home_avg_pace").alias("opponent_avg_pace"),
            pl.col("away_avg_stamina").alias("team_avg_stamina"),
            pl.col("home_avg_stamina").alias("opponent_avg_stamina"),
            pl.col("away_avg_goalkeeper_overall").alias("team_avg_goalkeeper_overall"),
            pl.col("home_avg_goalkeeper_overall").alias("opponent_avg_goalkeeper_overall"),
            pl.col("away_avg_defense_overall").alias("team_avg_defense_overall"),
            pl.col("home_avg_defense_overall").alias("opponent_avg_defense_overall"),
            pl.col("away_avg_midfield_overall").alias("team_avg_midfield_overall"),
            pl.col("home_avg_midfield_overall").alias("opponent_avg_midfield_overall"),
            pl.col("away_avg_attack_overall").alias("team_avg_attack_overall"),
            pl.col("home_avg_attack_overall").alias("opponent_avg_attack_overall"),
            pl.col("away_squad_depth_proxy").alias("team_squad_depth_proxy"),
            pl.col("home_squad_depth_proxy").alias("opponent_squad_depth_proxy"),
            pl.col("away_xg").alias("xg_for"),
            pl.col("home_xg").alias("xg_against"),
            pl.col("away_possession_pct").alias("possession_pct"),
            pl.col("away_shots").alias("shots_for"),
            pl.col("home_shots").alias("shots_against"),
            pl.col("away_shots_on_target").alias("shots_on_target_for"),
            pl.col("home_shots_on_target").alias("shots_on_target_against"),
            pl.col("away_corners").alias("corners_for"),
            pl.col("home_corners").alias("corners_against"),
            pl.col("away_yellow_cards").alias("yellow_cards"),
            pl.col("away_red_cards").alias("red_cards"),
            pl.col("away_team_score").alias("goals_for"),
            pl.col("home_team_score").alias("goals_against"),
            pl.col("away_team_score").alias("target"),
        ]
    )


def _recent_form(team_df: pl.DataFrame) -> pl.DataFrame:
    """Compute recent form using only prior non-holdout rows for each team."""
    sorted_df = team_df.sort(["team", "match_date", "match_id"])
    enriched = sorted_df.group_by("team", maintain_order=True).map_groups(_recent_form_for_team)

    return enriched.select(
        [
            pl.col("match_id"),
            pl.col("match_date"),
            pl.col("competition"),
            pl.col("is_current_world_cup"),
            pl.col("team_world_cup_probability_elo_before"),
            pl.col("opponent_world_cup_probability_elo_before"),
            pl.col("team_world_football_elo_ratings"),
            pl.col("opponent_world_football_elo_ratings"),
            pl.col("team_fifa_world_ranking_points"),
            pl.col("opponent_fifa_world_ranking_points"),
            pl.col("team_fifa_world_ranking_rank"),
            pl.col("opponent_fifa_world_ranking_rank"),
            pl.col("team_prior_world_cup_appearances"),
            pl.col("opponent_prior_world_cup_appearances"),
            pl.col("team_prior_world_cup_points_per_match"),
            pl.col("opponent_prior_world_cup_points_per_match"),
            pl.col("team_prior_world_cup_goal_diff_per_match"),
            pl.col("opponent_prior_world_cup_goal_diff_per_match"),
            pl.col("team_prior_world_cup_yellow_cards_per_match"),
            pl.col("opponent_prior_world_cup_yellow_cards_per_match"),
            pl.col("team_prior_world_cup_sending_offs_per_match"),
            pl.col("opponent_prior_world_cup_sending_offs_per_match"),
            pl.col("team_prior_world_cup_fair_play_penalty_per_match"),
            pl.col("opponent_prior_world_cup_fair_play_penalty_per_match"),
            pl.col("team_market_value_eur"),
            pl.col("opponent_market_value_eur"),
            pl.col("team_avg_overall"),
            pl.col("opponent_avg_overall"),
            pl.col("team_avg_pace"),
            pl.col("opponent_avg_pace"),
            pl.col("team_avg_stamina"),
            pl.col("opponent_avg_stamina"),
            pl.col("team_avg_goalkeeper_overall"),
            pl.col("opponent_avg_goalkeeper_overall"),
            pl.col("team_avg_defense_overall"),
            pl.col("opponent_avg_defense_overall"),
            pl.col("team_avg_midfield_overall"),
            pl.col("opponent_avg_midfield_overall"),
            pl.col("team_avg_attack_overall"),
            pl.col("opponent_avg_attack_overall"),
            pl.col("team_squad_depth_proxy"),
            pl.col("opponent_squad_depth_proxy"),
            pl.col("recent_form"),
            *[pl.col(column) for column in ROLLING_FEATURE_DEFAULTS],
            pl.col("is_friendly_match"),
            pl.col("target"),
        ]
    )


def _recent_form_for_team(team_df: pl.DataFrame) -> pl.DataFrame:
    """Compute recent form without letting 2026 World Cup rows enter history."""
    goals_for_history: list[float] = []
    goals_against_history: list[float] = []
    metric_histories: dict[str, list[float]] = {
        "xg_for": [],
        "xg_against": [],
        "possession_pct": [],
        "shots_for": [],
        "shots_against": [],
        "shots_on_target_for": [],
        "shots_on_target_against": [],
        "corners_for": [],
        "corners_against": [],
        "yellow_cards": [],
        "red_cards": [],
    }
    adjusted_form_history: list[float] = []
    official_form_history: list[float] = []
    friendly_form_history: list[float] = []
    recent_form: list[float] = []
    rolling_values: dict[str, list[float]] = {column: [] for column in ROLLING_FEATURE_DEFAULTS}
    is_friendly_match: list[float] = []

    for row in team_df.iter_rows(named=True):
        if goals_for_history:
            recent_goals_for = goals_for_history[-5:]
            recent_goals_against = goals_against_history[-5:]
            form = (sum(recent_goals_for) / len(recent_goals_for)) - (
                sum(recent_goals_against) / len(recent_goals_against)
            )
        else:
            form = 0.0

        recent_form.append(form)
        rolling_values["recent_xg_for"].append(_rolling_average(metric_histories["xg_for"]))
        rolling_values["recent_xg_against"].append(_rolling_average(metric_histories["xg_against"]))
        rolling_values["recent_possession_pct"].append(
            _rolling_average(metric_histories["possession_pct"])
        )
        rolling_values["recent_shots_for"].append(_rolling_average(metric_histories["shots_for"]))
        rolling_values["recent_shots_against"].append(
            _rolling_average(metric_histories["shots_against"])
        )
        rolling_values["recent_shots_on_target_for"].append(
            _rolling_average(metric_histories["shots_on_target_for"])
        )
        rolling_values["recent_shots_on_target_against"].append(
            _rolling_average(metric_histories["shots_on_target_against"])
        )
        rolling_values["recent_corners_for"].append(
            _rolling_average(metric_histories["corners_for"])
        )
        rolling_values["recent_corners_against"].append(
            _rolling_average(metric_histories["corners_against"])
        )
        rolling_values["recent_yellow_cards"].append(
            _rolling_average(metric_histories["yellow_cards"])
        )
        rolling_values["recent_red_cards"].append(_rolling_average(metric_histories["red_cards"]))
        rolling_values["opponent_adjusted_recent_form"].append(
            _rolling_average(adjusted_form_history)
        )
        rolling_values["recent_official_form"].append(_rolling_average(official_form_history))
        rolling_values["recent_friendly_form"].append(_rolling_average(friendly_form_history))
        friendly_match = _is_friendly_competition(row.get("competition"))
        is_friendly_match.append(1.0 if friendly_match else 0.0)

        if not bool(row["is_current_world_cup"]):
            goals_for = float(row["goals_for"])
            goals_against = float(row["goals_against"])
            goal_diff = goals_for - goals_against
            goals_for_history.append(goals_for)
            goals_against_history.append(goals_against)
            opponent_rating = _maybe_float(row.get("opponent_world_cup_probability_elo_before"))
            opponent_strength = (opponent_rating or 1500.0) / 1500.0
            adjusted_form_history.append(goal_diff * opponent_strength)
            if friendly_match:
                friendly_form_history.append(goal_diff)
            else:
                official_form_history.append(goal_diff)

            for metric_name, history in metric_histories.items():
                metric_value = _maybe_float(row.get(metric_name))
                if metric_value is not None:
                    history.append(metric_value)

    return team_df.with_columns(
        [
            pl.Series("recent_form", recent_form),
            *[pl.Series(column, values) for column, values in rolling_values.items()],
            pl.Series("is_friendly_match", is_friendly_match),
        ]
    )


def _rolling_average(values: list[float], *, window: int = 5) -> float:
    recent_values = values[-window:]
    if not recent_values:
        return 0.0
    return sum(recent_values) / len(recent_values)


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    numeric_value = float(value)
    if numeric_value != numeric_value:
        return None
    return numeric_value


def _is_friendly_competition(competition: object) -> bool:
    if competition is None:
        return False
    normalized = str(competition).casefold()
    return "friendly" in normalized or "amistoso" in normalized


def main() -> int:
    """CLI entrypoint."""
    frame = build_feature_frame()
    LOGGER.info("Generated feature frame with shape %s", frame.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
