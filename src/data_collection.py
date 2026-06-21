"""Collect and normalize real football data sources into the DuckDB warehouse.

The collectors in this module keep raw source files under ``data/raw`` and load
normalized analytical tables into ``data/warehouse/world_cup.duckdb``. The
business cutoff for historical match data is January 1st, 2010.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import duckdb
import httpx
import requests
from bs4 import BeautifulSoup

try:
    from .competition_filters import current_world_cup_exclusion_sql
    from .db_init import DB_PATH, RAW_DIR, initialize_database
    from .settings import DEFAULT_FBREF_MANIFEST, DEFAULT_TRANSFERMARKT_MANIFEST
except ImportError:  # pragma: no cover - supports direct script execution.
    from competition_filters import current_world_cup_exclusion_sql
    from db_init import DB_PATH, RAW_DIR, initialize_database
    from settings import DEFAULT_FBREF_MANIFEST, DEFAULT_TRANSFERMARKT_MANIFEST

LOGGER = logging.getLogger(__name__)

DATA_CUTOFF_DATE = date(2010, 1, 1)
MATCH_RESULTS_DATASET = "martj42/international-football-results-from-1872-to-2017"
DEFAULT_EA_FC_DATASET = "flynn28/eafc26-player-database"
DEFAULT_FBREF_LEAGUES = ("INT-World Cup", "INT-European Championship")
DEFAULT_FBREF_SEASONS = ("2010", "2014", "2018", "2022", "2024")
FJELSTUL_WORLDCUP_MATCHES_URL = (
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"
)
FJELSTUL_WORLDCUP_BOOKINGS_URL = (
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/bookings.csv"
)
FJELSTUL_WORLDCUP_RAW_FILENAME = "matches.csv"
FJELSTUL_WORLDCUP_BOOKINGS_RAW_FILENAME = "bookings.csv"
TRANSFERMARKT_USER_AGENT = (
    "Mozilla/5.0 (compatible; world-cup-probability-research/0.1; +https://localhost)"
)

TABULAR_SUFFIXES = {".csv", ".parquet"}
ALL_COLLECTION_SOURCES = ("matches", "fbref", "squad", "transfermarkt", "fjelstul")
DEFAULT_DOWNLOAD_SOURCES = ("matches", "squad", "transfermarkt", "fjelstul")
DEFAULT_LOAD_SOURCES = ("matches", "squad", "transfermarkt", "fjelstul")
EURO_VALUE_PATTERN = re.compile(
    r"€\s*(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>bn|m|k)?",
    re.IGNORECASE,
)


class KaggleDatasetClient(Protocol):
    """Subset of the official Kaggle API needed by this module."""

    def authenticate(self) -> None:
        """Authenticate the Kaggle client."""

    def dataset_download_files(
        self,
        dataset: str,
        path: str | None = None,
        force: bool = False,
        quiet: bool = True,
        unzip: bool = False,
        licenses: Sequence[str] = (),
    ) -> None:
        """Download all files for a Kaggle dataset."""


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Result of one source normalization step."""

    source_name: str
    raw_path: Path
    rows_loaded: int


@dataclass(frozen=True, slots=True)
class TransfermarktTeamTarget:
    """A Transfermarkt team page to scrape."""

    team_id: str
    team_name: str
    url: str | None = None
    search_query: str | None = None


@dataclass(frozen=True, slots=True)
class TransfermarktMarketValue:
    """Normalized market value scraped from Transfermarkt."""

    team_id: str
    team_name: str
    total_market_value_eur: float
    source_url: str
    scraped_at: datetime


def download_kaggle_dataset(
    dataset: str,
    destination: Path,
    *,
    client: KaggleDatasetClient | None = None,
    force: bool = False,
) -> Path:
    """Download and extract one Kaggle dataset into a raw source directory.

    Args:
        dataset: Kaggle dataset slug, for example
            ``martj42/international-football-results-from-1872-to-2017``.
        destination: Directory that will receive the extracted files.
        client: Optional test double or preconfigured Kaggle client.
        force: Whether Kaggle should redownload existing files.

    Returns:
        The destination directory containing extracted source files.
    """
    _validate_kaggle_dataset_slug(dataset)
    destination.mkdir(parents=True, exist_ok=True)
    if _has_existing_raw_files(destination) and not force:
        LOGGER.info(
            "Skipping Kaggle dataset '%s' because raw files already exist in %s",
            dataset,
            destination,
        )
        return destination

    kaggle_client = client if client is not None else _create_kaggle_client()
    kaggle_client.authenticate()
    try:
        kaggle_client.dataset_download_files(
            dataset,
            path=str(destination),
            force=force,
            quiet=False,
            unzip=True,
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(_kaggle_download_error_message(dataset, exc)) from exc
    _extract_nested_archives(destination)
    return destination


def load_historical_matches(
    source_root: Path,
    *,
    db_path: Path = DB_PATH,
    cutoff_date: date = DATA_CUTOFF_DATE,
) -> LoadResult:
    """Load Kaggle international results into ``f_matches`` with the 2010 cutoff."""
    initialize_database(db_path=db_path, load_raw_files=False)
    source_file = _find_tabular_file(
        source_root,
        required_alias_groups=(
            ("date", "match_date"),
            ("home_team", "home"),
            ("away_team", "away"),
            ("home_score", "home_team_score"),
            ("away_score", "away_team_score"),
            ("tournament", "competition"),
        ),
    )

    with duckdb.connect(str(db_path)) as con:
        source_sql = _source_relation_sql(source_file)
        columns = _source_columns(con, source_file)
        lookup = _column_lookup(columns)
        date_col = _required_column(lookup, ("date", "match_date"))
        home_col = _required_column(lookup, ("home_team", "home"))
        away_col = _required_column(lookup, ("away_team", "away"))
        home_score_col = _required_column(lookup, ("home_score", "home_team_score"))
        away_score_col = _required_column(lookup, ("away_score", "away_team_score"))
        tournament_col = _required_column(lookup, ("tournament", "competition"))
        neutral_col = _optional_column(lookup, ("neutral", "neutral_site"))
        country_col = _optional_column(lookup, ("country",))
        city_col = _optional_column(lookup, ("city",))

        home_expr = f"CAST({_quote_identifier(home_col)} AS VARCHAR)"
        away_expr = f"CAST({_quote_identifier(away_col)} AS VARCHAR)"
        date_expr = f"CAST({_quote_identifier(date_col)} AS DATE)"
        tournament_expr = f"CAST({_quote_identifier(tournament_col)} AS VARCHAR)"
        home_score_expr = f"TRY_CAST({_quote_identifier(home_score_col)} AS INTEGER)"
        away_score_expr = f"TRY_CAST({_quote_identifier(away_score_col)} AS INTEGER)"
        neutral_expr = (
            f"CAST({_quote_identifier(neutral_col)} AS BOOLEAN)"
            if neutral_col is not None
            else "NULL::BOOLEAN"
        )
        country_expr = (
            f"CAST({_quote_identifier(country_col)} AS VARCHAR)"
            if country_col is not None
            else "NULL::VARCHAR"
        )
        city_expr = (
            f"CAST({_quote_identifier(city_col)} AS VARCHAR)"
            if city_col is not None
            else "NULL::VARCHAR"
        )
        source_file_expr = _sql_string_literal(str(source_file))
        current_world_cup_exclusion = current_world_cup_exclusion_sql(
            date_expr=date_expr,
            competition_expr=tournament_expr,
        )
        valid_match_filter = (
            f"{date_expr} >= {_sql_date_literal(cutoff_date)} "
            f"AND {home_score_expr} IS NOT NULL "
            f"AND {away_score_expr} IS NOT NULL "
            f"AND {current_world_cup_exclusion}"
        )

        con.execute("DELETE FROM f_matches WHERE source_file = ?", [str(source_file)])
        _ensure_teams_from_query(
            con,
            source_sql,
            home_expr,
            source_file,
            where_sql=valid_match_filter,
        )
        _ensure_teams_from_query(
            con,
            source_sql,
            away_expr,
            source_file,
            where_sql=valid_match_filter,
        )
        con.execute(
            f"""
            INSERT INTO f_matches (
                match_id,
                match_date,
                competition,
                season,
                stage,
                home_team_id,
                away_team_id,
                home_team_score,
                away_team_score,
                home_xg,
                away_xg,
                venue,
                city,
                country,
                attendance,
                neutral_site,
                source_file,
                loaded_at
            )
            SELECT
                md5(
                    CAST({date_expr} AS VARCHAR) || ':' ||
                    {home_expr} || ':' ||
                    {away_expr} || ':' ||
                    {tournament_expr} || ':' ||
                    CAST({home_score_expr} AS VARCHAR) || ':' ||
                    CAST({away_score_expr} AS VARCHAR)
                ) AS match_id,
                {date_expr} AS match_date,
                {tournament_expr} AS competition,
                CAST(EXTRACT(year FROM {date_expr}) AS VARCHAR) AS season,
                NULL::VARCHAR AS stage,
                {home_expr} AS home_team_id,
                {away_expr} AS away_team_id,
                {home_score_expr} AS home_team_score,
                {away_score_expr} AS away_team_score,
                NULL::DOUBLE AS home_xg,
                NULL::DOUBLE AS away_xg,
                NULL::VARCHAR AS venue,
                {city_expr} AS city,
                {country_expr} AS country,
                NULL::INTEGER AS attendance,
                {neutral_expr} AS neutral_site,
                {source_file_expr} AS source_file,
                current_timestamp AS loaded_at
            FROM {source_sql}
            WHERE {valid_match_filter}
            ON CONFLICT (match_id) DO UPDATE SET
                match_date = excluded.match_date,
                competition = excluded.competition,
                season = excluded.season,
                home_team_id = excluded.home_team_id,
                away_team_id = excluded.away_team_id,
                home_team_score = excluded.home_team_score,
                away_team_score = excluded.away_team_score,
                city = excluded.city,
                country = excluded.country,
                neutral_site = excluded.neutral_site,
                source_file = excluded.source_file,
                loaded_at = excluded.loaded_at
            """,
        )
        _delete_unreferenced_auto_teams(con, source_file)
        rows_loaded = int(
            con.execute(
                "SELECT COUNT(*) FROM f_matches WHERE source_file = ?", [str(source_file)]
            ).fetchone()[0]
        )

    return LoadResult("historical_matches", source_file, rows_loaded)


def fetch_fbref_with_soccerdata(
    *,
    leagues: Sequence[str],
    seasons: Sequence[str | int],
    raw_dir: Path = RAW_DIR / "fbref",
    no_cache: bool = False,
) -> list[Path]:
    """Fetch FBref team match stats through ``soccerdata`` and store raw files.

    The caller must pass the FBref leagues/seasons to collect. This keeps the
    code source-agnostic because soccerdata league aliases can change over time.
    """
    if not leagues:
        raise ValueError("At least one FBref league must be provided.")
    if not seasons:
        raise ValueError("At least one FBref season must be provided.")

    try:
        import soccerdata as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("soccerdata is not installed.") from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    fbref = sd.FBref(leagues=list(leagues), seasons=list(seasons), no_cache=no_cache)
    written_paths: list[Path] = []

    schedule = fbref.read_schedule()
    written_paths.append(_write_soccerdata_frame(schedule, raw_dir / "fbref_schedule.parquet"))

    team_stats = fbref.read_team_match_stats(stat_type="schedule")
    written_paths.append(_write_soccerdata_frame(team_stats, raw_dir / "fbref_team_stats.parquet"))

    return written_paths


def load_fbref_match_stats(
    source_root: Path,
    *,
    db_path: Path = DB_PATH,
    cutoff_date: date = DATA_CUTOFF_DATE,
) -> LoadResult:
    """Load FBref xG and possession rows into ``f_match_stats``."""
    initialize_database(db_path=db_path, load_raw_files=False)
    source_file = _find_tabular_file(
        source_root,
        required_alias_groups=(
            ("date", "match_date"),
            ("team", "squad", "home_team", "home"),
            ("xg", "expected_goals"),
        ),
    )

    with duckdb.connect(str(db_path)) as con:
        source_sql = _source_relation_sql(source_file)
        columns = _source_columns(con, source_file)
        lookup = _column_lookup(columns)
        date_col = _required_column(lookup, ("date", "match_date"))
        team_col = _required_column(lookup, ("team", "squad", "home_team", "home"))
        opponent_col = _optional_column(lookup, ("opponent", "away_team", "away"))
        tournament_col = _optional_column(lookup, ("competition", "tournament", "league"))
        xg_col = _required_column(lookup, ("xg", "expected_goals"))
        possession_col = _optional_column(lookup, ("possession", "poss", "possession_pct"))

        date_expr = f"CAST({_quote_identifier(date_col)} AS DATE)"
        team_expr = f"CAST({_quote_identifier(team_col)} AS VARCHAR)"
        opponent_expr = (
            f"CAST({_quote_identifier(opponent_col)} AS VARCHAR)"
            if opponent_col is not None
            else "NULL::VARCHAR"
        )
        tournament_expr = (
            f"CAST({_quote_identifier(tournament_col)} AS VARCHAR)"
            if tournament_col is not None
            else "NULL::VARCHAR"
        )
        xg_expr = _numeric_expression(xg_col)
        possession_expr = (
            _numeric_expression(possession_col) if possession_col is not None else "NULL::DOUBLE"
        )
        source_file_expr = _sql_string_literal(str(source_file))

        _ensure_teams_from_query(con, source_sql, team_expr, source_file)
        if opponent_col is not None:
            _ensure_teams_from_query(con, source_sql, opponent_expr, source_file)

        con.execute(
            "DELETE FROM f_match_stats WHERE source_file = ? AND source = 'fbref'",
            [str(source_file)],
        )
        con.execute(
            f"""
            INSERT INTO f_match_stats (
                match_id,
                match_date,
                team_id,
                opponent_team_id,
                tournament,
                xg,
                possession_pct,
                source,
                source_file,
                loaded_at
            )
            WITH normalized AS (
                SELECT
                    {date_expr} AS match_date,
                    {team_expr} AS team_id,
                    {opponent_expr} AS opponent_team_id,
                    {tournament_expr} AS tournament,
                    {xg_expr} AS xg,
                    {possession_expr} AS possession_pct
                FROM {source_sql}
                WHERE {date_expr} >= ?
                  AND {team_expr} IS NOT NULL
            )
            SELECT
                COALESCE(
                    m.match_id,
                    md5(
                        CAST(n.match_date AS VARCHAR) || ':' ||
                        n.team_id || ':' ||
                        COALESCE(n.opponent_team_id, '') || ':fbref'
                    )
                ) AS match_id,
                n.match_date,
                n.team_id,
                n.opponent_team_id,
                n.tournament,
                n.xg,
                CASE
                    WHEN n.possession_pct > 1.0 THEN n.possession_pct
                    WHEN n.possession_pct IS NOT NULL THEN n.possession_pct * 100.0
                    ELSE NULL
                END AS possession_pct,
                'fbref' AS source,
                {source_file_expr} AS source_file,
                current_timestamp AS loaded_at
            FROM normalized AS n
            LEFT JOIN f_matches AS m
                ON m.match_date = n.match_date
               AND (
                    (m.home_team_id = n.team_id AND m.away_team_id = n.opponent_team_id)
                    OR (m.away_team_id = n.team_id AND m.home_team_id = n.opponent_team_id)
               )
            ON CONFLICT (match_id, team_id, source) DO UPDATE SET
                match_date = excluded.match_date,
                opponent_team_id = excluded.opponent_team_id,
                tournament = excluded.tournament,
                xg = excluded.xg,
                possession_pct = excluded.possession_pct,
                source_file = excluded.source_file,
                loaded_at = excluded.loaded_at
            """,
            [cutoff_date],
        )
        rows_loaded = int(
            con.execute(
                "SELECT COUNT(*) FROM f_match_stats WHERE source_file = ? AND source = 'fbref'",
                [str(source_file)],
            ).fetchone()[0]
        )

    return LoadResult("fbref_match_stats", source_file, rows_loaded)


def load_squad_attributes(
    source_root: Path,
    *,
    db_path: Path = DB_PATH,
    source_season: str,
    source_dataset: str,
) -> LoadResult:
    """Aggregate EA FC/FIFA player attributes into ``d_squad_attributes``.

    The source generally has one row per player. Because most public EA FC files
    do not identify a team's habitual XI, this loader uses the top 11 players by
    overall rating for each nationality as the initial quality proxy.
    """
    initialize_database(db_path=db_path, load_raw_files=False)
    source_file = _find_tabular_file(
        source_root,
        required_alias_groups=(
            ("nationality", "nation", "team", "country"),
            ("overall", "ova", "ovr"),
        ),
    )

    with duckdb.connect(str(db_path)) as con:
        source_sql = _source_relation_sql(source_file)
        columns = _source_columns(con, source_file)
        lookup = _column_lookup(columns)
        team_col = _required_column(lookup, ("nationality", "nation", "team", "country"))
        overall_col = _required_column(lookup, ("overall", "ova", "ovr"))
        pace_col = _optional_column(lookup, ("pace", "pac"))
        stamina_col = _optional_column(lookup, ("stamina", "sta"))

        team_expr = f"CAST({_quote_identifier(team_col)} AS VARCHAR)"
        overall_expr = _numeric_expression(overall_col)
        pace_expr = _numeric_expression(pace_col) if pace_col is not None else "NULL::DOUBLE"
        stamina_expr = (
            _numeric_expression(stamina_col) if stamina_col is not None else "NULL::DOUBLE"
        )
        source_file_expr = _sql_string_literal(str(source_file))

        _ensure_teams_from_query(con, source_sql, team_expr, source_file)
        con.execute(
            "DELETE FROM d_squad_attributes WHERE source_season = ? AND source_file = ?",
            [source_season, str(source_file)],
        )
        con.execute(
            f"""
            INSERT INTO d_squad_attributes (
                team_id,
                source_season,
                avg_overall,
                avg_pace,
                avg_stamina,
                sampled_player_count,
                source_dataset,
                source_file,
                loaded_at
            )
            WITH ranked_players AS (
                SELECT
                    {team_expr} AS team_id,
                    {overall_expr} AS overall,
                    {pace_expr} AS pace,
                    {stamina_expr} AS stamina,
                    ROW_NUMBER() OVER (
                        PARTITION BY {team_expr}
                        ORDER BY {overall_expr} DESC NULLS LAST
                    ) AS player_rank
                FROM {source_sql}
                WHERE {team_expr} IS NOT NULL
                  AND {overall_expr} IS NOT NULL
            ),
            top_players AS (
                SELECT *
                FROM ranked_players
                WHERE player_rank <= 11
            )
            SELECT
                team_id,
                ? AS source_season,
                AVG(overall) AS avg_overall,
                AVG(pace) AS avg_pace,
                AVG(stamina) AS avg_stamina,
                COUNT(*)::INTEGER AS sampled_player_count,
                ? AS source_dataset,
                {source_file_expr} AS source_file,
                current_timestamp AS loaded_at
            FROM top_players
            GROUP BY team_id
            ON CONFLICT (team_id, source_season) DO UPDATE SET
                avg_overall = excluded.avg_overall,
                avg_pace = excluded.avg_pace,
                avg_stamina = excluded.avg_stamina,
                sampled_player_count = excluded.sampled_player_count,
                source_dataset = excluded.source_dataset,
                source_file = excluded.source_file,
                loaded_at = excluded.loaded_at
            """,
            [source_season, source_dataset],
        )
        rows_loaded = int(
            con.execute(
                "SELECT COUNT(*) FROM d_squad_attributes WHERE source_file = ?",
                [str(source_file)],
            ).fetchone()[0]
        )

    return LoadResult("squad_attributes", source_file, rows_loaded)


def download_fjelstul_worldcup_matches(
    destination: Path,
    *,
    force: bool = False,
) -> Path:
    """Download the Fjelstul World Cup CSVs used by the project."""
    destination.mkdir(parents=True, exist_ok=True)

    for url, filename in (
        (FJELSTUL_WORLDCUP_MATCHES_URL, FJELSTUL_WORLDCUP_RAW_FILENAME),
        (FJELSTUL_WORLDCUP_BOOKINGS_URL, FJELSTUL_WORLDCUP_BOOKINGS_RAW_FILENAME),
    ):
        target_path = destination / filename
        if target_path.exists() and target_path.stat().st_size > 0 and not force:
            LOGGER.info(
                "Skipping Fjelstul World Cup download because %s already exists",
                target_path,
            )
            continue

        try:
            response = httpx.get(
                url,
                timeout=30.0,
                follow_redirects=True,
            )
            response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            raise RuntimeError(f"Failed to download Fjelstul World Cup file: {filename}") from exc
        target_path.write_bytes(response.content)

    return destination


def load_fjelstul_world_cup_history(
    source_root: Path,
    *,
    db_path: Path = DB_PATH,
) -> LoadResult:
    """Load temporally safe prior World Cup history features from Fjelstul matches."""
    initialize_database(db_path=db_path, load_raw_files=False)
    source_file = (
        source_root
        if source_root.is_file()
        else _find_tabular_file(
            source_root,
            required_alias_groups=(
                ("tournament_id",),
                ("tournament_name",),
                ("match_date",),
                ("home_team_id",),
                ("home_team_name",),
                ("away_team_id",),
                ("away_team_name",),
                ("home_team_score",),
                ("away_team_score",),
            ),
        )
    )
    bookings_file = _resolve_optional_fjelstul_bookings_raw(source_root)

    with duckdb.connect(str(db_path)) as con:
        source_sql = _source_relation_sql(source_file)
        columns = _source_columns(con, source_file)
        lookup = _column_lookup(columns)
        tournament_id_col = _required_column(lookup, ("tournament_id",))
        tournament_name_col = _required_column(lookup, ("tournament_name",))
        match_date_col = _required_column(lookup, ("match_date",))
        home_team_id_col = _required_column(lookup, ("home_team_id",))
        home_team_name_col = _required_column(lookup, ("home_team_name",))
        home_team_code_col = _optional_column(lookup, ("home_team_code",))
        away_team_id_col = _required_column(lookup, ("away_team_id",))
        away_team_name_col = _required_column(lookup, ("away_team_name",))
        away_team_code_col = _optional_column(lookup, ("away_team_code",))
        home_score_col = _required_column(lookup, ("home_team_score",))
        away_score_col = _required_column(lookup, ("away_team_score",))

        tournament_id_expr = f"CAST({_quote_identifier(tournament_id_col)} AS VARCHAR)"
        tournament_name_expr = f"CAST({_quote_identifier(tournament_name_col)} AS VARCHAR)"
        match_date_expr = f"CAST({_quote_identifier(match_date_col)} AS DATE)"
        home_team_id_expr = f"CAST({_quote_identifier(home_team_id_col)} AS VARCHAR)"
        home_team_name_expr = f"CAST({_quote_identifier(home_team_name_col)} AS VARCHAR)"
        home_team_code_expr = (
            f"CAST({_quote_identifier(home_team_code_col)} AS VARCHAR)"
            if home_team_code_col is not None
            else "NULL::VARCHAR"
        )
        away_team_id_expr = f"CAST({_quote_identifier(away_team_id_col)} AS VARCHAR)"
        away_team_name_expr = f"CAST({_quote_identifier(away_team_name_col)} AS VARCHAR)"
        away_team_code_expr = (
            f"CAST({_quote_identifier(away_team_code_col)} AS VARCHAR)"
            if away_team_code_col is not None
            else "NULL::VARCHAR"
        )
        home_score_expr = f"TRY_CAST({_quote_identifier(home_score_col)} AS INTEGER)"
        away_score_expr = f"TRY_CAST({_quote_identifier(away_score_col)} AS INTEGER)"
        source_file_expr = _sql_string_literal(str(source_file))
        max_as_of_year = max(date.today().year + 4, 2026)

        con.execute("DELETE FROM d_world_cup_prior_team_history")
        con.execute(
            f"""
            INSERT INTO d_world_cup_prior_team_history (
                team_id,
                team_name,
                team_code,
                normalized_team_name,
                as_of_year,
                prior_world_cup_appearances,
                prior_world_cup_points_per_match,
                prior_world_cup_goal_diff_per_match,
                source_file,
                loaded_at
            )
            WITH raw_matches AS (
                SELECT
                    {tournament_id_expr} AS tournament_id,
                    {tournament_name_expr} AS tournament_name,
                    {match_date_expr} AS match_date,
                    CAST(EXTRACT(year FROM {match_date_expr}) AS INTEGER) AS tournament_year,
                    {home_team_id_expr} AS home_team_id,
                    {home_team_name_expr} AS home_team_name,
                    {home_team_code_expr} AS home_team_code,
                    {away_team_id_expr} AS away_team_id,
                    {away_team_name_expr} AS away_team_name,
                    {away_team_code_expr} AS away_team_code,
                    {home_score_expr} AS home_team_score,
                    {away_score_expr} AS away_team_score
                FROM {source_sql}
                WHERE LOWER({tournament_name_expr}) LIKE '%men''s world cup%'
                  AND {home_score_expr} IS NOT NULL
                  AND {away_score_expr} IS NOT NULL
            ),
            team_matches AS (
                SELECT
                    tournament_id,
                    tournament_year,
                    home_team_id AS team_id,
                    home_team_name AS team_name,
                    home_team_code AS team_code,
                    1 AS played,
                    CASE
                        WHEN home_team_score > away_team_score THEN 3
                        WHEN home_team_score = away_team_score THEN 1
                        ELSE 0
                    END AS points,
                    home_team_score - away_team_score AS goal_difference
                FROM raw_matches
                UNION ALL
                SELECT
                    tournament_id,
                    tournament_year,
                    away_team_id AS team_id,
                    away_team_name AS team_name,
                    away_team_code AS team_code,
                    1 AS played,
                    CASE
                        WHEN away_team_score > home_team_score THEN 3
                        WHEN away_team_score = home_team_score THEN 1
                        ELSE 0
                    END AS points,
                    away_team_score - home_team_score AS goal_difference
                FROM raw_matches
            ),
            team_tournament_stats AS (
                SELECT
                    team_id,
                    ANY_VALUE(team_name) AS team_name,
                    ANY_VALUE(team_code) AS team_code,
                    tournament_id,
                    tournament_year,
                    SUM(played) AS played,
                    SUM(points) AS points,
                    SUM(goal_difference) AS goal_difference
                FROM team_matches
                GROUP BY team_id, tournament_id, tournament_year
            ),
            teams AS (
                SELECT
                    team_id,
                    ARG_MAX(team_name, tournament_year) AS team_name,
                    ARG_MAX(team_code, tournament_year) AS team_code
                FROM team_tournament_stats
                GROUP BY team_id
            ),
            year_bounds AS (
                SELECT
                    MIN(tournament_year) AS min_year,
                    GREATEST(MAX(tournament_year) + 4, ?::INTEGER) AS max_year
                FROM team_tournament_stats
            ),
            team_year_grid AS (
                SELECT
                    teams.team_id,
                    teams.team_name,
                    teams.team_code,
                    years.as_of_year
                FROM teams
                CROSS JOIN year_bounds
                CROSS JOIN generate_series(
                    year_bounds.min_year,
                    year_bounds.max_year
                ) AS years(as_of_year)
            )
            SELECT
                grid.team_id,
                grid.team_name,
                grid.team_code,
                regexp_replace(lower(grid.team_name), '[^a-z0-9]+', '', 'g')
                    AS normalized_team_name,
                CAST(grid.as_of_year AS INTEGER) AS as_of_year,
                COUNT(stats.tournament_id)::INTEGER AS prior_world_cup_appearances,
                COALESCE(
                    SUM(stats.points)::DOUBLE / NULLIF(SUM(stats.played), 0),
                    0.0
                ) AS prior_world_cup_points_per_match,
                COALESCE(
                    SUM(stats.goal_difference)::DOUBLE / NULLIF(SUM(stats.played), 0),
                    0.0
                ) AS prior_world_cup_goal_diff_per_match,
                {source_file_expr} AS source_file,
                current_timestamp AS loaded_at
            FROM team_year_grid AS grid
            LEFT JOIN team_tournament_stats AS stats
                ON stats.team_id = grid.team_id
                AND stats.tournament_year < grid.as_of_year
            GROUP BY grid.team_id, grid.team_name, grid.team_code, grid.as_of_year
            """,
            [max_as_of_year],
        )
        rows_loaded = int(
            con.execute("SELECT COUNT(*) FROM d_world_cup_prior_team_history").fetchone()[0]
        )
        if bookings_file is not None:
            _load_fjelstul_world_cup_discipline_history(
                con,
                matches_source_file=source_file,
                bookings_source_file=bookings_file,
            )

    return LoadResult("fjelstul_world_cup_history", source_file, rows_loaded)


def _load_fjelstul_world_cup_discipline_history(
    con: duckdb.DuckDBPyConnection,
    *,
    matches_source_file: Path,
    bookings_source_file: Path,
) -> None:
    matches_sql = _source_relation_sql(matches_source_file)
    bookings_sql = _source_relation_sql(bookings_source_file)
    source_file_expr = _sql_string_literal(str(bookings_source_file))
    con.execute("DELETE FROM d_world_cup_prior_discipline_history")
    con.execute(
        f"""
        INSERT INTO d_world_cup_prior_discipline_history (
            team_id,
            team_name,
            team_code,
            normalized_team_name,
            as_of_year,
            prior_world_cup_yellow_cards_per_match,
            prior_world_cup_sending_offs_per_match,
            prior_world_cup_fair_play_penalty_per_match,
            source_file,
            loaded_at
        )
        WITH raw_matches AS (
            SELECT
                CAST(tournament_id AS VARCHAR) AS tournament_id,
                CAST(tournament_name AS VARCHAR) AS tournament_name,
                CAST(match_date AS DATE) AS match_date,
                CAST(EXTRACT(year FROM CAST(match_date AS DATE)) AS INTEGER) AS tournament_year,
                CAST(home_team_id AS VARCHAR) AS home_team_id,
                CAST(home_team_name AS VARCHAR) AS home_team_name,
                CAST(home_team_code AS VARCHAR) AS home_team_code,
                CAST(away_team_id AS VARCHAR) AS away_team_id,
                CAST(away_team_name AS VARCHAR) AS away_team_name,
                CAST(away_team_code AS VARCHAR) AS away_team_code
            FROM {matches_sql}
            WHERE LOWER(CAST(tournament_name AS VARCHAR)) LIKE '%men''s world cup%'
        ),
        team_matches AS (
            SELECT
                tournament_id,
                tournament_year,
                home_team_id AS team_id,
                home_team_name AS team_name,
                home_team_code AS team_code,
                1 AS played
            FROM raw_matches
            UNION ALL
            SELECT
                tournament_id,
                tournament_year,
                away_team_id AS team_id,
                away_team_name AS team_name,
                away_team_code AS team_code,
                1 AS played
            FROM raw_matches
        ),
        team_tournament_matches AS (
            SELECT
                team_id,
                ANY_VALUE(team_name) AS team_name,
                ANY_VALUE(team_code) AS team_code,
                tournament_id,
                tournament_year,
                SUM(played) AS played
            FROM team_matches
            GROUP BY team_id, tournament_id, tournament_year
        ),
        raw_bookings AS (
            SELECT
                CAST(tournament_id AS VARCHAR) AS tournament_id,
                CAST(team_id AS VARCHAR) AS team_id,
                TRY_CAST(yellow_card AS INTEGER) AS yellow_card,
                TRY_CAST(red_card AS INTEGER) AS red_card,
                TRY_CAST(second_yellow_card AS INTEGER) AS second_yellow_card,
                TRY_CAST(sending_off AS INTEGER) AS sending_off
            FROM {bookings_sql}
            WHERE LOWER(CAST(tournament_name AS VARCHAR)) LIKE '%men''s world cup%'
        ),
        team_tournament_bookings AS (
            SELECT
                m.team_id,
                m.tournament_id,
                SUM(COALESCE(b.yellow_card, 0)) AS yellow_cards,
                SUM(COALESCE(b.sending_off, 0)) AS sending_offs,
                SUM(
                    CASE
                        WHEN COALESCE(b.second_yellow_card, 0) = 1 THEN 3
                        WHEN COALESCE(b.red_card, 0) = 1
                            AND COALESCE(b.yellow_card, 0) = 1 THEN 5
                        WHEN COALESCE(b.red_card, 0) = 1 THEN 4
                        WHEN COALESCE(b.yellow_card, 0) = 1 THEN 1
                        ELSE 0
                    END
                ) AS fair_play_penalty
            FROM team_tournament_matches AS m
            LEFT JOIN raw_bookings AS b
                ON b.team_id = m.team_id
                AND b.tournament_id = m.tournament_id
            GROUP BY m.team_id, m.tournament_id
        ),
        teams AS (
            SELECT
                team_id,
                ARG_MAX(team_name, tournament_year) AS team_name,
                ARG_MAX(team_code, tournament_year) AS team_code
            FROM team_tournament_matches
            GROUP BY team_id
        ),
        year_bounds AS (
            SELECT
                MIN(tournament_year) AS min_year,
                GREATEST(MAX(tournament_year) + 4, ?::INTEGER) AS max_year
            FROM team_tournament_matches
        ),
        team_year_grid AS (
            SELECT
                teams.team_id,
                teams.team_name,
                teams.team_code,
                years.as_of_year
            FROM teams
            CROSS JOIN year_bounds
            CROSS JOIN generate_series(
                year_bounds.min_year,
                year_bounds.max_year
            ) AS years(as_of_year)
        )
        SELECT
            grid.team_id,
            grid.team_name,
            grid.team_code,
            regexp_replace(lower(grid.team_name), '[^a-z0-9]+', '', 'g')
                AS normalized_team_name,
            CAST(grid.as_of_year AS INTEGER) AS as_of_year,
            COALESCE(
                SUM(bookings.yellow_cards)::DOUBLE / NULLIF(SUM(matches.played), 0),
                0.0
            ) AS prior_world_cup_yellow_cards_per_match,
            COALESCE(
                SUM(bookings.sending_offs)::DOUBLE / NULLIF(SUM(matches.played), 0),
                0.0
            ) AS prior_world_cup_sending_offs_per_match,
            COALESCE(
                SUM(bookings.fair_play_penalty)::DOUBLE / NULLIF(SUM(matches.played), 0),
                0.0
            ) AS prior_world_cup_fair_play_penalty_per_match,
            {source_file_expr} AS source_file,
            current_timestamp AS loaded_at
        FROM team_year_grid AS grid
        LEFT JOIN team_tournament_matches AS matches
            ON matches.team_id = grid.team_id
            AND matches.tournament_year < grid.as_of_year
        LEFT JOIN team_tournament_bookings AS bookings
            ON bookings.team_id = matches.team_id
            AND bookings.tournament_id = matches.tournament_id
        GROUP BY grid.team_id, grid.team_name, grid.team_code, grid.as_of_year
        """,
        [max(date.today().year + 4, 2026)],
    )


def read_transfermarkt_manifest(path: Path) -> list[TransfermarktTeamTarget]:
    """Read a Transfermarkt team manifest JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_teams = payload.get("teams") if isinstance(payload, dict) else None
    if not isinstance(raw_teams, list):
        raise ValueError("Transfermarkt manifest must contain a 'teams' list.")

    targets: list[TransfermarktTeamTarget] = []
    for index, raw_team in enumerate(raw_teams):
        if not isinstance(raw_team, dict):
            raise ValueError(f"Manifest team at index {index} must be an object.")
        targets.append(
            TransfermarktTeamTarget(
                team_id=_read_manifest_str(raw_team, "team_id", index),
                team_name=_read_manifest_str(raw_team, "team_name", index),
                url=_read_optional_manifest_str(raw_team, "url"),
                search_query=_read_optional_manifest_str(raw_team, "search_query"),
            )
        )

    return targets


def scrape_transfermarkt_market_values(
    targets: Iterable[TransfermarktTeamTarget],
    *,
    timeout_seconds: float = 30.0,
) -> list[TransfermarktMarketValue]:
    """Scrape total team market values from Transfermarkt pages."""
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": TRANSFERMARKT_USER_AGENT,
    }
    results: list[TransfermarktMarketValue] = []

    with httpx.Client(headers=headers, timeout=timeout_seconds, follow_redirects=True) as client:
        for target in targets:
            LOGGER.info("Scraping Transfermarkt market value for %s", target.team_name)
            source_url = target.url or _resolve_transfermarkt_team_url(client, target)
            response = _transfermarkt_get(client, source_url)
            value = parse_market_value_eur(response.text)
            results.append(
                TransfermarktMarketValue(
                    team_id=target.team_id,
                    team_name=target.team_name,
                    total_market_value_eur=value,
                    source_url=source_url,
                    scraped_at=datetime.now(UTC),
                )
            )

    return results


def parse_market_value_eur(html_or_text: str) -> float:
    """Extract the largest Euro-denominated market value from HTML or text."""
    soup = BeautifulSoup(html_or_text, "lxml")
    text = soup.get_text(" ", strip=True)
    values = [_parse_euro_match(match) for match in EURO_VALUE_PATTERN.finditer(text)]
    values = [value for value in values if value is not None]
    if not values:
        raise ValueError("Could not find a Euro market value in the Transfermarkt page.")
    return max(values)


def persist_transfermarkt_market_values(
    values: Sequence[TransfermarktMarketValue],
    *,
    db_path: Path = DB_PATH,
    source_file: Path | None = None,
) -> LoadResult:
    """Upsert Transfermarkt market values into ``d_teams``."""
    initialize_database(db_path=db_path, load_raw_files=False)
    source_file_label = str(source_file) if source_file is not None else "transfermarkt"

    with duckdb.connect(str(db_path)) as con:
        rows = [
            (
                value.team_id,
                value.team_name,
                value.total_market_value_eur,
                value.total_market_value_eur,
                "EUR",
                value.source_url,
                value.scraped_at,
                source_file_label,
            )
            for value in values
        ]
        con.executemany(
            """
            INSERT INTO d_teams (
                team_id,
                team_name,
                country,
                confederation,
                market_value_eur,
                total_market_value_eur,
                market_value_currency,
                market_value_source_url,
                market_value_scraped_at,
                squad_size,
                coach_name,
                source_file,
                loaded_at
            ) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?, NULL, NULL, ?, current_timestamp)
            ON CONFLICT (team_id) DO UPDATE SET
                team_name = excluded.team_name,
                market_value_eur = excluded.market_value_eur,
                total_market_value_eur = excluded.total_market_value_eur,
                market_value_currency = excluded.market_value_currency,
                market_value_source_url = excluded.market_value_source_url,
                market_value_scraped_at = excluded.market_value_scraped_at,
                source_file = excluded.source_file,
                loaded_at = excluded.loaded_at
            """,
            rows,
        )

    return LoadResult("transfermarkt_market_values", Path(source_file_label), len(values))


def write_transfermarkt_raw(
    values: Sequence[TransfermarktMarketValue],
    output_path: Path,
) -> Path:
    """Persist scraped Transfermarkt values as JSON Lines for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "team_id": value.team_id,
                "team_name": value.team_name,
                "total_market_value_eur": value.total_market_value_eur,
                "source_url": value.source_url,
                "scraped_at": value.scraped_at.isoformat(),
            },
            sort_keys=True,
        )
        for value in values
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def read_transfermarkt_raw(path: Path) -> list[TransfermarktMarketValue]:
    """Read Transfermarkt JSON Lines produced by ``write_transfermarkt_raw``."""
    values: list[TransfermarktMarketValue] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            values.append(
                TransfermarktMarketValue(
                    team_id=_read_manifest_str(payload, "team_id", line_number),
                    team_name=_read_manifest_str(payload, "team_name", line_number),
                    total_market_value_eur=float(payload["total_market_value_eur"]),
                    source_url=_read_manifest_str(payload, "source_url", line_number),
                    scraped_at=datetime.fromisoformat(
                        _read_manifest_str(payload, "scraped_at", line_number)
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid Transfermarkt raw row at line {line_number}: {path}"
            ) from exc
    return values


def run_downloads(args: argparse.Namespace) -> list[Path]:
    """Download requested sources into ``data/raw`` without loading DuckDB."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    raw_dir = Path(args.raw_dir)
    explicit_sources = args.sources is not None
    requested_sources = set(args.sources or DEFAULT_DOWNLOAD_SOURCES)
    downloaded_paths: list[Path] = []

    if "matches" in requested_sources:
        match_raw_dir = raw_dir / "kaggle" / _dataset_directory_name(MATCH_RESULTS_DATASET)
        if _should_skip_existing_raw("matches", match_raw_dir, args.force_download):
            downloaded_paths.append(match_raw_dir)
        else:
            downloaded_paths.append(
                download_kaggle_dataset(
                    MATCH_RESULTS_DATASET,
                    match_raw_dir,
                    force=args.force_download,
                )
            )

    if "fbref" in requested_sources:
        if not args.fbref_leagues or not args.fbref_seasons:
            args.fbref_leagues, args.fbref_seasons = _resolve_fbref_defaults()
        fbref_raw_dir = raw_dir / "fbref"
        if _should_skip_existing_raw("fbref", fbref_raw_dir, args.force_download):
            downloaded_paths.append(fbref_raw_dir)
        else:
            try:
                downloaded_paths.extend(
                    fetch_fbref_with_soccerdata(
                        leagues=args.fbref_leagues,
                        seasons=args.fbref_seasons,
                        raw_dir=fbref_raw_dir,
                        no_cache=args.fbref_no_cache,
                    )
                )
            except (
                ConnectionError,
                RuntimeError,
                ValueError,
                requests.exceptions.RequestException,
            ) as exc:
                if explicit_sources:
                    raise
                LOGGER.warning(
                    "Skipping FBref download because the source is unavailable with the "
                    "default settings: %s",
                    exc,
                )

    if "squad" in requested_sources:
        if not args.ea_fc_dataset:
            args.ea_fc_dataset = DEFAULT_EA_FC_DATASET
        squad_raw_dir = raw_dir / "kaggle" / _dataset_directory_name(args.ea_fc_dataset)
        if _should_skip_existing_raw("squad", squad_raw_dir, args.force_download):
            downloaded_paths.append(squad_raw_dir)
        else:
            downloaded_paths.append(
                download_kaggle_dataset(
                    args.ea_fc_dataset,
                    squad_raw_dir,
                    force=args.force_download,
                )
            )

    if "transfermarkt" in requested_sources:
        transfermarkt_raw_dir = raw_dir / "transfermarkt"
        if _should_skip_existing_raw("transfermarkt", transfermarkt_raw_dir, args.force_download):
            downloaded_paths.append(_resolve_transfermarkt_raw(raw_dir, None))
        elif args.transfermarkt_manifest is None:
            args.transfermarkt_manifest = DEFAULT_TRANSFERMARKT_MANIFEST
            if not Path(args.transfermarkt_manifest).exists():
                if explicit_sources:
                    raise FileNotFoundError(
                        f"Transfermarkt manifest not found: {args.transfermarkt_manifest}"
                    )
                LOGGER.warning(
                    "Skipping Transfermarkt download because the default manifest does "
                    "not exist: %s",
                    args.transfermarkt_manifest,
                )
                requested_sources.remove("transfermarkt")
            else:
                values = scrape_transfermarkt_market_values(
                    read_transfermarkt_manifest(Path(args.transfermarkt_manifest))
                )
                downloaded_paths.append(
                    write_transfermarkt_raw(
                        values,
                        raw_dir
                        / "transfermarkt"
                        / f"market_values_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl",
                    )
                )
        elif not Path(args.transfermarkt_manifest).exists():
            if explicit_sources:
                raise FileNotFoundError(
                    f"Transfermarkt manifest not found: {args.transfermarkt_manifest}"
                )
            LOGGER.warning(
                "Skipping Transfermarkt download because the default manifest does not exist: %s",
                args.transfermarkt_manifest,
            )
            requested_sources.remove("transfermarkt")
        else:
            values = scrape_transfermarkt_market_values(
                read_transfermarkt_manifest(Path(args.transfermarkt_manifest))
            )
            downloaded_paths.append(
                write_transfermarkt_raw(
                    values,
                    raw_dir
                    / "transfermarkt"
                    / f"market_values_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl",
                )
            )

    if "fjelstul" in requested_sources:
        fjelstul_raw_dir = raw_dir / "fjelstul_worldcup"
        if _should_skip_existing_raw("fjelstul", fjelstul_raw_dir, args.force_download):
            downloaded_paths.append(_resolve_fjelstul_worldcup_raw(raw_dir))
        else:
            downloaded_paths.append(
                download_fjelstul_worldcup_matches(
                    fjelstul_raw_dir,
                    force=args.force_download,
                )
            )

    for path in downloaded_paths:
        LOGGER.info("Raw data available at %s", path)
    return downloaded_paths


def run_loads(args: argparse.Namespace) -> list[LoadResult]:
    """Load requested raw sources from ``data/raw`` into DuckDB."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    db_path = Path(args.db_path)
    raw_dir = Path(args.raw_dir)
    cutoff_date = date.fromisoformat(args.cutoff_date)
    initialize_database(db_path=db_path, raw_dir=raw_dir, load_raw_files=False)

    explicit_sources = args.sources is not None
    requested_sources = set(args.sources or DEFAULT_LOAD_SOURCES)
    results: list[LoadResult] = []

    if "matches" in requested_sources:
        match_raw_dir = raw_dir / "kaggle" / _dataset_directory_name(MATCH_RESULTS_DATASET)
        results.append(
            _run_load_step(
                "historical_matches",
                lambda: load_historical_matches(
                    match_raw_dir,
                    db_path=db_path,
                    cutoff_date=cutoff_date,
                ),
                db_path,
                cutoff_date,
            )
        )

    if "fbref" in requested_sources:
        try:
            results.append(
                _run_load_step(
                    "fbref_match_stats",
                    lambda: load_fbref_match_stats(
                        raw_dir / "fbref",
                        db_path=db_path,
                        cutoff_date=cutoff_date,
                    ),
                    db_path,
                    cutoff_date,
                )
            )
        except FileNotFoundError as exc:
            if explicit_sources:
                raise
            LOGGER.warning("Skipping FBref load because no raw stats are available: %s", exc)

    if "squad" in requested_sources:
        source_dataset_arg = args.ea_fc_dataset or DEFAULT_EA_FC_DATASET
        try:
            squad_raw_dir, source_dataset = _resolve_squad_source(raw_dir, source_dataset_arg)
        except FileNotFoundError as exc:
            if explicit_sources:
                raise
            LOGGER.warning(
                "Skipping squad attributes load because no raw data is available: %s",
                exc,
            )
        else:
            results.append(
                _run_load_step(
                    "squad_attributes",
                    lambda: load_squad_attributes(
                        squad_raw_dir,
                        db_path=db_path,
                        source_season=args.squad_season,
                        source_dataset=source_dataset,
                    ),
                    db_path,
                    cutoff_date,
                )
            )

    if "transfermarkt" in requested_sources:
        try:
            transfermarkt_raw = _resolve_transfermarkt_raw(raw_dir, args.transfermarkt_raw)
        except FileNotFoundError as exc:
            if explicit_sources:
                raise
            LOGGER.warning("Skipping Transfermarkt load because no raw data is available: %s", exc)
        else:
            transfermarkt_values = read_transfermarkt_raw(transfermarkt_raw)
            results.append(
                _run_load_step(
                    "transfermarkt_market_values",
                    lambda: persist_transfermarkt_market_values(
                        transfermarkt_values,
                        db_path=db_path,
                        source_file=transfermarkt_raw,
                    ),
                    db_path,
                    cutoff_date,
                )
            )

    if "fjelstul" in requested_sources:
        try:
            fjelstul_raw = _resolve_fjelstul_worldcup_raw(raw_dir)
        except FileNotFoundError as exc:
            if explicit_sources:
                raise
            LOGGER.warning("Skipping Fjelstul load because no raw data is available: %s", exc)
        else:
            results.append(
                _run_load_step(
                    "fjelstul_world_cup_history",
                    lambda: load_fjelstul_world_cup_history(
                        fjelstul_raw,
                        db_path=db_path,
                    ),
                    db_path,
                    cutoff_date,
                )
            )

    for result in results:
        LOGGER.info(
            "Loaded %s rows from %s into %s",
            result.rows_loaded,
            result.raw_path,
            result.source_name,
        )
    return results


def run_collection(args: argparse.Namespace) -> list[LoadResult]:
    """Run requested collectors and loaders from parsed CLI arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    db_path = Path(args.db_path)
    raw_dir = Path(args.raw_dir)
    cutoff_date = date.fromisoformat(args.cutoff_date)
    initialize_database(db_path=db_path, raw_dir=raw_dir, load_raw_files=False)

    results: list[LoadResult] = []
    requested_sources = set(args.sources)

    if "matches" in requested_sources:
        match_raw_dir = raw_dir / "kaggle" / _dataset_directory_name(MATCH_RESULTS_DATASET)
        if not args.load_existing and not _should_skip_existing_raw(
            "matches", match_raw_dir, args.force_download
        ):
            download_kaggle_dataset(MATCH_RESULTS_DATASET, match_raw_dir, force=args.force_download)
        results.append(
            _run_load_step(
                "historical_matches",
                lambda: load_historical_matches(
                    match_raw_dir,
                    db_path=db_path,
                    cutoff_date=cutoff_date,
                ),
                db_path,
                cutoff_date,
            )
        )

    if "fbref" in requested_sources:
        fbref_raw_dir = raw_dir / "fbref"
        if (
            args.fbref_leagues
            and args.fbref_seasons
            and not args.load_existing
            and not _should_skip_existing_raw("fbref", fbref_raw_dir, args.force_download)
        ):
            fetch_fbref_with_soccerdata(
                leagues=args.fbref_leagues,
                seasons=args.fbref_seasons,
                raw_dir=fbref_raw_dir,
                no_cache=args.fbref_no_cache,
            )
        results.append(
            _run_load_step(
                "fbref_match_stats",
                lambda: load_fbref_match_stats(
                    fbref_raw_dir,
                    db_path=db_path,
                    cutoff_date=cutoff_date,
                ),
                db_path,
                cutoff_date,
            )
        )

    if "squad" in requested_sources:
        if not args.ea_fc_dataset:
            args.ea_fc_dataset = DEFAULT_EA_FC_DATASET
        squad_raw_dir = raw_dir / "kaggle" / _dataset_directory_name(args.ea_fc_dataset)
        if not args.load_existing and not _should_skip_existing_raw(
            "squad", squad_raw_dir, args.force_download
        ):
            download_kaggle_dataset(args.ea_fc_dataset, squad_raw_dir, force=args.force_download)
        results.append(
            _run_load_step(
                "squad_attributes",
                lambda: load_squad_attributes(
                    squad_raw_dir,
                    db_path=db_path,
                    source_season=args.squad_season,
                    source_dataset=args.ea_fc_dataset,
                ),
                db_path,
                cutoff_date,
            )
        )

    if "transfermarkt" in requested_sources:
        transfermarkt_raw_dir = raw_dir / "transfermarkt"
        use_existing_transfermarkt = args.load_existing or _should_skip_existing_raw(
            "transfermarkt", transfermarkt_raw_dir, args.force_download
        )
        if args.transfermarkt_manifest is None and not use_existing_transfermarkt:
            raise ValueError(
                "--transfermarkt-manifest is required when source 'transfermarkt' is selected."
            )
        if not use_existing_transfermarkt:
            manifest_path = Path(args.transfermarkt_manifest)
            values = scrape_transfermarkt_market_values(read_transfermarkt_manifest(manifest_path))
            raw_path = write_transfermarkt_raw(
                values,
                raw_dir
                / "transfermarkt"
                / f"market_values_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl",
            )
        else:
            raw_path = _resolve_transfermarkt_raw(raw_dir, None)
            values = read_transfermarkt_raw(raw_path)
        results.append(
            _run_load_step(
                "transfermarkt_market_values",
                lambda: persist_transfermarkt_market_values(
                    values,
                    db_path=db_path,
                    source_file=raw_path,
                ),
                db_path,
                cutoff_date,
            )
        )

    if "fjelstul" in requested_sources:
        fjelstul_raw_dir = raw_dir / "fjelstul_worldcup"
        if args.load_existing or _should_skip_existing_raw(
            "fjelstul", fjelstul_raw_dir, args.force_download
        ):
            raw_path = _resolve_fjelstul_worldcup_raw(raw_dir)
        else:
            raw_path = download_fjelstul_worldcup_matches(
                fjelstul_raw_dir,
                force=args.force_download,
            )
        results.append(
            _run_load_step(
                "fjelstul_world_cup_history",
                lambda: load_fjelstul_world_cup_history(
                    raw_path,
                    db_path=db_path,
                ),
                db_path,
                cutoff_date,
            )
        )

    for result in results:
        LOGGER.info(
            "Loaded %s rows from %s into %s",
            result.rows_loaded,
            result.raw_path,
            result.source_name,
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the data collection command-line parser."""
    parser = argparse.ArgumentParser(
        description="Collect real football datasets and normalize them into DuckDB.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_COLLECTION_SOURCES,
        default=["matches"],
        help="Sources to collect/load.",
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="DuckDB warehouse path.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw data directory.")
    parser.add_argument(
        "--cutoff-date",
        default=DATA_CUTOFF_DATE.isoformat(),
        help="Minimum match/stat date to load, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Skip network downloads and only load files already present under data/raw.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force Kaggle to redownload source files.",
    )
    parser.add_argument(
        "--fbref-leagues",
        nargs="*",
        default=[],
        help="soccerdata/FBref league aliases to collect.",
    )
    parser.add_argument(
        "--fbref-seasons",
        nargs="*",
        default=[],
        help="soccerdata/FBref seasons to collect.",
    )
    parser.add_argument(
        "--fbref-no-cache",
        action="store_true",
        help="Disable soccerdata cache when fetching FBref data.",
    )
    parser.add_argument(
        "--ea-fc-dataset",
        default="",
        help="Kaggle slug for the EA FC/FIFA player attributes dataset.",
    )
    parser.add_argument(
        "--squad-season",
        default="2025-2026",
        help="Season label to store for aggregated squad attributes.",
    )
    parser.add_argument(
        "--transfermarkt-manifest",
        type=Path,
        help="JSON file listing Transfermarkt team pages to scrape.",
    )
    return parser


def build_download_parser() -> argparse.ArgumentParser:
    """Build the raw data download command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download all configured raw football datasets into data/raw.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_COLLECTION_SOURCES,
        default=None,
        help="Sources to download. Defaults to the project ingestion defaults.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw data directory.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force Kaggle to redownload source files.",
    )
    parser.add_argument(
        "--fbref-leagues",
        nargs="*",
        default=[],
        help="soccerdata/FBref league aliases to collect.",
    )
    parser.add_argument(
        "--fbref-seasons",
        nargs="*",
        default=[],
        help="soccerdata/FBref seasons to collect.",
    )
    parser.add_argument(
        "--fbref-no-cache",
        action="store_true",
        help="Disable soccerdata cache when fetching FBref data.",
    )
    parser.add_argument(
        "--ea-fc-dataset",
        default="",
        help="Kaggle slug for the EA FC/FIFA player attributes dataset.",
    )
    parser.add_argument(
        "--transfermarkt-manifest",
        type=Path,
        help="JSON file listing Transfermarkt team pages to scrape.",
    )
    return parser


def build_load_parser() -> argparse.ArgumentParser:
    """Build the raw-to-warehouse loading command-line parser."""
    parser = argparse.ArgumentParser(
        description="Load all configured raw football datasets into DuckDB.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_COLLECTION_SOURCES,
        default=None,
        help="Sources to load. Defaults to the project ingestion defaults.",
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="DuckDB warehouse path.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw data directory.")
    parser.add_argument(
        "--cutoff-date",
        default=DATA_CUTOFF_DATE.isoformat(),
        help="Minimum match/stat date to load, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--ea-fc-dataset",
        default="",
        help=(
            "Kaggle slug for the EA FC/FIFA player attributes dataset. If omitted, "
            "load-data tries to infer the single compatible squad directory."
        ),
    )
    parser.add_argument(
        "--squad-season",
        default="2025-2026",
        help="Season label to store for aggregated squad attributes.",
    )
    parser.add_argument(
        "--transfermarkt-raw",
        type=Path,
        help="Transfermarkt JSONL file to load. Defaults to the newest data/raw file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    try:
        run_collection(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def download_main(argv: Sequence[str] | None = None) -> int:
    """Download-only CLI entrypoint."""
    args = build_download_parser().parse_args(argv)
    try:
        run_downloads(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def load_main(argv: Sequence[str] | None = None) -> int:
    """Load-only CLI entrypoint."""
    args = build_load_parser().parse_args(argv)
    try:
        run_loads(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_load_step(
    source_name: str,
    load_step: Callable[[], LoadResult],
    db_path: Path,
    cutoff_date: date,
) -> LoadResult:
    started_at = datetime.now(UTC)
    run_id = uuid4().hex
    try:
        result = load_step()
    except Exception as exc:
        _record_collection_run(
            db_path=db_path,
            run_id=run_id,
            source_name=source_name,
            source_kind="loader",
            status="failed",
            raw_path=None,
            rows_loaded=0,
            cutoff_date=cutoff_date,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            error_message=str(exc),
        )
        raise

    _record_collection_run(
        db_path=db_path,
        run_id=run_id,
        source_name=source_name,
        source_kind="loader",
        status="loaded",
        raw_path=result.raw_path,
        rows_loaded=result.rows_loaded,
        cutoff_date=cutoff_date,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        error_message=None,
    )
    return result


def _record_collection_run(
    *,
    db_path: Path,
    run_id: str,
    source_name: str,
    source_kind: str,
    status: str,
    raw_path: Path | None,
    rows_loaded: int,
    cutoff_date: date,
    started_at: datetime,
    finished_at: datetime,
    error_message: str | None,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO data_collection_runs (
                run_id,
                source_name,
                source_kind,
                status,
                raw_path,
                rows_loaded,
                cutoff_date,
                started_at,
                finished_at,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                source_name,
                source_kind,
                status,
                str(raw_path) if raw_path is not None else None,
                rows_loaded,
                cutoff_date,
                started_at,
                finished_at,
                error_message,
            ],
        )


def _create_kaggle_client() -> KaggleDatasetClient:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:
        raise RuntimeError("The Kaggle package is not installed.") from exc

    return KaggleApi()


def _resolve_fbref_defaults() -> tuple[list[str], list[str]]:
    if DEFAULT_FBREF_MANIFEST.exists():
        payload = json.loads(DEFAULT_FBREF_MANIFEST.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"FBref manifest must be an object: {DEFAULT_FBREF_MANIFEST}")
        leagues = payload.get("leagues")
        seasons = payload.get("seasons")
        if not isinstance(leagues, list) or not all(isinstance(item, str) for item in leagues):
            raise ValueError("FBref manifest must contain a string list field: leagues")
        if not isinstance(seasons, list) or not all(
            isinstance(item, str | int) for item in seasons
        ):
            raise ValueError("FBref manifest must contain a string/integer list field: seasons")
        return [str(item) for item in leagues], [str(item) for item in seasons]

    return list(DEFAULT_FBREF_LEAGUES), list(DEFAULT_FBREF_SEASONS)


def _resolve_transfermarkt_team_url(
    client: httpx.Client,
    target: TransfermarktTeamTarget,
) -> str:
    query = target.search_query or target.team_name
    response = _transfermarkt_get(
        client,
        "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche",
        params={"query": query},
    )
    soup = BeautifulSoup(response.text, "lxml")
    candidates: list[tuple[str, str]] = []

    for anchor in soup.select('a[href*="/startseite/verein/"]'):
        label = anchor.get_text(" ", strip=True)
        href = anchor.get("href")
        if not label or not href or "vereinslos" in href or "unbekannt" in href:
            continue
        if re.search(r"\bU\d{2}\b|Olympic Team", label, flags=re.IGNORECASE):
            continue
        candidates.append((label, href))

    expected_labels = {
        target.team_name.casefold(),
        target.team_id.casefold(),
        query.casefold(),
    }
    for label, href in candidates:
        if label.casefold() in expected_labels:
            return _absolute_transfermarkt_url(href)

    for label, href in candidates:
        normalized_label = label.casefold()
        if target.team_name.casefold() in normalized_label or query.casefold() in normalized_label:
            return _absolute_transfermarkt_url(href)

    if candidates:
        label, href = candidates[0]
        LOGGER.warning(
            "Using first Transfermarkt search result for %s: %s",
            target.team_name,
            label,
        )
        return _absolute_transfermarkt_url(href)

    raise ValueError(f"Could not resolve Transfermarkt URL for {target.team_name}.")


def _transfermarkt_get(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    attempts: int = 3,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            status_code = (
                exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            )
            if status_code is not None and status_code < 500 and status_code != 429:
                break
            if attempt < attempts:
                time.sleep(float(attempt * 2))

    raise RuntimeError(
        f"Transfermarkt request failed after {attempts} attempts: {url}"
    ) from last_error


def _absolute_transfermarkt_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return f"https://www.transfermarkt.com{href}"


def _resolve_squad_source(raw_dir: Path, dataset: str) -> tuple[Path, str]:
    if dataset:
        _validate_kaggle_dataset_slug(dataset)
        return raw_dir / "kaggle" / _dataset_directory_name(dataset), dataset

    kaggle_root = raw_dir / "kaggle"
    if not kaggle_root.exists():
        raise FileNotFoundError(
            "No Kaggle raw directory found. Run download-data with --ea-fc-dataset first."
        )

    match_dir_name = _dataset_directory_name(MATCH_RESULTS_DATASET)
    candidates: list[Path] = []
    for path in sorted(candidate for candidate in kaggle_root.iterdir() if candidate.is_dir()):
        if path.name == match_dir_name:
            continue
        try:
            _find_tabular_file(
                path,
                required_alias_groups=(
                    ("nationality", "nation", "team", "country"),
                    ("overall", "ova", "ovr"),
                ),
            )
        except FileNotFoundError:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No compatible EA FC/FIFA raw directory found. Pass --ea-fc-dataset "
            "or run download-data for the squad source first."
        )
    if len(candidates) > 1:
        labels = ", ".join(candidate.name for candidate in candidates)
        raise ValueError(
            "Multiple compatible squad datasets found. Pass --ea-fc-dataset to choose one: "
            f"{labels}"
        )

    return candidates[0], candidates[0].name.replace("__", "/")


def _resolve_transfermarkt_raw(raw_dir: Path, transfermarkt_raw: Path | None) -> Path:
    if transfermarkt_raw is not None:
        if not transfermarkt_raw.exists():
            raise FileNotFoundError(f"Transfermarkt raw file not found: {transfermarkt_raw}")
        return transfermarkt_raw

    transfermarkt_dir = raw_dir / "transfermarkt"
    candidates = sorted(
        (
            path
            for path in transfermarkt_dir.glob("market_values_*.jsonl")
            if _has_existing_file(path)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No Transfermarkt JSONL raw file found. Run download-data with "
            "--transfermarkt-manifest first."
        )
    return candidates[0]


def _resolve_fjelstul_worldcup_raw(raw_dir: Path) -> Path:
    raw_file = raw_dir / "fjelstul_worldcup" / FJELSTUL_WORLDCUP_RAW_FILENAME
    if not _has_existing_file(raw_file):
        raise FileNotFoundError(
            "No Fjelstul World Cup raw CSV found. Run download-data with "
            "--sources fjelstul first."
        )
    return raw_file


def _resolve_optional_fjelstul_bookings_raw(source_root: Path) -> Path | None:
    if source_root.is_file():
        raw_file = source_root.parent / FJELSTUL_WORLDCUP_BOOKINGS_RAW_FILENAME
    else:
        raw_file = source_root / FJELSTUL_WORLDCUP_BOOKINGS_RAW_FILENAME
    return raw_file if _has_existing_file(raw_file) else None


def _should_skip_existing_raw(source_name: str, raw_path: Path, force_download: bool) -> bool:
    if force_download:
        return False

    if not _source_raw_exists(source_name, raw_path):
        return False

    LOGGER.info(
        "Skipping %s download because raw data already exists under %s",
        source_name,
        raw_path,
    )
    return True


def _source_raw_exists(source_name: str, raw_path: Path) -> bool:
    if source_name in {"matches", "squad"}:
        return _has_existing_raw_files(raw_path)
    if source_name == "fbref":
        return all(
            _has_existing_file(raw_path / filename)
            for filename in ("fbref_schedule.parquet", "fbref_team_stats.parquet")
        )
    if source_name == "transfermarkt":
        return any(
            _has_existing_file(path)
            for path in raw_path.glob("market_values_*.jsonl")
            if path.is_file()
        )
    if source_name == "fjelstul":
        return all(
            _has_existing_file(raw_path / filename)
            for filename in (
                FJELSTUL_WORLDCUP_RAW_FILENAME,
                FJELSTUL_WORLDCUP_BOOKINGS_RAW_FILENAME,
            )
        )
    raise ValueError(f"Unsupported raw source: {source_name}")


def _has_existing_raw_files(raw_path: Path) -> bool:
    if not raw_path.exists():
        return False
    return any(
        _has_existing_file(path)
        for path in raw_path.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )


def _has_existing_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _validate_kaggle_dataset_slug(dataset: str) -> None:
    if dataset in {"owner/dataset", "<owner/dataset>"}:
        raise ValueError(
            "--ea-fc-dataset must be a real Kaggle dataset slug, not the README "
            "placeholder. Example: flynn28/eafc26-player-database"
        )

    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", dataset):
        raise ValueError(
            "Kaggle dataset must use the '<owner>/<dataset>' format. "
            "Open a Kaggle dataset page and copy the path after /datasets/."
        )


def _kaggle_download_error_message(dataset: str, exc: requests.exceptions.HTTPError) -> str:
    status_code = exc.response.status_code if exc.response is not None else None
    if status_code == 403:
        return (
            f"Kaggle refused access to dataset '{dataset}' with HTTP 403. "
            "Check that the slug is real, your Kaggle credentials are active, "
            "and your account can access the dataset page. If Kaggle shows terms "
            "or a license prompt for that dataset, accept it in the browser first."
        )
    if status_code == 404:
        return (
            f"Kaggle dataset '{dataset}' was not found. Open the dataset page in "
            "the browser and copy the '<owner>/<dataset>' slug from the URL."
        )
    return f"Failed to download Kaggle dataset '{dataset}': {exc}"


def _write_soccerdata_frame(frame: object, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(frame, "to_parquet"):
        frame.to_parquet(output_path, index=False)
        return output_path
    if hasattr(frame, "write_parquet"):
        frame.write_parquet(output_path)
        return output_path
    raise TypeError("soccerdata returned an unsupported frame object.")


def _extract_nested_archives(directory: Path) -> None:
    for archive_path in directory.rglob("*.zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(archive_path.with_suffix(""))


def _find_tabular_file(
    source_root: Path,
    *,
    required_alias_groups: tuple[tuple[str, ...], ...],
) -> Path:
    candidates = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV or Parquet files found under {source_root}.")

    with duckdb.connect() as con:
        for candidate in candidates:
            try:
                lookup = _column_lookup(_source_columns(con, candidate))
            except duckdb.Error:
                LOGGER.warning("Skipping unreadable source file: %s", candidate)
                continue
            if all(any(alias in lookup for alias in aliases) for aliases in required_alias_groups):
                return candidate

    required = [" | ".join(group) for group in required_alias_groups]
    raise FileNotFoundError(
        f"No source file under {source_root} has required columns: {', '.join(required)}"
    )


def _source_columns(con: duckdb.DuckDBPyConnection, raw_file: Path) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {_source_relation_sql(raw_file)}").fetchall()
    return [str(row[0]) for row in rows]


def _source_relation_sql(raw_file: Path) -> str:
    file_path = _sql_string_literal(raw_file.as_posix())
    if raw_file.suffix.lower() == ".parquet":
        return f"read_parquet({file_path})"
    if raw_file.suffix.lower() == ".csv":
        return (
            "read_csv_auto("
            f"{file_path}, header=true, nullstr=['', 'NA', 'N/A', 'NULL'], "
            "sample_size=-1, ignore_errors=false"
            ")"
        )
    raise ValueError(f"Unsupported raw file extension: {raw_file.suffix}")


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {_normalize_column_name(column): column for column in columns}


def _required_column(column_lookup: Mapping[str, str], aliases: tuple[str, ...]) -> str:
    column = _optional_column(column_lookup, aliases)
    if column is None:
        raise KeyError(f"Missing required column. Accepted aliases: {', '.join(aliases)}")
    return column


def _optional_column(column_lookup: Mapping[str, str], aliases: tuple[str, ...]) -> str | None:
    return next(
        (
            column_lookup[_normalize_column_name(alias)]
            for alias in aliases
            if _normalize_column_name(alias) in column_lookup
        ),
        None,
    )


def _ensure_teams_from_query(
    con: duckdb.DuckDBPyConnection,
    source_sql: str,
    team_expression: str,
    source_file: Path,
    *,
    where_sql: str | None = None,
) -> None:
    source_file_expr = _sql_string_literal(f"{source_file}#auto-team")
    filter_sql = f"AND {where_sql}" if where_sql is not None else ""
    con.execute(
        f"""
        INSERT INTO d_teams (
            team_id,
            team_name,
            country,
            confederation,
            market_value_eur,
            total_market_value_eur,
            market_value_currency,
            market_value_source_url,
            market_value_scraped_at,
            squad_size,
            coach_name,
            source_file,
            loaded_at
        )
        SELECT DISTINCT
            {team_expression} AS team_id,
            {team_expression} AS team_name,
            NULL::VARCHAR AS country,
            NULL::VARCHAR AS confederation,
            NULL::DOUBLE AS market_value_eur,
            NULL::DOUBLE AS total_market_value_eur,
            NULL::VARCHAR AS market_value_currency,
            NULL::VARCHAR AS market_value_source_url,
            NULL::TIMESTAMP AS market_value_scraped_at,
            NULL::INTEGER AS squad_size,
            NULL::VARCHAR AS coach_name,
            {source_file_expr} AS source_file,
            current_timestamp AS loaded_at
        FROM {source_sql}
        WHERE {team_expression} IS NOT NULL
          {filter_sql}
        ON CONFLICT (team_id) DO NOTHING
        """,
    )


def _delete_unreferenced_auto_teams(con: duckdb.DuckDBPyConnection, source_file: Path) -> None:
    source_file_label = f"{source_file}#auto-team"
    con.execute(
        """
        DELETE FROM d_teams AS t
        WHERE t.source_file = ?
          AND NOT EXISTS (
              SELECT 1
              FROM f_matches AS m
              WHERE m.home_team_id = t.team_id OR m.away_team_id = t.team_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM f_match_stats AS s
              WHERE s.team_id = t.team_id OR s.opponent_team_id = t.team_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM d_squad_attributes AS a
              WHERE a.team_id = t.team_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM d_squads AS q
              WHERE q.team_id = t.team_id
          )
        """,
        [source_file_label],
    )


def _numeric_expression(column_name: str) -> str:
    identifier = _quote_identifier(column_name)
    return f"TRY_CAST(REPLACE(CAST({identifier} AS VARCHAR), '%', '') AS DOUBLE)"


def _parse_euro_match(match: re.Match[str]) -> float | None:
    raw_number = match.group("number")
    if not raw_number:
        return None
    number = float(raw_number.replace(",", "."))
    unit = (match.group("unit") or "").casefold()
    multiplier = {
        "bn": 1_000_000_000.0,
        "m": 1_000_000.0,
        "k": 1_000.0,
        "": 1.0,
    }[unit]
    return number * multiplier


def _read_manifest_str(payload: Mapping[str, object], field_name: str, index: int) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest team at index {index} needs non-empty '{field_name}'.")
    return value.strip()


def _read_optional_manifest_str(payload: Mapping[str, object], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field '{field_name}' must be a non-empty string.")
    return value.strip()


def _dataset_directory_name(dataset: str) -> str:
    return dataset.replace("/", "__")


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column_name.casefold()).strip("_")


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_date_literal(value: date) -> str:
    return f"DATE {_sql_string_literal(value.isoformat())}"


if __name__ == "__main__":
    raise SystemExit(main())
