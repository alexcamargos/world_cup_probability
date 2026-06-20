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
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import duckdb
import httpx
from bs4 import BeautifulSoup

try:
    from .db_init import DB_PATH, RAW_DIR, initialize_database
except ImportError:  # pragma: no cover - supports direct script execution.
    from db_init import DB_PATH, RAW_DIR, initialize_database

LOGGER = logging.getLogger(__name__)

DATA_CUTOFF_DATE = date(2010, 1, 1)
MATCH_RESULTS_DATASET = "martj42/international-football-results-from-1872-to-2017"
TRANSFERMARKT_USER_AGENT = (
    "Mozilla/5.0 (compatible; world-cup-probability-research/0.1; +https://localhost)"
)

TABULAR_SUFFIXES = {".csv", ".parquet"}
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
    url: str


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
    destination.mkdir(parents=True, exist_ok=True)
    kaggle_client = client if client is not None else _create_kaggle_client()
    kaggle_client.authenticate()
    kaggle_client.dataset_download_files(
        dataset,
        path=str(destination),
        force=force,
        quiet=False,
        unzip=True,
    )
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
        valid_match_filter = (
            f"{date_expr} >= {_sql_date_literal(cutoff_date)} "
            f"AND {home_score_expr} IS NOT NULL "
            f"AND {away_score_expr} IS NOT NULL"
        )

        con.execute("DELETE FROM f_matches WHERE source_file = ?", [str(source_file)])
        _delete_unreferenced_auto_teams(con, source_file)
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
        rows_loaded = int(
            con.execute("SELECT COUNT(*) FROM f_matches WHERE source_file = ?", [str(source_file)])
            .fetchone()[0]
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
            ("overall", "ova"),
        ),
    )

    with duckdb.connect(str(db_path)) as con:
        source_sql = _source_relation_sql(source_file)
        columns = _source_columns(con, source_file)
        lookup = _column_lookup(columns)
        team_col = _required_column(lookup, ("nationality", "nation", "team", "country"))
        overall_col = _required_column(lookup, ("overall", "ova"))
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
                url=_read_manifest_str(raw_team, "url", index),
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
            response = client.get(target.url)
            response.raise_for_status()
            value = parse_market_value_eur(response.text)
            results.append(
                TransfermarktMarketValue(
                    team_id=target.team_id,
                    team_name=target.team_name,
                    total_market_value_eur=value,
                    source_url=target.url,
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
        if not args.load_existing:
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
        if args.fbref_leagues and args.fbref_seasons and not args.load_existing:
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
            raise ValueError("--ea-fc-dataset is required when source 'squad' is selected.")
        squad_raw_dir = raw_dir / "kaggle" / _dataset_directory_name(args.ea_fc_dataset)
        if not args.load_existing:
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
        if args.transfermarkt_manifest is None:
            raise ValueError(
                "--transfermarkt-manifest is required when source 'transfermarkt' is selected."
            )
        manifest_path = Path(args.transfermarkt_manifest)
        values = scrape_transfermarkt_market_values(read_transfermarkt_manifest(manifest_path))
        raw_path = write_transfermarkt_raw(
            values,
            raw_dir / "transfermarkt" / f"market_values_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl",
        )
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
        choices=("matches", "fbref", "squad", "transfermarkt"),
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    run_collection(args)
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
