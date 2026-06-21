"""Load FIFA World Ranking data for men's national teams."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import httpx

try:
    from .db_init import DB_PATH, RAW_DIR, initialize_database
    from .settings import DEFAULT_SOURCE_USER_AGENT
except ImportError:  # pragma: no cover - supports direct script execution.
    from db_init import DB_PATH, RAW_DIR, initialize_database
    from settings import DEFAULT_SOURCE_USER_AGENT

LOGGER = logging.getLogger(__name__)

DEFAULT_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
DEFAULT_API_URL = "https://api.fifa.com/api/v3/rankings?gender=1&count=300"
DEFAULT_RAW_PATH = RAW_DIR / "fifa_world_ranking" / "men_snapshot.jsonl"
USER_AGENT = DEFAULT_SOURCE_USER_AGENT

MANUAL_TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "Côte d'Ivoire": ("Ivory Coast", "Cote d'Ivoire"),
    "Czechia": ("Czech Republic",),
    "IR Iran": ("Iran",),
    "Korea Republic": ("South Korea", "Republic of Korea"),
    "Türkiye": ("Turkey", "Turkiye"),
    "USA": ("United States", "United States of America"),
}


@dataclass(frozen=True, slots=True)
class FifaWorldRankingRow:
    """One row from FIFA World Ranking."""

    fifa_country_code: str
    fifa_team_id: str | None
    team_name: str
    confederation: str | None
    fifa_rank: int
    previous_rank: int | None
    ranking_points: float
    previous_points: float | None
    ranking_movement: int | None
    matches: int | None
    ranking_date: date
    next_update_date: date | None
    source_url: str
    api_url: str


@dataclass(frozen=True, slots=True)
class FifaWorldRankingAlias:
    """One local-name matching alias for a FIFA World Ranking country code."""

    fifa_country_code: str
    team_alias: str

    @property
    def team_alias_key(self) -> str:
        return self.team_alias.casefold()

    @property
    def normalized_team_alias(self) -> str:
        return normalize_team_name(self.team_alias)


@dataclass(frozen=True, slots=True)
class FifaWorldRankingSnapshot:
    """FIFA World Ranking rows plus aliases."""

    rows: tuple[FifaWorldRankingRow, ...]
    aliases: tuple[FifaWorldRankingAlias, ...]


def fetch_fifa_world_ranking_snapshot(
    *,
    page_url: str = DEFAULT_PAGE_URL,
    api_url: str = DEFAULT_API_URL,
) -> FifaWorldRankingSnapshot:
    """Fetch FIFA World Ranking page metadata and ranking rows."""
    with httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
    ) as client:
        page_response = client.get(page_url)
        page_response.raise_for_status()
        api_response = client.get(api_url)
        api_response.raise_for_status()

    last_update_date, next_update_date = parse_update_dates(page_response.text)
    return parse_fifa_world_ranking_api(
        api_response.json(),
        ranking_date=last_update_date,
        next_update_date=next_update_date,
        source_url=page_url,
        api_url=api_url,
    )


def parse_update_dates(page_html: str) -> tuple[date, date | None]:
    """Parse official update dates from the FIFA ranking page."""
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        page_html,
    )
    if not match:
        raise ValueError("FIFA World Ranking page did not include __NEXT_DATA__.")

    payload = json.loads(match.group(1))
    ranking = payload["props"]["pageProps"]["pageData"]["ranking"]
    last_update_date = _datetime_to_date(ranking["lastUpdateDate"])
    next_update_date = _optional_datetime_to_date(ranking.get("nextUpdateDate"))
    return last_update_date, next_update_date


def parse_fifa_world_ranking_api(
    payload: dict[str, object],
    *,
    ranking_date: date,
    next_update_date: date | None,
    source_url: str = DEFAULT_PAGE_URL,
    api_url: str = DEFAULT_API_URL,
) -> FifaWorldRankingSnapshot:
    """Parse the FIFA rankings API payload."""
    raw_rows = payload.get("Results")
    if not isinstance(raw_rows, list):
        raise ValueError("FIFA World Ranking API payload does not contain Results.")

    rows: list[FifaWorldRankingRow] = []
    aliases: list[FifaWorldRankingAlias] = []

    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, dict):
            raise ValueError(f"Invalid FIFA World Ranking row at index {index}.")
        country_code = str(raw_row["IdCountry"])
        team_name = _localized_team_name(raw_row)
        row_ranking_date = _optional_datetime_to_date(raw_row.get("PubDate")) or ranking_date
        row = FifaWorldRankingRow(
            fifa_country_code=country_code,
            fifa_team_id=_optional_str(raw_row.get("IdTeam")),
            team_name=team_name,
            confederation=_optional_str(raw_row.get("ConfederationName")),
            fifa_rank=int(raw_row["Rank"]),
            previous_rank=_optional_int(raw_row.get("PrevRank")),
            ranking_points=float(raw_row["DecimalTotalPoints"]),
            previous_points=_optional_float(raw_row.get("DecimalPrevPoints")),
            ranking_movement=_optional_int(raw_row.get("RankingMovement")),
            matches=_optional_int(raw_row.get("Matches")),
            ranking_date=row_ranking_date,
            next_update_date=next_update_date,
            source_url=source_url,
            api_url=api_url,
        )
        rows.append(row)

        for alias in _aliases_for_team(team_name, country_code):
            aliases.append(FifaWorldRankingAlias(fifa_country_code=country_code, team_alias=alias))

    if not rows:
        raise ValueError("No FIFA World Ranking rows were parsed.")

    return FifaWorldRankingSnapshot(
        rows=tuple(rows),
        aliases=tuple(_dedupe_aliases(aliases)),
    )


def write_fifa_world_ranking_raw(
    snapshot: FifaWorldRankingSnapshot,
    raw_path: Path = DEFAULT_RAW_PATH,
) -> Path:
    """Persist a FIFA World Ranking snapshot as JSONL."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in snapshot.rows:
            handle.write(json.dumps({"type": "ranking", **_row_to_json(row)}) + "\n")
        for alias in snapshot.aliases:
            handle.write(json.dumps({"type": "alias", **_alias_to_json(alias)}) + "\n")
    return raw_path


def read_fifa_world_ranking_raw(raw_path: Path) -> FifaWorldRankingSnapshot:
    """Read a FIFA World Ranking JSONL snapshot."""
    rows: list[FifaWorldRankingRow] = []
    aliases: list[FifaWorldRankingAlias] = []

    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                row_type = payload["type"]
                if row_type == "ranking":
                    rows.append(_row_from_json(payload))
                elif row_type == "alias":
                    aliases.append(
                        FifaWorldRankingAlias(
                            fifa_country_code=str(payload["fifa_country_code"]),
                            team_alias=str(payload["team_alias"]),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported row type: {row_type}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid FIFA World Ranking raw row at line {line_number}."
                ) from exc

    if not rows:
        raise ValueError(f"No FIFA World Ranking rows found in {raw_path}.")
    if not aliases:
        raise ValueError(f"No FIFA World Ranking aliases found in {raw_path}.")
    return FifaWorldRankingSnapshot(rows=tuple(rows), aliases=tuple(aliases))


def persist_fifa_world_ranking_snapshot(
    snapshot: FifaWorldRankingSnapshot,
    *,
    db_path: Path = DB_PATH,
    source_file: Path | None = None,
) -> int:
    """Upsert FIFA World Ranking rows and aliases into DuckDB."""
    initialize_database(db_path=db_path, load_raw_files=False)
    source_file_value = str(source_file) if source_file is not None else None

    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            """
            INSERT INTO d_fifa_world_ranking (
                fifa_country_code,
                fifa_team_id,
                team_name,
                confederation,
                fifa_rank,
                previous_rank,
                ranking_points,
                previous_points,
                ranking_movement,
                matches,
                ranking_date,
                next_update_date,
                source_url,
                api_url,
                source_file,
                loaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (fifa_country_code) DO UPDATE SET
                fifa_team_id = excluded.fifa_team_id,
                team_name = excluded.team_name,
                confederation = excluded.confederation,
                fifa_rank = excluded.fifa_rank,
                previous_rank = excluded.previous_rank,
                ranking_points = excluded.ranking_points,
                previous_points = excluded.previous_points,
                ranking_movement = excluded.ranking_movement,
                matches = excluded.matches,
                ranking_date = excluded.ranking_date,
                next_update_date = excluded.next_update_date,
                source_url = excluded.source_url,
                api_url = excluded.api_url,
                source_file = excluded.source_file,
                loaded_at = now()
            """,
            [
                [
                    row.fifa_country_code,
                    row.fifa_team_id,
                    row.team_name,
                    row.confederation,
                    row.fifa_rank,
                    row.previous_rank,
                    row.ranking_points,
                    row.previous_points,
                    row.ranking_movement,
                    row.matches,
                    row.ranking_date,
                    row.next_update_date,
                    row.source_url,
                    row.api_url,
                    source_file_value,
                ]
                for row in snapshot.rows
            ],
        )
        con.executemany(
            """
            INSERT INTO d_fifa_world_ranking_team_aliases (
                team_alias_key,
                team_alias,
                normalized_team_alias,
                fifa_country_code,
                source_file,
                loaded_at
            ) VALUES (?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (team_alias_key) DO UPDATE SET
                team_alias = excluded.team_alias,
                normalized_team_alias = excluded.normalized_team_alias,
                fifa_country_code = excluded.fifa_country_code,
                source_file = excluded.source_file,
                loaded_at = now()
            """,
            [
                [
                    alias.team_alias_key,
                    alias.team_alias,
                    alias.normalized_team_alias,
                    alias.fifa_country_code,
                    source_file_value,
                ]
                for alias in snapshot.aliases
            ],
        )

    return len(snapshot.rows)


def load_fifa_world_ranking(
    *,
    db_path: Path = DB_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    load_existing: bool = False,
    force_download: bool = False,
    page_url: str = DEFAULT_PAGE_URL,
    api_url: str = DEFAULT_API_URL,
) -> int:
    """Fetch or read FIFA World Ranking, then persist it into DuckDB."""
    if load_existing or (raw_path.is_file() and raw_path.stat().st_size > 0 and not force_download):
        if not load_existing:
            LOGGER.info(
                "Skipping FIFA World Ranking download because raw data exists at %s",
                raw_path,
            )
        snapshot = read_fifa_world_ranking_raw(raw_path)
    else:
        snapshot = fetch_fifa_world_ranking_snapshot(page_url=page_url, api_url=api_url)
        write_fifa_world_ranking_raw(snapshot, raw_path)

    rows_loaded = persist_fifa_world_ranking_snapshot(
        snapshot,
        db_path=db_path,
        source_file=raw_path,
    )
    LOGGER.info("Loaded %d FIFA World Ranking rows into DuckDB.", rows_loaded)
    return rows_loaded


def normalize_team_name(value: str) -> str:
    """Normalize a team name for loose matching."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Load FIFA World Ranking.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="DuckDB warehouse path.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="JSONL snapshot path for offline/reproducible loads.",
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load the existing JSONL snapshot instead of downloading a fresh one.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download a fresh snapshot even when the JSONL raw file already exists.",
    )
    parser.add_argument("--page-url", default=DEFAULT_PAGE_URL, help="FIFA ranking page URL.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="FIFA rankings API URL.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        load_fifa_world_ranking(
            db_path=args.db_path,
            raw_path=args.raw_path,
            load_existing=args.load_existing,
            force_download=args.force_download,
            page_url=args.page_url,
            api_url=args.api_url,
        )
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _localized_team_name(raw_row: dict[str, object]) -> str:
    team_names = raw_row.get("TeamName")
    if not isinstance(team_names, list):
        raise ValueError("FIFA World Ranking row is missing TeamName.")
    for item in team_names:
        if isinstance(item, dict) and item.get("Locale") == "en-GB" and item.get("Description"):
            return str(item["Description"])
    for item in team_names:
        if isinstance(item, dict) and item.get("Description"):
            return str(item["Description"])
    raise ValueError("FIFA World Ranking row has no localized team name.")


def _aliases_for_team(team_name: str, fifa_country_code: str) -> tuple[str, ...]:
    aliases = [team_name, fifa_country_code, *MANUAL_TEAM_ALIASES.get(team_name, ())]
    return tuple(alias for alias in dict.fromkeys(aliases) if alias)


def _dedupe_aliases(aliases: list[FifaWorldRankingAlias]) -> list[FifaWorldRankingAlias]:
    deduped: dict[str, FifaWorldRankingAlias] = {}
    for alias in aliases:
        deduped[alias.team_alias_key] = alias
    return list(deduped.values())


def _datetime_to_date(value: object) -> date:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _optional_datetime_to_date(value: object) -> date | None:
    if value in {None, ""}:
        return None
    return _datetime_to_date(value)


def _optional_str(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _row_to_json(row: FifaWorldRankingRow) -> dict[str, object]:
    return {
        "fifa_country_code": row.fifa_country_code,
        "fifa_team_id": row.fifa_team_id,
        "team_name": row.team_name,
        "confederation": row.confederation,
        "fifa_rank": row.fifa_rank,
        "previous_rank": row.previous_rank,
        "ranking_points": row.ranking_points,
        "previous_points": row.previous_points,
        "ranking_movement": row.ranking_movement,
        "matches": row.matches,
        "ranking_date": row.ranking_date.isoformat(),
        "next_update_date": row.next_update_date.isoformat() if row.next_update_date else None,
        "source_url": row.source_url,
        "api_url": row.api_url,
    }


def _row_from_json(payload: dict[str, object]) -> FifaWorldRankingRow:
    return FifaWorldRankingRow(
        fifa_country_code=str(payload["fifa_country_code"]),
        fifa_team_id=_optional_str(payload.get("fifa_team_id")),
        team_name=str(payload["team_name"]),
        confederation=_optional_str(payload.get("confederation")),
        fifa_rank=int(payload["fifa_rank"]),
        previous_rank=_optional_int(payload.get("previous_rank")),
        ranking_points=float(payload["ranking_points"]),
        previous_points=_optional_float(payload.get("previous_points")),
        ranking_movement=_optional_int(payload.get("ranking_movement")),
        matches=_optional_int(payload.get("matches")),
        ranking_date=date.fromisoformat(str(payload["ranking_date"])),
        next_update_date=(
            date.fromisoformat(str(payload["next_update_date"]))
            if payload.get("next_update_date")
            else None
        ),
        source_url=str(payload["source_url"]),
        api_url=str(payload["api_url"]),
    )


def _alias_to_json(alias: FifaWorldRankingAlias) -> dict[str, object]:
    return {
        "fifa_country_code": alias.fifa_country_code,
        "team_alias": alias.team_alias,
    }


if __name__ == "__main__":
    raise SystemExit(main())
