"""Load national-team ratings from World Football Elo Ratings.

The default source is the public TSV data used by https://www.eloratings.net.
Only the current global snapshot is persisted; World Cup Probability Elo remains
the chronological match-by-match rating calculated by this project.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import duckdb
import httpx

try:
    from .db_init import DB_PATH, RAW_DIR, initialize_database
except ImportError:  # pragma: no cover - supports direct script execution.
    from db_init import DB_PATH, RAW_DIR, initialize_database

LOGGER = logging.getLogger(__name__)

ELORATINGS_BASE_URL = "https://www.eloratings.net"
DEFAULT_WORLD_RATINGS_URL = f"{ELORATINGS_BASE_URL}/World.tsv"
DEFAULT_TEAM_DICTIONARY_URL = f"{ELORATINGS_BASE_URL}/en.teams.tsv"
DEFAULT_RAW_PATH = RAW_DIR / "eloratings" / "world_football_elo_ratings_snapshot.jsonl"
USER_AGENT = "world-cup-probability/0.1 (+https://localhost)"

MANUAL_TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "Cape Verde": ("Cabo Verde",),
    "Czechia": ("Czech Republic",),
    "DR Congo": ("Congo DR", "Democratic Republic of the Congo"),
    "Iran": ("IR Iran",),
    "Ivory Coast": ("Cote d'Ivoire", "Côte d'Ivoire"),
    "South Korea": ("Korea Republic", "Republic of Korea"),
    "Turkey": ("Turkiye", "Türkiye"),
    "United States": ("United States of America",),
}


@dataclass(frozen=True, slots=True)
class WorldFootballEloRatingsRow:
    """One team rating from the World Football Elo Ratings snapshot."""

    world_football_team_code: str
    team_name: str
    elo_rank: int
    elo_rating: float
    rating_date: date | None
    source_url: str


@dataclass(frozen=True, slots=True)
class WorldFootballEloRatingsAlias:
    """One local-name matching alias for a World Football Elo Ratings team code."""

    world_football_team_code: str
    team_alias: str

    @property
    def team_alias_key(self) -> str:
        return self.team_alias.casefold()

    @property
    def normalized_team_alias(self) -> str:
        return normalize_team_name(self.team_alias)


@dataclass(frozen=True, slots=True)
class WorldFootballEloRatingsSnapshot:
    """Parsed World Football Elo Ratings rows plus matching aliases."""

    ratings: tuple[WorldFootballEloRatingsRow, ...]
    aliases: tuple[WorldFootballEloRatingsAlias, ...]


def fetch_world_football_elo_ratings_snapshot(
    *,
    ratings_url: str = DEFAULT_WORLD_RATINGS_URL,
    team_dictionary_url: str = DEFAULT_TEAM_DICTIONARY_URL,
) -> WorldFootballEloRatingsSnapshot:
    """Fetch and parse the current World Football Elo Ratings snapshot."""
    with httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        ratings_response = client.get(ratings_url)
        ratings_response.raise_for_status()
        teams_response = client.get(team_dictionary_url)
        teams_response.raise_for_status()

    rating_date = _rating_date_from_headers(ratings_response.headers)
    team_dictionary = parse_team_dictionary(teams_response.text)
    return parse_world_ratings(
        ratings_response.text,
        team_dictionary=team_dictionary,
        rating_date=rating_date,
        source_url=ratings_url,
    )


def parse_team_dictionary(tsv_text: str) -> dict[str, tuple[str, ...]]:
    """Parse ``en.teams.tsv`` into code -> aliases."""
    teams: dict[str, tuple[str, ...]] = {}
    for line in tsv_text.splitlines():
        fields = [field.strip() for field in line.split("\t") if field.strip()]
        if len(fields) < 2:
            continue
        code = fields[0]
        if code.endswith("_loc"):
            continue
        aliases = tuple(dict.fromkeys(fields[1:]))
        teams[code] = aliases
    return teams


def parse_world_ratings(
    tsv_text: str,
    *,
    team_dictionary: dict[str, tuple[str, ...]],
    rating_date: date | None,
    source_url: str = DEFAULT_WORLD_RATINGS_URL,
) -> WorldFootballEloRatingsSnapshot:
    """Parse a global rating TSV into normalized records."""
    ratings: list[WorldFootballEloRatingsRow] = []
    aliases: list[WorldFootballEloRatingsAlias] = []

    for line_number, line in enumerate(tsv_text.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) < 4:
            continue

        team_code = fields[2].strip()
        team_aliases = team_dictionary.get(team_code)
        if not team_aliases:
            raise ValueError(
                f"Unknown World Football Elo Ratings team code at line {line_number}: {team_code}"
            )

        team_name = team_aliases[0]
        ratings.append(
            WorldFootballEloRatingsRow(
                world_football_team_code=team_code,
                team_name=team_name,
                elo_rank=_parse_int(fields[1], line_number=line_number, field_name="rank"),
                elo_rating=float(fields[3]),
                rating_date=rating_date,
                source_url=source_url,
            )
        )

        for alias in _aliases_for_team(team_name, team_aliases):
            aliases.append(
                WorldFootballEloRatingsAlias(
                    world_football_team_code=team_code,
                    team_alias=alias,
                )
            )

    if not ratings:
        raise ValueError("No World Football Elo Ratings rows were parsed.")

    return WorldFootballEloRatingsSnapshot(
        ratings=tuple(ratings),
        aliases=tuple(_dedupe_aliases(aliases)),
    )


def write_world_football_elo_ratings_raw(
    snapshot: WorldFootballEloRatingsSnapshot, raw_path: Path = DEFAULT_RAW_PATH
) -> Path:
    """Persist a parsed snapshot as JSONL for reproducible offline loading."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        for rating in snapshot.ratings:
            handle.write(json.dumps({"type": "rating", **_rating_to_json(rating)}) + "\n")
        for alias in snapshot.aliases:
            handle.write(json.dumps({"type": "alias", **_alias_to_json(alias)}) + "\n")
    return raw_path


def read_world_football_elo_ratings_raw(raw_path: Path) -> WorldFootballEloRatingsSnapshot:
    """Read a JSONL snapshot previously written by ``write_world_football_elo_ratings_raw``."""
    ratings: list[WorldFootballEloRatingsRow] = []
    aliases: list[WorldFootballEloRatingsAlias] = []

    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                row_type = payload["type"]
                if row_type == "rating":
                    ratings.append(
                        WorldFootballEloRatingsRow(
                            world_football_team_code=str(payload["world_football_team_code"]),
                            team_name=str(payload["team_name"]),
                            elo_rank=int(payload["elo_rank"]),
                            elo_rating=float(payload["elo_rating"]),
                            rating_date=_optional_date(payload.get("rating_date")),
                            source_url=str(payload["source_url"]),
                        )
                    )
                elif row_type == "alias":
                    aliases.append(
                        WorldFootballEloRatingsAlias(
                            world_football_team_code=str(payload["world_football_team_code"]),
                            team_alias=str(payload["team_alias"]),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported row type: {row_type}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid World Football Elo Ratings raw row at line {line_number}."
                ) from exc

    if not ratings:
        raise ValueError(f"No World Football Elo Ratings rows found in {raw_path}.")
    if not aliases:
        raise ValueError(f"No World Football Elo Ratings aliases found in {raw_path}.")
    return WorldFootballEloRatingsSnapshot(ratings=tuple(ratings), aliases=tuple(aliases))


def persist_world_football_elo_ratings_snapshot(
    snapshot: WorldFootballEloRatingsSnapshot,
    *,
    db_path: Path = DB_PATH,
    source_file: Path | None = None,
) -> int:
    """Upsert World Football Elo Ratings rows and aliases into DuckDB."""
    initialize_database(db_path=db_path, load_raw_files=False)
    source_file_value = str(source_file) if source_file is not None else None

    with duckdb.connect(str(db_path)) as con:
        con.executemany(
            """
            INSERT INTO d_world_football_elo_ratings (
                world_football_team_code,
                team_name,
                elo_rank,
                elo_rating,
                rating_date,
                source_url,
                source_file,
                loaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (world_football_team_code) DO UPDATE SET
                team_name = excluded.team_name,
                elo_rank = excluded.elo_rank,
                elo_rating = excluded.elo_rating,
                rating_date = excluded.rating_date,
                source_url = excluded.source_url,
                source_file = excluded.source_file,
                loaded_at = now()
            """,
            [
                [
                    rating.world_football_team_code,
                    rating.team_name,
                    rating.elo_rank,
                    rating.elo_rating,
                    rating.rating_date,
                    rating.source_url,
                    source_file_value,
                ]
                for rating in snapshot.ratings
            ],
        )
        con.executemany(
            """
            INSERT INTO d_world_football_elo_team_aliases (
                team_alias_key,
                team_alias,
                normalized_team_alias,
                world_football_team_code,
                source_file,
                loaded_at
            ) VALUES (?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (team_alias_key) DO UPDATE SET
                team_alias = excluded.team_alias,
                normalized_team_alias = excluded.normalized_team_alias,
                world_football_team_code = excluded.world_football_team_code,
                source_file = excluded.source_file,
                loaded_at = now()
            """,
            [
                [
                    alias.team_alias_key,
                    alias.team_alias,
                    alias.normalized_team_alias,
                    alias.world_football_team_code,
                    source_file_value,
                ]
                for alias in snapshot.aliases
            ],
        )

    return len(snapshot.ratings)


def load_world_football_elo_ratings(
    *,
    db_path: Path = DB_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    load_existing: bool = False,
    ratings_url: str = DEFAULT_WORLD_RATINGS_URL,
    team_dictionary_url: str = DEFAULT_TEAM_DICTIONARY_URL,
) -> int:
    """Fetch or read a snapshot, then persist it into DuckDB."""
    if load_existing:
        snapshot = read_world_football_elo_ratings_raw(raw_path)
    else:
        snapshot = fetch_world_football_elo_ratings_snapshot(
            ratings_url=ratings_url,
            team_dictionary_url=team_dictionary_url,
        )
        write_world_football_elo_ratings_raw(snapshot, raw_path)

    rows_loaded = persist_world_football_elo_ratings_snapshot(
        snapshot, db_path=db_path, source_file=raw_path
    )
    LOGGER.info("Loaded %d World Football Elo Ratings rows into DuckDB.", rows_loaded)
    return rows_loaded


def normalize_team_name(value: str) -> str:
    """Normalize a team name for loose matching."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Load World Football Elo Ratings from eloratings.net."
    )
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
        "--ratings-url",
        default=DEFAULT_WORLD_RATINGS_URL,
        help="World Football Elo Ratings TSV URL.",
    )
    parser.add_argument(
        "--team-dictionary-url",
        default=DEFAULT_TEAM_DICTIONARY_URL,
        help="World Football Elo Ratings team dictionary TSV URL.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        load_world_football_elo_ratings(
            db_path=args.db_path,
            raw_path=args.raw_path,
            load_existing=args.load_existing,
            ratings_url=args.ratings_url,
            team_dictionary_url=args.team_dictionary_url,
        )
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _aliases_for_team(team_name: str, source_aliases: tuple[str, ...]) -> tuple[str, ...]:
    aliases = [team_name, *source_aliases, *MANUAL_TEAM_ALIASES.get(team_name, ())]
    return tuple(alias for alias in dict.fromkeys(aliases) if alias)


def _dedupe_aliases(
    aliases: list[WorldFootballEloRatingsAlias],
) -> list[WorldFootballEloRatingsAlias]:
    deduped: dict[str, WorldFootballEloRatingsAlias] = {}
    for alias in aliases:
        deduped[alias.team_alias_key] = alias
    return list(deduped.values())


def _parse_int(value: str, *, line_number: int, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name} at line {line_number}: {value}") from exc


def _rating_date_from_headers(headers: httpx.Headers) -> date | None:
    last_modified = headers.get("last-modified")
    if not last_modified:
        return datetime.now(UTC).date()
    return parsedate_to_datetime(last_modified).date()


def _optional_date(value: object) -> date | None:
    if value in {None, ""}:
        return None
    return date.fromisoformat(str(value))


def _rating_to_json(rating: WorldFootballEloRatingsRow) -> dict[str, object]:
    return {
        "world_football_team_code": rating.world_football_team_code,
        "team_name": rating.team_name,
        "elo_rank": rating.elo_rank,
        "elo_rating": rating.elo_rating,
        "rating_date": rating.rating_date.isoformat() if rating.rating_date else None,
        "source_url": rating.source_url,
    }


def _alias_to_json(alias: WorldFootballEloRatingsAlias) -> dict[str, object]:
    return {
        "world_football_team_code": alias.world_football_team_code,
        "team_alias": alias.team_alias,
    }


if __name__ == "__main__":
    raise SystemExit(main())
