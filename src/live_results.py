"""Fetch and cache real FIFA World Cup 2026 fixture results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

try:
    from .settings import DEFAULT_SOURCE_USER_AGENT, DEFAULT_WORLD_CUP_RESULTS_CACHE
    from .world_cup_2026_schedule import WorldCupFixture, world_cup_2026_fixtures
except ImportError:  # pragma: no cover - supports direct script execution.
    from settings import DEFAULT_SOURCE_USER_AGENT, DEFAULT_WORLD_CUP_RESULTS_CACHE
    from world_cup_2026_schedule import WorldCupFixture, world_cup_2026_fixtures

FIFA_WORLD_CUP_2026_MATCHES_URL = (
    "https://api.fifa.com/api/v3/calendar/matches"
    "?from=2026-06-11&to=2026-07-20&language=en&count=200"
    "&idCompetition=17&idSeason=285023"
)
DEFAULT_CACHE_TTL = timedelta(minutes=10)
FINISHED_MATCH_GRACE_PERIOD = timedelta(hours=2)


@dataclass(frozen=True, slots=True)
class LiveFixtureScore:
    """Score and realized teams for one FIFA fixture."""

    match_number: int
    home_team_id: str | None
    away_team_id: str | None
    home_goals: int | None
    away_goals: int | None
    match_status: int | None = None
    result_type: int | None = None

    @property
    def has_score(self) -> bool:
        """Return whether both goals are known."""
        return self.home_goals is not None and self.away_goals is not None


@dataclass(frozen=True, slots=True)
class LiveFixtureSnapshot:
    """Fixture set after applying cached or freshly fetched real results."""

    fixtures: tuple[WorldCupFixture, ...]
    cache_path: Path
    source_url: str
    fetched_at: datetime | None
    refreshed: bool
    scored_fixture_count: int
    stale_missing_fixture_count: int

    @property
    def has_live_data(self) -> bool:
        """Return whether at least one real score came from live/cache data."""
        return self.fetched_at is not None and self.scored_fixture_count > 0


def load_world_cup_fixture_snapshot(
    *,
    cache_path: Path = DEFAULT_WORLD_CUP_RESULTS_CACHE,
    fetch_if_needed: bool = True,
    force: bool = False,
    as_of: datetime | None = None,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
    timeout_seconds: float = 30.0,
) -> LiveFixtureSnapshot:
    """Return fixtures with cached/fetched FIFA results applied.

    A network fetch is performed only when forced or when at least one fixture
    that should already have a result still lacks a score and the cache is old.
    """
    reference_time = _utc_now() if as_of is None else _ensure_utc(as_of)
    cache_payload = _read_cache_payload(cache_path)
    cached_scores = _scores_from_cache_payload(cache_payload)
    cached_fetched_at = _cache_fetched_at(cache_payload)
    fixtures = apply_live_scores(world_cup_2026_fixtures(), cached_scores)

    should_fetch = force or (
        fetch_if_needed
        and fixtures_need_update(fixtures, as_of=reference_time)
        and not _cache_is_fresh(cached_fetched_at, reference_time, cache_ttl)
    )
    refreshed = False

    if should_fetch:
        fetched_at = reference_time
        fetched_scores = fetch_fifa_world_cup_scores(timeout_seconds=timeout_seconds)
        write_scores_cache(fetched_scores, cache_path=cache_path, fetched_at=fetched_at)
        fixtures = apply_live_scores(world_cup_2026_fixtures(), fetched_scores)
        cached_fetched_at = fetched_at
        refreshed = True

    return LiveFixtureSnapshot(
        fixtures=fixtures,
        cache_path=cache_path,
        source_url=FIFA_WORLD_CUP_2026_MATCHES_URL,
        fetched_at=cached_fetched_at,
        refreshed=refreshed,
        scored_fixture_count=sum(
            1
            for fixture in fixtures
            if fixture.played_home_goals is not None and fixture.played_away_goals is not None
        ),
        stale_missing_fixture_count=len(stale_missing_fixtures(fixtures, as_of=reference_time)),
    )


def fixtures_need_update(
    fixtures: tuple[WorldCupFixture, ...],
    *,
    as_of: datetime | None = None,
) -> bool:
    """Return whether past fixtures lack known scores."""
    return bool(stale_missing_fixtures(fixtures, as_of=as_of))


def stale_missing_fixtures(
    fixtures: tuple[WorldCupFixture, ...],
    *,
    as_of: datetime | None = None,
) -> tuple[WorldCupFixture, ...]:
    """Return fixtures old enough to be final but still missing scores."""
    reference_time = _utc_now() if as_of is None else _ensure_utc(as_of)
    return tuple(
        fixture
        for fixture in fixtures
        if fixture.match_date + FINISHED_MATCH_GRACE_PERIOD <= reference_time
        and (fixture.played_home_goals is None or fixture.played_away_goals is None)
    )


def fetch_fifa_world_cup_scores(
    *,
    timeout_seconds: float = 30.0,
) -> dict[int, LiveFixtureScore]:
    """Fetch current World Cup fixture scores from FIFA's public calendar API."""
    with httpx.Client(
        timeout=timeout_seconds,
        headers={"User-Agent": DEFAULT_SOURCE_USER_AGENT},
        follow_redirects=True,
    ) as client:
        response = client.get(FIFA_WORLD_CUP_2026_MATCHES_URL)
        response.raise_for_status()
    return parse_fifa_scores(response.json())


def parse_fifa_scores(payload: dict[str, Any]) -> dict[int, LiveFixtureScore]:
    """Parse FIFA calendar API payload into scores keyed by match number."""
    raw_results = payload.get("Results")
    if not isinstance(raw_results, list):
        raise ValueError("FIFA payload does not contain a Results list.")

    scores: dict[int, LiveFixtureScore] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        match_number = _optional_int(item.get("MatchNumber"))
        if match_number is None:
            continue
        scores[match_number] = LiveFixtureScore(
            match_number=match_number,
            home_team_id=_team_code(item.get("Home")),
            away_team_id=_team_code(item.get("Away")),
            home_goals=_score_value(item, "HomeTeamScore", "Home"),
            away_goals=_score_value(item, "AwayTeamScore", "Away"),
            match_status=_optional_int(item.get("MatchStatus")),
            result_type=_optional_int(item.get("ResultType")),
        )
    return scores


def apply_live_scores(
    fixtures: tuple[WorldCupFixture, ...],
    scores: dict[int, LiveFixtureScore],
) -> tuple[WorldCupFixture, ...]:
    """Overlay live score/team data onto the static fixture schedule."""
    updated: list[WorldCupFixture] = []
    for fixture in fixtures:
        score = scores.get(fixture.match_number)
        if score is None:
            updated.append(fixture)
            continue
        updated.append(
            replace(
                fixture,
                home_slot=score.home_team_id or fixture.home_slot,
                away_slot=score.away_team_id or fixture.away_slot,
                played_home_goals=score.home_goals,
                played_away_goals=score.away_goals,
            )
        )
    return tuple(updated)


def write_scores_cache(
    scores: dict[int, LiveFixtureScore],
    *,
    cache_path: Path = DEFAULT_WORLD_CUP_RESULTS_CACHE,
    fetched_at: datetime | None = None,
) -> Path:
    """Persist normalized live scores to the project raw-data cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    reference_time = _utc_now() if fetched_at is None else _ensure_utc(fetched_at)
    payload = {
        "fetched_at": reference_time.isoformat().replace("+00:00", "Z"),
        "source_url": FIFA_WORLD_CUP_2026_MATCHES_URL,
        "scores": [
            {
                "match_number": score.match_number,
                "home_team_id": score.home_team_id,
                "away_team_id": score.away_team_id,
                "home_goals": score.home_goals,
                "away_goals": score.away_goals,
                "match_status": score.match_status,
                "result_type": score.result_type,
            }
            for score in sorted(scores.values(), key=lambda item: item.match_number)
        ],
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for live result updates."""
    parser = argparse.ArgumentParser(
        description="Fetch updated FIFA World Cup 2026 real scores into the local cache.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_WORLD_CUP_RESULTS_CACHE,
        help="Path to the local FIFA results cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch even when the local cache is still fresh.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only read the existing cache and never call the FIFA API.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    try:
        snapshot = load_world_cup_fixture_snapshot(
            cache_path=args.cache_path,
            fetch_if_needed=not args.offline,
            force=args.force,
        )
    except (httpx.HTTPError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    action = "Fetched" if snapshot.refreshed else "Using cached"
    fetched_at = snapshot.fetched_at.isoformat() if snapshot.fetched_at else "static schedule"
    print(
        f"{action} World Cup results: {snapshot.scored_fixture_count} matches with scores; "
        f"{snapshot.stale_missing_fixture_count} past matches still missing. "
        f"Cache: {snapshot.cache_path}. Updated at: {fetched_at}."
    )
    return 0


def _read_cache_payload(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _scores_from_cache_payload(payload: dict[str, Any] | None) -> dict[int, LiveFixtureScore]:
    if payload is None:
        return {}
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        return {}

    scores: dict[int, LiveFixtureScore] = {}
    for raw_score in raw_scores:
        if not isinstance(raw_score, dict):
            continue
        match_number = _optional_int(raw_score.get("match_number"))
        if match_number is None:
            continue
        scores[match_number] = LiveFixtureScore(
            match_number=match_number,
            home_team_id=_optional_str(raw_score.get("home_team_id")),
            away_team_id=_optional_str(raw_score.get("away_team_id")),
            home_goals=_optional_int(raw_score.get("home_goals")),
            away_goals=_optional_int(raw_score.get("away_goals")),
            match_status=_optional_int(raw_score.get("match_status")),
            result_type=_optional_int(raw_score.get("result_type")),
        )
    return scores


def _cache_fetched_at(payload: dict[str, Any] | None) -> datetime | None:
    if payload is None:
        return None
    raw_value = payload.get("fetched_at")
    if not isinstance(raw_value, str):
        return None
    return _ensure_utc(datetime.fromisoformat(raw_value.replace("Z", "+00:00")))


def _cache_is_fresh(
    fetched_at: datetime | None,
    as_of: datetime,
    cache_ttl: timedelta,
) -> bool:
    return fetched_at is not None and as_of - fetched_at < cache_ttl


def _team_code(team_payload: object) -> str | None:
    if not isinstance(team_payload, dict):
        return None
    return _optional_str(team_payload.get("Abbreviation")) or _optional_str(
        team_payload.get("IdCountry")
    )


def _score_value(item: dict[str, Any], score_key: str, team_key: str) -> int | None:
    direct_score = _optional_int(item.get(score_key))
    if direct_score is not None:
        return direct_score
    team_payload = item.get(team_key)
    if not isinstance(team_payload, dict):
        return None
    return _optional_int(team_payload.get("Score"))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
