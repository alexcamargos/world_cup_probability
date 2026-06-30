from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import src.live_results as live_results
from src.live_results import (
    LiveFixtureScore,
    apply_live_scores,
    fixtures_need_update,
    load_world_cup_fixture_snapshot,
    main,
    parse_fifa_scores,
    write_scores_cache,
)
from src.world_cup_2026_schedule import WorldCupFixture


def test_parse_fifa_scores_reads_scores_and_team_codes() -> None:
    payload = {
        "Results": [
            {
                "MatchNumber": 36,
                "Home": {"Abbreviation": "TUN", "Score": 2},
                "Away": {"Abbreviation": "JPN", "Score": 1},
                "HomeTeamScore": 2,
                "AwayTeamScore": 1,
                "MatchStatus": 0,
                "ResultType": 1,
            }
        ]
    }

    scores = parse_fifa_scores(payload)

    assert scores[36] == LiveFixtureScore(
        match_number=36,
        home_team_id="TUN",
        away_team_id="JPN",
        home_goals=2,
        away_goals=1,
        match_status=0,
        result_type=1,
    )


def test_apply_live_scores_overlays_missing_static_result() -> None:
    fixture = WorldCupFixture(
        match_number=36,
        match_date=datetime(2026, 6, 21, 4, tzinfo=UTC),
        round_name="group_stage",
        group_name="Group F",
        home_slot="TUN",
        away_slot="JPN",
        stadium="Test",
        city="Test",
        country="USA",
    )

    updated = apply_live_scores(
        (fixture,),
        {
            36: LiveFixtureScore(
                match_number=36,
                home_team_id="TUN",
                away_team_id="JPN",
                home_goals=2,
                away_goals=1,
            )
        },
    )

    assert updated[0].played_home_goals == 2
    assert updated[0].played_away_goals == 1


def test_fixtures_need_update_when_past_match_has_no_score() -> None:
    fixture = WorldCupFixture(
        match_number=36,
        match_date=datetime(2026, 6, 21, 4, tzinfo=UTC),
        round_name="group_stage",
        group_name="Group F",
        home_slot="TUN",
        away_slot="JPN",
        stadium="Test",
        city="Test",
        country="USA",
    )

    assert fixtures_need_update(
        (fixture,),
        as_of=fixture.match_date + timedelta(hours=3),
    )


def test_snapshot_fetches_when_cache_lacks_past_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "world_cup_2026_results.json"

    def fake_fetch(*, timeout_seconds: float = 30.0) -> dict[int, LiveFixtureScore]:
        return {
            36: LiveFixtureScore(
                match_number=36,
                home_team_id="TUN",
                away_team_id="JPN",
                home_goals=2,
                away_goals=1,
            )
        }

    monkeypatch.setattr(live_results, "fetch_fifa_world_cup_scores", fake_fetch)

    snapshot = load_world_cup_fixture_snapshot(
        cache_path=cache_path,
        as_of=datetime(2026, 6, 30, tzinfo=UTC),
    )
    fixture = next(item for item in snapshot.fixtures if item.match_number == 36)

    assert snapshot.refreshed
    assert cache_path.exists()
    assert fixture.played_home_goals == 2
    assert fixture.played_away_goals == 1


def test_snapshot_uses_fresh_cache_without_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "world_cup_2026_results.json"
    fetched_at = datetime(2026, 6, 30, 12, tzinfo=UTC)
    write_scores_cache(
        {
            36: LiveFixtureScore(
                match_number=36,
                home_team_id="TUN",
                away_team_id="JPN",
                home_goals=2,
                away_goals=1,
            )
        },
        cache_path=cache_path,
        fetched_at=fetched_at,
    )

    def fail_fetch(*, timeout_seconds: float = 30.0) -> dict[int, LiveFixtureScore]:
        raise AssertionError("Fresh cache should prevent a network fetch.")

    monkeypatch.setattr(live_results, "fetch_fifa_world_cup_scores", fail_fetch)

    snapshot = load_world_cup_fixture_snapshot(
        cache_path=cache_path,
        as_of=fetched_at + timedelta(minutes=5),
    )

    assert not snapshot.refreshed
    assert snapshot.fetched_at == fetched_at


def test_cli_offline_reports_cached_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_path = tmp_path / "world_cup_2026_results.json"
    write_scores_cache(
        {
            36: LiveFixtureScore(
                match_number=36,
                home_team_id="TUN",
                away_team_id="JPN",
                home_goals=2,
                away_goals=1,
            )
        },
        cache_path=cache_path,
        fetched_at=datetime(2026, 6, 30, tzinfo=UTC),
    )

    exit_code = main(["--offline", "--cache-path", str(cache_path)])

    assert exit_code == 0
    assert "Using cached World Cup results" in capsys.readouterr().out
