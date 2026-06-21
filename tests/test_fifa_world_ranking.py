from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pytest

import src.fifa_world_ranking as fifa_world_ranking
from src.db_init import initialize_database
from src.elo_engine import initialize_elo_history
from src.feature_pipeline import build_feature_frame
from src.fifa_world_ranking import (
    load_fifa_world_ranking,
    parse_fifa_world_ranking_api,
    parse_update_dates,
    persist_fifa_world_ranking_snapshot,
    read_fifa_world_ranking_raw,
    write_fifa_world_ranking_raw,
)


def _sample_page_html() -> str:
    payload = {
        "props": {
            "pageProps": {
                "pageData": {
                    "ranking": {
                        "lastUpdateDate": "2026-06-11T10:00:59.636Z",
                        "nextUpdateDate": "2026-07-20T00:00:00.000Z",
                    }
                }
            }
        }
    }
    return (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}"
        "</script></body></html>"
    )


def _sample_api_payload() -> dict[str, object]:
    return {
        "Results": [
            {
                "IdCountry": "ARG",
                "IdTeam": "43922",
                "TeamName": [{"Locale": "en-GB", "Description": "Argentina"}],
                "ConfederationName": "CONMEBOL",
                "Rank": 1,
                "PrevRank": 1,
                "DecimalTotalPoints": 1877.27,
                "DecimalPrevPoints": 1867.25,
                "RankingMovement": 0,
                "Matches": 2,
                "PubDate": "2026-06-11T10:00:00+00:00",
            },
            {
                "IdCountry": "BRA",
                "IdTeam": "43924",
                "TeamName": [{"Locale": "en-GB", "Description": "Brazil"}],
                "ConfederationName": "CONMEBOL",
                "Rank": 5,
                "PrevRank": 5,
                "DecimalTotalPoints": 1761.00,
                "DecimalPrevPoints": 1760.01,
                "RankingMovement": 0,
                "Matches": 2,
                "PubDate": "2026-06-11T10:00:00+00:00",
            },
        ]
    }


def test_fifa_world_ranking_raw_round_trip(tmp_path: Path) -> None:
    ranking_date, next_update_date = parse_update_dates(_sample_page_html())
    snapshot = parse_fifa_world_ranking_api(
        _sample_api_payload(),
        ranking_date=ranking_date,
        next_update_date=next_update_date,
    )
    raw_path = tmp_path / "fifa_world_ranking_snapshot.jsonl"

    write_fifa_world_ranking_raw(snapshot, raw_path)
    loaded = read_fifa_world_ranking_raw(raw_path)

    assert ranking_date == date(2026, 6, 11)
    assert next_update_date == date(2026, 7, 20)
    assert loaded.rows == snapshot.rows
    assert loaded.aliases == snapshot.aliases
    assert any(alias.team_alias == "Brazil" for alias in loaded.aliases)


def test_fifa_world_ranking_load_skips_download_when_raw_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = parse_fifa_world_ranking_api(
        _sample_api_payload(),
        ranking_date=date(2026, 6, 11),
        next_update_date=date(2026, 7, 20),
    )
    raw_path = tmp_path / "raw" / "fifa_world_ranking" / "men_snapshot.jsonl"
    write_fifa_world_ranking_raw(snapshot, raw_path)

    def fail_fetch(*args: object, **kwargs: object) -> object:
        raise AssertionError("FIFA World Ranking download should not run.")

    monkeypatch.setattr(fifa_world_ranking, "fetch_fifa_world_ranking_snapshot", fail_fetch)

    rows_loaded = load_fifa_world_ranking(
        db_path=tmp_path / "warehouse" / "world_cup.duckdb",
        raw_path=raw_path,
    )

    assert rows_loaded == 2


def test_fifa_world_ranking_supports_feature_training_frame(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"
    initialize_database(db_path=db_path, load_raw_files=False)
    initialize_elo_history(db_path=db_path)

    snapshot = parse_fifa_world_ranking_api(
        _sample_api_payload(),
        ranking_date=date(2026, 6, 11),
        next_update_date=date(2026, 7, 20),
    )
    rows_loaded = persist_fifa_world_ranking_snapshot(snapshot, db_path=db_path)

    assert rows_loaded == 2
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO d_teams (team_id, team_name, loaded_at)
            VALUES
                ('Brazil', 'Brazil', current_timestamp),
                ('Argentina', 'Argentina', current_timestamp)
            """
        )
        con.execute(
            """
            INSERT INTO f_matches (
                match_id,
                match_date,
                competition,
                home_team_id,
                away_team_id,
                home_team_score,
                away_team_score,
                neutral_site
            ) VALUES ('m1', DATE '2026-06-20', 'Friendly', 'Brazil', 'Argentina', 2, 1, TRUE)
            """
        )
        con.execute(
            """
            INSERT INTO f_elo_history (
                match_id,
                match_date,
                home_team_id,
                away_team_id,
                home_rating_before,
                away_rating_before,
                home_rating_after,
                away_rating_after,
                home_expected_score,
                away_expected_score,
                home_actual_score,
                away_actual_score,
                k_factor,
                competition_weight,
                home_advantage_points,
                neutral_site,
                updated_at
            ) VALUES (
                'm1',
                DATE '2026-06-20',
                'Brazil',
                'Argentina',
                1600.0,
                1500.0,
                1610.0,
                1490.0,
                0.5,
                0.5,
                1.0,
                0.0,
                20.0,
                1.0,
                0.0,
                TRUE,
                current_timestamp
            )
            """
        )
        dates = con.execute(
            """
            SELECT ranking_date, next_update_date
            FROM d_fifa_world_ranking
            WHERE fifa_country_code = 'ARG'
            """
        ).fetchone()

    frame = build_feature_frame(db_path)
    row = frame.row(0, named=True)

    assert dates == (date(2026, 6, 11), date(2026, 7, 20))
    assert row["fifa_world_ranking_points_diff"] == pytest.approx(-116.27)
    assert row["fifa_world_ranking_rank_diff"] == pytest.approx(-4.0)
