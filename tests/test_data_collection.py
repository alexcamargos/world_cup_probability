from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

import src.data_collection as data_collection
from src.data_collection import (
    TransfermarktMarketValue,
    download_kaggle_dataset,
    load_fjelstul_world_cup_history,
    load_historical_matches,
    load_squad_attributes,
    parse_market_value_eur,
    persist_transfermarkt_market_values,
    read_transfermarkt_manifest,
    read_transfermarkt_raw,
    run_downloads,
    write_transfermarkt_raw,
)


def test_load_historical_matches_applies_2010_cutoff(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "matches"
    raw_dir.mkdir(parents=True)
    source_file = raw_dir / "results.csv"
    source_file.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "2009-12-31,Brazil,Argentina,1,0,Friendly,Doha,Qatar,TRUE",
                "2010-01-01,Brazil,Argentina,2,1,Friendly,Doha,Qatar,TRUE",
                "2014-07-13,Germany,Argentina,1,0,FIFA World Cup,Rio de Janeiro,Brazil,TRUE",
                "2026-06-18,Scotland,Morocco,2,1,FIFA World Cup,Foxborough,United States,TRUE",
                "2026-06-19,Scotland,Morocco,NA,NA,FIFA World Cup,Foxborough,United States,TRUE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"

    result = load_historical_matches(raw_dir, db_path=db_path)

    assert result.rows_loaded == 2
    with duckdb.connect(str(db_path), read_only=True) as con:
        min_date, match_count, team_count = con.execute(
            """
            SELECT
                MIN(match_date),
                COUNT(*),
                (SELECT COUNT(*) FROM d_teams)
            FROM f_matches
            """
        ).fetchone()

    assert str(min_date) == "2010-01-01"
    assert match_count == 2
    assert team_count == 3


def test_load_squad_attributes_aggregates_top_11_players(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "squad"
    raw_dir.mkdir(parents=True)
    source_file = raw_dir / "players.csv"
    lines = ["nationality,overall,pace,stamina"]
    for index in range(12):
        lines.append(f"Brazil,{90 - index},{80 + index},{70 + index}")
    for index in range(3):
        lines.append(f"Canada,{75 - index},{70 + index},{65 + index}")
    source_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"

    result = load_squad_attributes(
        raw_dir,
        db_path=db_path,
        source_season="2025-2026",
        source_dataset="example/ea-fc",
    )

    assert result.rows_loaded == 2
    with duckdb.connect(str(db_path), read_only=True) as con:
        brazil = con.execute(
            """
            SELECT avg_overall, sampled_player_count
            FROM d_squad_attributes
            WHERE team_id = 'Brazil'
            """
        ).fetchone()

    assert brazil == (85.0, 11)


def test_load_fjelstul_world_cup_history_uses_prior_tournaments_only(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "fjelstul_worldcup"
    raw_dir.mkdir(parents=True)
    source_file = raw_dir / "matches.csv"
    source_file.write_text(
        "\n".join(
            [
                (
                    "tournament_id,tournament_name,match_id,match_date,home_team_id,"
                    "home_team_name,home_team_code,away_team_id,away_team_name,"
                    "away_team_code,home_team_score,away_team_score"
                ),
                (
                    "WC-1930,1930 FIFA Men's World Cup,M-1,1930-07-13,T-01,"
                    "Brazil,BRA,T-02,Argentina,ARG,2,1"
                ),
                (
                    "WC-1934,1934 FIFA Men's World Cup,M-2,1934-05-27,T-01,"
                    "Brazil,BRA,T-03,Spain,ESP,0,0"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"

    result = load_fjelstul_world_cup_history(raw_dir, db_path=db_path)

    assert result.rows_loaded > 0
    with duckdb.connect(str(db_path), read_only=True) as con:
        before_1930 = con.execute(
            """
            SELECT
                prior_world_cup_appearances,
                prior_world_cup_points_per_match,
                prior_world_cup_goal_diff_per_match
            FROM d_world_cup_prior_team_history
            WHERE team_name = 'Brazil' AND as_of_year = 1930
            """
        ).fetchone()
        before_1934 = con.execute(
            """
            SELECT
                prior_world_cup_appearances,
                prior_world_cup_points_per_match,
                prior_world_cup_goal_diff_per_match
            FROM d_world_cup_prior_team_history
            WHERE team_name = 'Brazil' AND as_of_year = 1934
            """
        ).fetchone()

    assert before_1930 == (0, 0.0, 0.0)
    assert before_1934 == (1, 3.0, 1.0)


def test_load_fjelstul_world_cup_history_builds_prior_discipline_rates(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "fjelstul_worldcup"
    raw_dir.mkdir(parents=True)
    (raw_dir / "matches.csv").write_text(
        "\n".join(
            [
                (
                    "tournament_id,tournament_name,match_id,match_date,home_team_id,"
                    "home_team_name,home_team_code,away_team_id,away_team_name,"
                    "away_team_code,home_team_score,away_team_score"
                ),
                (
                    "WC-1970,1970 FIFA Men's World Cup,M-1,1970-05-31,T-01,"
                    "Brazil,BRA,T-02,Argentina,ARG,2,1"
                ),
                (
                    "WC-1970,1970 FIFA Men's World Cup,M-2,1970-06-03,T-01,"
                    "Brazil,BRA,T-03,Spain,ESP,1,1"
                ),
                (
                    "WC-1974,1974 FIFA Men's World Cup,M-3,1974-06-13,T-01,"
                    "Brazil,BRA,T-02,Argentina,ARG,0,0"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "bookings.csv").write_text(
        "\n".join(
            [
                (
                    "tournament_id,tournament_name,match_id,match_date,team_id,team_name,"
                    "team_code,yellow_card,red_card,second_yellow_card,sending_off"
                ),
                "WC-1970,1970 FIFA Men's World Cup,M-1,1970-05-31,T-01,Brazil,BRA,1,0,0,0",
                "WC-1970,1970 FIFA Men's World Cup,M-2,1970-06-03,T-01,Brazil,BRA,1,0,1,1",
                "WC-1970,1970 FIFA Men's World Cup,M-1,1970-05-31,T-02,Argentina,ARG,0,1,0,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"

    load_fjelstul_world_cup_history(raw_dir, db_path=db_path)

    with duckdb.connect(str(db_path), read_only=True) as con:
        brazil = con.execute(
            """
            SELECT
                prior_world_cup_yellow_cards_per_match,
                prior_world_cup_sending_offs_per_match,
                prior_world_cup_fair_play_penalty_per_match
            FROM d_world_cup_prior_discipline_history
            WHERE team_name = 'Brazil' AND as_of_year = 1974
            """
        ).fetchone()
        before_1970 = con.execute(
            """
            SELECT
                prior_world_cup_yellow_cards_per_match,
                prior_world_cup_sending_offs_per_match,
                prior_world_cup_fair_play_penalty_per_match
            FROM d_world_cup_prior_discipline_history
            WHERE team_name = 'Brazil' AND as_of_year = 1970
            """
        ).fetchone()

    assert before_1970 == (0.0, 0.0, 0.0)
    assert brazil == (1.0, 0.5, 2.0)


def test_transfermarkt_parser_and_persistence(tmp_path: Path) -> None:
    html = """
    <html>
      <body>
        <span>Market value: €45.00m</span>
        <div>Total market value</div>
        <a>€1.23bn</a>
      </body>
    </html>
    """
    value = parse_market_value_eur(html)
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"

    result = persist_transfermarkt_market_values(
        [
            TransfermarktMarketValue(
                team_id="Brazil",
                team_name="Brazil",
                total_market_value_eur=value,
                source_url="https://www.transfermarkt.com/example",
                scraped_at=datetime(2026, 6, 20, tzinfo=UTC),
            )
        ],
        db_path=db_path,
    )

    assert result.rows_loaded == 1
    with duckdb.connect(str(db_path), read_only=True) as con:
        stored = con.execute(
            """
            SELECT total_market_value_eur, market_value_eur, market_value_currency
            FROM d_teams
            WHERE team_id = 'Brazil'
            """
        ).fetchone()

    assert stored == (1_230_000_000.0, 1_230_000_000.0, "EUR")


def test_transfermarkt_raw_round_trip(tmp_path: Path) -> None:
    raw_path = tmp_path / "transfermarkt" / "market_values.jsonl"
    expected = TransfermarktMarketValue(
        team_id="Brazil",
        team_name="Brazil",
        total_market_value_eur=1_000_000.0,
        source_url="https://www.transfermarkt.com/example",
        scraped_at=datetime(2026, 6, 20, tzinfo=UTC),
    )

    write_transfermarkt_raw([expected], raw_path)
    actual = read_transfermarkt_raw(raw_path)

    assert actual == [expected]


def test_transfermarkt_manifest_accepts_search_only_targets(tmp_path: Path) -> None:
    manifest_path = tmp_path / "transfermarkt_teams.json"
    manifest_path.write_text(
        """
        {
          "teams": [
            {
              "team_id": "Brazil",
              "team_name": "Brazil"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    targets = read_transfermarkt_manifest(manifest_path)

    assert targets[0].team_id == "Brazil"
    assert targets[0].url is None


def test_download_kaggle_dataset_rejects_placeholder_slug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not the README placeholder"):
        download_kaggle_dataset("owner/dataset", tmp_path / "raw")


def test_download_kaggle_dataset_skips_when_raw_files_exist(tmp_path: Path) -> None:
    class FakeKaggleClient:
        authenticated = False
        downloaded = False

        def authenticate(self) -> None:
            self.authenticated = True

        def dataset_download_files(
            self,
            dataset: str,
            path: str | None = None,
            force: bool = False,
            quiet: bool = True,
            unzip: bool = False,
            licenses: tuple[str, ...] = (),
        ) -> None:
            self.downloaded = True

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "results.csv").write_text("date,home_team\n", encoding="utf-8")
    client = FakeKaggleClient()

    result = download_kaggle_dataset("example/dataset", raw_dir, client=client)

    assert result == raw_dir
    assert not client.authenticated
    assert not client.downloaded


def test_download_kaggle_dataset_force_downloads_existing_raw(tmp_path: Path) -> None:
    class FakeKaggleClient:
        authenticated = False
        downloaded = False

        def authenticate(self) -> None:
            self.authenticated = True

        def dataset_download_files(
            self,
            dataset: str,
            path: str | None = None,
            force: bool = False,
            quiet: bool = True,
            unzip: bool = False,
            licenses: tuple[str, ...] = (),
        ) -> None:
            self.downloaded = True
            assert force

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "results.csv").write_text("date,home_team\n", encoding="utf-8")
    client = FakeKaggleClient()

    result = download_kaggle_dataset("example/dataset", raw_dir, client=client, force=True)

    assert result == raw_dir
    assert client.authenticated
    assert client.downloaded


def test_run_downloads_skips_existing_fbref_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fbref_dir = tmp_path / "raw" / "fbref"
    fbref_dir.mkdir(parents=True)
    (fbref_dir / "fbref_schedule.parquet").write_bytes(b"schedule")
    (fbref_dir / "fbref_team_stats.parquet").write_bytes(b"team-stats")

    def fail_fetch(*args: object, **kwargs: object) -> list[Path]:
        raise AssertionError("FBref download should not run when raw files exist.")

    monkeypatch.setattr(data_collection, "fetch_fbref_with_soccerdata", fail_fetch)

    result = run_downloads(
        SimpleNamespace(
            raw_dir=tmp_path / "raw",
            sources=["fbref"],
            force_download=False,
            fbref_leagues=["INT-World Cup"],
            fbref_seasons=["2022"],
            fbref_no_cache=False,
            ea_fc_dataset="",
            transfermarkt_manifest=None,
        )
    )

    assert result == [fbref_dir]


def test_run_downloads_skips_existing_transfermarkt_raw_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfermarkt_dir = tmp_path / "raw" / "transfermarkt"
    transfermarkt_dir.mkdir(parents=True)
    raw_file = transfermarkt_dir / "market_values_20260621T120000Z.jsonl"
    raw_file.write_text("{}\n", encoding="utf-8")

    def fail_scrape(*args: object, **kwargs: object) -> list[TransfermarktMarketValue]:
        raise AssertionError("Transfermarkt scrape should not run when raw files exist.")

    monkeypatch.setattr(data_collection, "scrape_transfermarkt_market_values", fail_scrape)

    result = run_downloads(
        SimpleNamespace(
            raw_dir=tmp_path / "raw",
            sources=["transfermarkt"],
            force_download=False,
            fbref_leagues=[],
            fbref_seasons=[],
            fbref_no_cache=False,
            ea_fc_dataset="",
            transfermarkt_manifest=None,
        )
    )

    assert result == [raw_file]
