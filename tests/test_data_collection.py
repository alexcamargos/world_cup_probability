from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from src.data_collection import (
    TransfermarktMarketValue,
    load_historical_matches,
    load_squad_attributes,
    parse_market_value_eur,
    persist_transfermarkt_market_values,
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
