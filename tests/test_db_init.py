from __future__ import annotations

from pathlib import Path

import duckdb

from src.db_init import initialize_database


def test_initialize_database_reloads_referenced_teams_without_fk_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_file = raw_dir / "teams.csv"
    source_file.write_text(
        "\n".join(
            [
                "team_id,team_name",
                "T-46,Referenced Team",
                "T-47,Unreferenced Team",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"

    initialize_database(db_path=db_path, raw_dir=raw_dir, load_raw_files=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO f_matches (
                match_id,
                match_date,
                home_team_id,
                away_team_id,
                home_team_score,
                away_team_score,
                source_file,
                loaded_at
            ) VALUES (
                'm1',
                DATE '2025-01-01',
                'T-46',
                'T-47',
                1,
                0,
                'test',
                current_timestamp
            )
            """
        )

    initialize_database(db_path=db_path, raw_dir=raw_dir, load_raw_files=True)

    with duckdb.connect(str(db_path), read_only=True) as con:
        team_count = con.execute("SELECT COUNT(*) FROM d_teams").fetchone()[0]
        match_count = con.execute("SELECT COUNT(*) FROM f_matches").fetchone()[0]

    assert team_count == 2
    assert match_count == 1


def test_initialize_database_loads_csv_scores_with_na_as_null(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "results.csv").write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament",
                "2025-01-01,Brazil,Argentina,2,1,Friendly",
                "2026-06-19,Scotland,Morocco,NA,NA,FIFA World Cup",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"

    initialize_database(db_path=db_path, raw_dir=raw_dir, load_raw_files=True)

    with duckdb.connect(str(db_path), read_only=True) as con:
        null_score_count = con.execute(
            """
            SELECT COUNT(*)
            FROM f_matches
            WHERE home_team_score IS NULL AND away_team_score IS NULL
            """
        ).fetchone()[0]

    assert null_score_count == 1
