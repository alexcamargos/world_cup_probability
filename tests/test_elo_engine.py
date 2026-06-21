from __future__ import annotations

from pathlib import Path

import duckdb

from src.db_init import initialize_database
from src.elo_engine import build_elo_history


def test_build_elo_history_upserts_existing_match(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"
    initialize_database(db_path=db_path, load_raw_files=False)

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO d_teams (team_id, team_name, loaded_at)
            VALUES
                ('Brazil', 'Brazil', current_timestamp),
                ('Argentina', 'Argentina', current_timestamp),
                ('Scotland', 'Scotland', current_timestamp),
                ('Morocco', 'Morocco', current_timestamp)
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
            ) VALUES
                ('m1', DATE '2026-06-20', 'Friendly', 'Brazil', 'Argentina', 2, 1, TRUE),
                ('m_wc', DATE '2026-06-20', 'FIFA World Cup', 'Scotland', 'Morocco', 2, 1, TRUE)
            """
        )

    build_elo_history(db_path=db_path)
    build_elo_history(db_path=db_path)

    with duckdb.connect(str(db_path)) as con:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                max(home_rating_before) AS home_rating_before,
                max(away_rating_before) AS away_rating_before,
                max(home_actual_score) AS home_actual_score,
                max(away_actual_score) AS away_actual_score,
                bool_and(updated_at IS NOT NULL) AS has_updated_at
            FROM f_elo_history
            WHERE match_id = 'm1'
            """
        ).fetchone()
        world_cup_rows = con.execute(
            """
            SELECT COUNT(*)
            FROM f_elo_history
            WHERE match_id = 'm_wc'
            """
        ).fetchone()[0]

    assert row == (1, 1500.0, 1500.0, 1.0, 0.0, True)
    assert world_cup_rows == 0
