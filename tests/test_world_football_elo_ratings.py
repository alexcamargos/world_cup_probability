from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from src.db_init import initialize_database
from src.elo_engine import initialize_elo_history
from src.feature_pipeline import build_feature_frame
from src.world_football_elo_ratings import (
    parse_team_dictionary,
    parse_world_ratings,
    persist_world_football_elo_ratings_snapshot,
    read_world_football_elo_ratings_raw,
    write_world_football_elo_ratings_raw,
)

TEAM_TSV = "\n".join(
    [
        "BR\tBrazil",
        "AR\tArgentina",
        "CZ\tCzechia",
    ]
)

WORLD_TSV = "\n".join(
    [
        "1\t1\tAR\t2128\t1\t2172",
        "2\t2\tBR\t1986\t1\t2196",
        "3\t48\tCZ\t1696\t1\t1935",
    ]
)


def test_world_football_elo_ratings_raw_round_trip(tmp_path: Path) -> None:
    snapshot = parse_world_ratings(
        WORLD_TSV,
        team_dictionary=parse_team_dictionary(TEAM_TSV),
        rating_date=date(2026, 6, 20),
        source_url="https://www.eloratings.net/World.tsv",
    )
    raw_path = tmp_path / "world_football_elo_ratings_snapshot.jsonl"

    write_world_football_elo_ratings_raw(snapshot, raw_path)
    loaded = read_world_football_elo_ratings_raw(raw_path)

    assert loaded.ratings == snapshot.ratings
    assert loaded.aliases == snapshot.aliases
    assert any(alias.team_alias == "Czech Republic" for alias in loaded.aliases)


def test_world_football_elo_ratings_supports_feature_training_frame(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse" / "world_cup.duckdb"
    initialize_database(db_path=db_path, load_raw_files=False)
    initialize_elo_history(db_path=db_path)

    snapshot = parse_world_ratings(
        WORLD_TSV,
        team_dictionary=parse_team_dictionary(TEAM_TSV),
        rating_date=date(2026, 6, 20),
        source_url="https://www.eloratings.net/World.tsv",
    )
    rows_loaded = persist_world_football_elo_ratings_snapshot(snapshot, db_path=db_path)

    assert rows_loaded == 3
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

    frame = build_feature_frame(db_path)

    assert frame.columns == [
        "world_cup_probability_elo_diff",
        "world_football_elo_ratings_diff",
        "fifa_world_ranking_points_diff",
        "fifa_world_ranking_rank_diff",
        "market_value_diff",
        "recent_form_diff",
        "target",
    ]
    row = frame.row(0, named=True)
    assert row["world_cup_probability_elo_diff"] == pytest.approx(100.0)
    assert row["world_football_elo_ratings_diff"] == pytest.approx(-142.0)
    assert row["fifa_world_ranking_points_diff"] == pytest.approx(100.0)
    assert row["fifa_world_ranking_rank_diff"] == pytest.approx(0.0)
    assert row["target"] == 2
