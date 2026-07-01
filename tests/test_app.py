from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from src.app import (
    PREDICTION_SOURCE_SIMULATION,
    RoundOption,
    _accuracy_summary,
    _actual_group_standings,
    _add_played_fixture_result,
    _filter_played_analysis_by_round,
    _format_actual_score,
    _format_score,
    _group_round_accuracy_rows,
    _home_projection_showcase_html,
    _load_round_probabilities,
    _match_prediction_cards_html,
    _metric_cards_html,
    _phase_accuracy_rows,
    _prediction_result,
    _third_place_rows,
)
from src.world_cup_2026_schedule import WorldCupFixture


def test_load_round_probabilities_includes_modal_model_score(tmp_path: Path) -> None:
    db_path = tmp_path / "world_cup.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE simulated_results (
                simulation_id INTEGER NOT NULL,
                match_number INTEGER NOT NULL,
                group_name VARCHAR,
                home_team_id VARCHAR NOT NULL,
                home_team_name VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                away_team_name VARCHAR NOT NULL,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL
            )
            """,
        )
        con.executemany(
            """
            INSERT INTO simulated_results (
                simulation_id,
                match_number,
                group_name,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_goals,
                away_goals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 2, 1),
                (2, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 2, 1),
                (3, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 1, 0),
                (4, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 1, 0),
                (5, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 0, 0),
            ],
        )
        con.execute(
            """
            CREATE TABLE outcome_predictions (
                match_number INTEGER PRIMARY KEY,
                round_name VARCHAR NOT NULL,
                group_name VARCHAR,
                match_date TIMESTAMP,
                home_team_id VARCHAR NOT NULL,
                home_team_name VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                away_team_name VARCHAR NOT NULL,
                home_win_pct DOUBLE NOT NULL,
                draw_pct DOUBLE NOT NULL,
                away_win_pct DOUBLE NOT NULL,
                predicted_outcome VARCHAR NOT NULL,
                calibration_temperature DOUBLE,
                model_path VARCHAR,
                created_at TIMESTAMP NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO outcome_predictions (
                match_number,
                round_name,
                group_name,
                match_date,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_win_pct,
                draw_pct,
                away_win_pct,
                predicted_outcome,
                calibration_temperature,
                model_path,
                created_at
            ) VALUES (
                1,
                'group_stage',
                'Group A',
                TIMESTAMP '2026-06-11 19:00:00',
                'BRA',
                'Brazil',
                'ARG',
                'Argentina',
                12.5,
                60.0,
                27.5,
                'draw',
                2.0,
                'models/xgb_outcome_model.json',
                current_timestamp
            )
            """
        )

    rows = _load_round_probabilities(
        str(db_path),
        (1,),
        show_all_matchups=True,
        dynamic_matchups=False,
    )

    assert len(rows) == 1
    assert rows[0]["predicted_home_goals"] == 1
    assert rows[0]["predicted_away_goals"] == 0
    assert rows[0]["score_occurrence_pct"] == 40.0
    assert rows[0]["home_win_pct"] == 12.5
    assert rows[0]["draw_pct"] == 60.0
    assert rows[0]["away_win_pct"] == 27.5
    assert rows[0]["probability_source"] == "outcome_model"
    assert _format_score(rows[0]) == "1 x 0"


def test_load_round_probabilities_can_use_simulation_source(tmp_path: Path) -> None:
    db_path = tmp_path / "world_cup.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE simulated_results (
                simulation_id INTEGER NOT NULL,
                match_number INTEGER NOT NULL,
                group_name VARCHAR,
                home_team_id VARCHAR NOT NULL,
                home_team_name VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                away_team_name VARCHAR NOT NULL,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL
            )
            """,
        )
        con.executemany(
            """
            INSERT INTO simulated_results (
                simulation_id,
                match_number,
                group_name,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_goals,
                away_goals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 2, 1),
                (2, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 2, 1),
                (3, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 0, 0),
                (4, 1, "Group A", "BRA", "Brazil", "ARG", "Argentina", 0, 1),
            ],
        )
        con.execute(
            """
            CREATE TABLE outcome_predictions (
                match_number INTEGER PRIMARY KEY,
                round_name VARCHAR NOT NULL,
                group_name VARCHAR,
                match_date TIMESTAMP,
                home_team_id VARCHAR NOT NULL,
                home_team_name VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                away_team_name VARCHAR NOT NULL,
                home_win_pct DOUBLE NOT NULL,
                draw_pct DOUBLE NOT NULL,
                away_win_pct DOUBLE NOT NULL,
                predicted_outcome VARCHAR NOT NULL,
                calibration_temperature DOUBLE,
                model_path VARCHAR,
                created_at TIMESTAMP NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO outcome_predictions (
                match_number,
                round_name,
                group_name,
                match_date,
                home_team_id,
                home_team_name,
                away_team_id,
                away_team_name,
                home_win_pct,
                draw_pct,
                away_win_pct,
                predicted_outcome,
                calibration_temperature,
                model_path,
                created_at
            ) VALUES (
                1,
                'group_stage',
                'Group A',
                TIMESTAMP '2026-06-11 19:00:00',
                'BRA',
                'Brazil',
                'ARG',
                'Argentina',
                1.0,
                98.0,
                1.0,
                'draw',
                2.0,
                'models/xgb_outcome_model.json',
                current_timestamp
            )
            """
        )

    rows = _load_round_probabilities(
        str(db_path),
        (1,),
        show_all_matchups=True,
        dynamic_matchups=False,
        prediction_source=PREDICTION_SOURCE_SIMULATION,
    )

    assert len(rows) == 1
    assert rows[0]["home_win_pct"] == 50.0
    assert rows[0]["draw_pct"] == 25.0
    assert rows[0]["away_win_pct"] == 25.0
    assert rows[0]["probability_source"] == "simulation"


def test_add_played_fixture_result_marks_winner_hit() -> None:
    row = {
        "match_number": 1,
        "home_team_id": "BRA",
        "away_team_id": "ARG",
        "predicted_home_goals": 2,
        "predicted_away_goals": 1,
    }
    fixture = WorldCupFixture(
        match_number=1,
        match_date=datetime(2026, 6, 11, tzinfo=UTC),
        round_name="group_stage",
        group_name="Group A",
        home_slot="BRA",
        away_slot="ARG",
        stadium="Test Stadium",
        city="Test City",
        country="USA",
        played_home_goals=1,
        played_away_goals=0,
    )

    _add_played_fixture_result(row, {1: fixture})

    assert _format_actual_score(row) == "1 x 0"
    assert row["prediction_result"] == "Acerto"


def test_add_played_fixture_result_ignores_non_matching_fixture_teams() -> None:
    row = {
        "match_number": 1,
        "home_team_id": "BRA",
        "away_team_id": "ARG",
        "predicted_home_goals": 2,
        "predicted_away_goals": 1,
    }
    fixture = WorldCupFixture(
        match_number=1,
        match_date=datetime(2026, 6, 11, tzinfo=UTC),
        round_name="group_stage",
        group_name="Group A",
        home_slot="ARG",
        away_slot="BRA",
        stadium="Test Stadium",
        city="Test City",
        country="USA",
        played_home_goals=0,
        played_away_goals=1,
    )

    _add_played_fixture_result(row, {1: fixture})

    assert _format_actual_score(row) == "-"
    assert row["prediction_result"] is None


def test_prediction_result_uses_most_likely_outcome_when_available() -> None:
    row = {
        "predicted_home_goals": 1,
        "predicted_away_goals": 0,
        "actual_home_goals": 1,
        "actual_away_goals": 1,
        "home_win_pct": 35.0,
        "draw_pct": 45.0,
        "away_win_pct": 20.0,
    }

    assert _prediction_result(row) == "Acerto"


def test_accuracy_summary_and_breakdowns_count_result_and_score_hits() -> None:
    rows = [
        {
            "outcome_hit": True,
            "score_hit": True,
            "confidence_pct": 62.0,
            "round_name": "group_stage",
            "group_round": 1,
        },
        {
            "outcome_hit": True,
            "score_hit": False,
            "confidence_pct": 48.0,
            "round_name": "group_stage",
            "group_round": 1,
        },
        {
            "outcome_hit": False,
            "score_hit": False,
            "confidence_pct": 55.0,
            "round_name": "round_of_32",
            "group_round": None,
        },
    ]

    summary = _accuracy_summary(rows)
    group_rows = _group_round_accuracy_rows(rows)
    phase_rows = _phase_accuracy_rows(rows)

    assert summary["played_count"] == 3
    assert summary["outcome_hits"] == 2
    assert summary["score_hits"] == 1
    assert summary["outcome_accuracy_pct"] == 66.66666666666667
    assert summary["score_accuracy_pct"] == 33.333333333333336
    assert summary["avg_confidence_pct"] == 55.0

    assert group_rows[0]["rodada"] == "Rodada 1"
    assert group_rows[0]["jogos"] == 2
    assert group_rows[0]["resultado_pct"] == 100.0
    assert group_rows[1]["jogos"] == 0

    assert phase_rows[0]["fase"] == "Fase de grupos"
    assert phase_rows[0]["jogos"] == 2
    assert phase_rows[0]["placar_pct"] == 50.0
    assert phase_rows[1]["fase"] == "16 avos de final"
    assert phase_rows[1]["jogos"] == 1


def test_filter_played_analysis_by_round_keeps_overall_or_selected_matches() -> None:
    rows = [
        {"match_number": 1, "outcome_hit": True},
        {"match_number": 2, "outcome_hit": False},
        {"match_number": 89, "outcome_hit": True},
    ]
    overall = RoundOption(
        key="overall",
        label="Geral",
        match_numbers=(1, 2, 89),
        first_column_header="Jogo",
    )
    group_round = RoundOption(
        key="group_stage_1",
        label="Fase de grupos - Rodada 1",
        match_numbers=(1, 2),
        first_column_header="Grupo",
    )

    assert _filter_played_analysis_by_round(rows, overall) == rows
    assert _filter_played_analysis_by_round(rows, group_round) == rows[:2]


def test_metric_cards_html_renders_labels_and_values_without_streamlit_metric() -> None:
    rendered = _metric_cards_html(
        [
            {"label": "Acerto de resultado", "value": "51.4%", "delta": "18 acertos"},
            {"label": "Placar exato", "value": "2.9%"},
        ],
    )

    assert 'class="metric-card"' in rendered
    assert 'class="metric-label">Acerto de resultado</div>' in rendered
    assert 'class="metric-value">51.4%</div>' in rendered
    assert 'class="metric-delta">18 acertos</div>' in rendered
    assert "stMetric" not in rendered


def test_match_prediction_cards_html_compares_prediction_with_actual_result() -> None:
    rendered = _match_prediction_cards_html(
        [
            {
                "match_number": 1,
                "bucket": "A",
                "home_team_id": "MEX",
                "home_team_name": "Mexico",
                "away_team_id": "RSA",
                "away_team_name": "Africa do Sul",
                "home_win_pct": 71.0,
                "draw_pct": 20.0,
                "away_win_pct": 9.0,
                "predicted_home_goals": 2,
                "predicted_away_goals": 0,
                "actual_home_goals": 2,
                "actual_away_goals": 0,
                "prediction_result": "Acerto",
                "probability_source": "outcome_model",
                "occurrence_pct": 88.0,
            }
        ],
        include_occurrence=True,
    )

    assert 'class="match-prediction-card"' in rendered
    assert "Jogo 1" in rendered
    assert "Ocorr. 88.00%" in rendered
    assert "Modelo previa" in rendered
    assert "71.0%" in rendered
    assert "2 x 0" in rendered
    assert "Acerto" in rendered
    assert "source-pill" in rendered


def test_match_prediction_cards_html_marks_pending_unplayed_match() -> None:
    rendered = _match_prediction_cards_html(
        [
            {
                "match_number": 2,
                "bucket": "A",
                "home_team_id": "BRA",
                "home_team_name": "Brasil",
                "away_team_id": "ARG",
                "away_team_name": "Argentina",
                "home_win_pct": 40.0,
                "draw_pct": 35.0,
                "away_win_pct": 25.0,
                "predicted_home_goals": 1,
                "predicted_away_goals": 1,
                "actual_home_goals": None,
                "actual_away_goals": None,
                "prediction_result": None,
                "probability_source": "simulation",
            }
        ],
        include_occurrence=False,
    )

    assert "Aguardando real" in rendered
    assert "vs" in rendered
    assert "A jogar" in rendered
    assert "Sim." in rendered


def test_home_projection_showcase_html_renders_favorite_and_challengers() -> None:
    rendered = _home_projection_showcase_html(
        [
            {
                "team_id": "FRA",
                "team_name": "Franca",
                "champion_pct": 24.5,
                "final_pct": 41.0,
                "semifinal_pct": 67.0,
                "round_of_16_pct": 100.0,
            },
            {
                "team_id": "ESP",
                "team_name": "Espanha",
                "champion_pct": 14.8,
                "final_pct": 25.0,
                "semifinal_pct": 42.0,
                "round_of_16_pct": 98.0,
            },
        ],
        {"simulation_count": 50000, "last_created_at": "2026-06-30 20:00"},
    )

    assert "Favorita do modelo" in rendered
    assert "Franca" in rendered
    assert "24.5%" in rendered
    assert "FNL" in rendered
    assert "41.0%" in rendered
    assert "Candidatas seguintes" in rendered
    assert "02" in rendered
    assert "Espanha" in rendered
    assert "FNL 25% - SF 42%" in rendered
    assert "50.000 simulacoes" in rendered


def test_actual_group_standings_rank_by_points_goal_diff_and_goals_for() -> None:
    fixtures = (
        WorldCupFixture(
            match_number=1,
            match_date=datetime(2026, 6, 11, tzinfo=UTC),
            round_name="group_stage",
            group_name="Group A",
            home_slot="BRA",
            away_slot="ARG",
            stadium="Test Stadium",
            city="Test City",
            country="USA",
            played_home_goals=2,
            played_away_goals=0,
        ),
        WorldCupFixture(
            match_number=2,
            match_date=datetime(2026, 6, 12, tzinfo=UTC),
            round_name="group_stage",
            group_name="Group A",
            home_slot="CAN",
            away_slot="MEX",
            stadium="Test Stadium",
            city="Test City",
            country="USA",
            played_home_goals=1,
            played_away_goals=1,
        ),
    )

    standings = _actual_group_standings(fixtures)
    third_rows = _third_place_rows(standings, predicted=False)

    assert [row["team_id"] for row in standings["A"]] == ["BRA", "CAN", "MEX", "ARG"]
    assert standings["A"][0]["points"] == 3
    assert standings["A"][0]["goal_diff"] == 2
    assert third_rows[0]["Selecao"] == "Mexico"
