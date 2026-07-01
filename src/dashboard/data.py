"""DuckDB-backed data access for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import duckdb

try:
    from ..world_cup_2026_schedule import WorldCupFixture
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`.
    from world_cup_2026_schedule import WorldCupFixture

from .analysis import (
    _add_played_fixture_result,
    _group_round_lookup,
    _group_rows_by_letter,
    _is_exact_score_hit,
    _most_likely_outcome,
    _outcome_probability,
    _score_outcome,
)
from .constants import (
    KNOCKOUT_MATCH_NUMBERS,
    PREDICTION_SOURCE_OUTCOME_MODEL,
    PREDICTION_SOURCE_SIMULATION,
    ROUND_DISPLAY_ORDER,
    ROUND_LABELS,
)
from .formatting import _team_display_name


def _load_summary(db_path: str) -> dict[str, Any]:
    with duckdb.connect(db_path, read_only=True) as con:
        _ensure_simulation_table(con)
        row = con.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT simulation_id) AS simulation_count,
                STRFTIME(MAX(created_at), '%Y-%m-%d %H:%M') AS last_created_at
            FROM simulated_results
            """,
        ).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "simulation_count": int(row[1] or 0),
        "last_created_at": row[2],
    }


def _load_round_probabilities(
    db_path: str,
    match_numbers: tuple[int, ...],
    *,
    show_all_matchups: bool,
    dynamic_matchups: bool,
    prediction_source: str = PREDICTION_SOURCE_OUTCOME_MODEL,
) -> list[dict[str, Any]]:
    placeholders = ", ".join("?" for _ in match_numbers)
    query = f"""
        WITH selected AS (
            SELECT *
            FROM simulated_results
            WHERE match_number IN ({placeholders})
        ),
        match_totals AS (
            SELECT match_number, COUNT(*) AS match_rows
            FROM selected
            GROUP BY match_number
        ),
        pair_stats AS (
            SELECT
                s.match_number,
                s.group_name,
                s.home_team_id,
                s.home_team_name,
                s.away_team_id,
                s.away_team_name,
                COUNT(*) AS appearances,
                ROUND(100.0 * COUNT(*) / NULLIF(MAX(mt.match_rows), 0), 2) AS occurrence_pct,
                ROUND(
                    100.0 * SUM(CASE WHEN s.home_goals > s.away_goals THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0),
                    2
                ) AS home_win_pct,
                ROUND(
                    100.0 * SUM(CASE WHEN s.home_goals = s.away_goals THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0),
                    2
                ) AS draw_pct,
                ROUND(
                    100.0 * SUM(CASE WHEN s.away_goals > s.home_goals THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0),
                    2
                ) AS away_win_pct
            FROM selected AS s
            JOIN match_totals AS mt USING (match_number)
            GROUP BY
                s.match_number,
                s.group_name,
                s.home_team_id,
                s.home_team_name,
                s.away_team_id,
                s.away_team_name
        ),
        score_stats AS (
            SELECT
                s.match_number,
                s.home_team_id,
                s.away_team_id,
                s.home_goals AS predicted_home_goals,
                s.away_goals AS predicted_away_goals,
                COUNT(*) AS score_appearances,
                ROW_NUMBER() OVER (
                    PARTITION BY s.match_number, s.home_team_id, s.away_team_id
                    ORDER BY COUNT(*) DESC, s.home_goals + s.away_goals ASC, s.home_goals DESC
                ) AS score_rank
            FROM selected AS s
            GROUP BY
                s.match_number,
                s.home_team_id,
                s.away_team_id,
                s.home_goals,
                s.away_goals
        )
        SELECT
            p.*,
            ss.predicted_home_goals,
            ss.predicted_away_goals,
            ROUND(100.0 * ss.score_appearances / NULLIF(p.appearances, 0), 2)
                AS score_occurrence_pct
        FROM pair_stats AS p
        LEFT JOIN score_stats AS ss
            ON ss.match_number = p.match_number
            AND ss.home_team_id = p.home_team_id
            AND ss.away_team_id = p.away_team_id
            AND ss.score_rank = 1
        ORDER BY match_number ASC, appearances DESC, home_team_name ASC, away_team_name ASC
    """
    with duckdb.connect(db_path, read_only=True) as con:
        _ensure_simulation_table(con)
        result = con.execute(query, match_numbers)
        columns = [column[0] for column in result.description]
        rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
        if prediction_source == PREDICTION_SOURCE_OUTCOME_MODEL:
            rows = _apply_outcome_prediction_overrides(con, rows, match_numbers)
        else:
            rows = _mark_probability_source(rows, PREDICTION_SOURCE_SIMULATION)

    if dynamic_matchups and not show_all_matchups:
        top_rows: list[dict[str, Any]] = []
        seen_match_numbers: set[int] = set()
        for row in rows:
            match_number = int(row["match_number"])
            if match_number in seen_match_numbers:
                continue
            top_rows.append(row)
            seen_match_numbers.add(match_number)
        return top_rows
    return rows


def _mark_probability_source(
    rows: list[dict[str, Any]],
    probability_source: str,
) -> list[dict[str, Any]]:
    for row in rows:
        row["probability_source"] = probability_source
    return rows


def _apply_outcome_prediction_overrides(
    con: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
    match_numbers: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Replace simulated V/E/D probabilities with direct outcome-model probabilities."""
    for row in rows:
        row["probability_source"] = "simulation"

    if not rows or not _outcome_predictions_available(con) or not match_numbers:
        return rows

    placeholders = ", ".join("?" for _ in match_numbers)
    result = con.execute(
        f"""
        SELECT
            match_number,
            home_team_id,
            away_team_id,
            home_win_pct,
            draw_pct,
            away_win_pct,
            predicted_outcome
        FROM outcome_predictions
        WHERE match_number IN ({placeholders})
        """,
        match_numbers,
    )
    overrides = {
        (
            int(row["match_number"]),
            str(row["home_team_id"]),
            str(row["away_team_id"]),
        ): row
        for row in _query_dicts(result)
    }

    for row in rows:
        override = overrides.get(
            (
                int(row["match_number"]),
                str(row["home_team_id"]),
                str(row["away_team_id"]),
            )
        )
        if override is None:
            continue
        row["home_win_pct"] = float(override["home_win_pct"])
        row["draw_pct"] = float(override["draw_pct"])
        row["away_win_pct"] = float(override["away_win_pct"])
        row["direct_predicted_outcome"] = override["predicted_outcome"]
        row["probability_source"] = "outcome_model"
    return rows


def _load_played_match_analysis(
    db_path: str,
    fixtures: tuple[WorldCupFixture, ...],
    *,
    prediction_source: str = PREDICTION_SOURCE_OUTCOME_MODEL,
) -> list[dict[str, Any]]:
    played_fixtures = tuple(
        fixture
        for fixture in fixtures
        if fixture.played_home_goals is not None and fixture.played_away_goals is not None
    )
    if not played_fixtures:
        return []

    rows = _load_round_probabilities(
        db_path,
        tuple(fixture.match_number for fixture in played_fixtures),
        show_all_matchups=True,
        dynamic_matchups=True,
        prediction_source=prediction_source,
    )
    fixtures_by_match = {fixture.match_number: fixture for fixture in fixtures}
    group_round_by_match = _group_round_lookup(fixtures)
    analysis_rows: list[dict[str, Any]] = []

    for row in rows:
        fixture = fixtures_by_match.get(int(row["match_number"]))
        if fixture is None:
            continue

        display_row = dict(row)
        _add_played_fixture_result(display_row, fixtures_by_match)
        if display_row.get("actual_home_goals") is None:
            continue

        predicted_outcome = _most_likely_outcome(display_row)
        actual_outcome = _score_outcome(
            int(display_row["actual_home_goals"]),
            int(display_row["actual_away_goals"]),
        )
        group_round = group_round_by_match.get(fixture.match_number)
        display_row.update(
            {
                "round_name": fixture.round_name,
                "phase_label": ROUND_LABELS.get(fixture.round_name, fixture.round_name),
                "phase_order": ROUND_DISPLAY_ORDER.get(fixture.round_name, 99),
                "group_round": group_round,
                "group_round_label": (
                    f"Rodada {group_round}" if group_round is not None else "Eliminatorias"
                ),
                "home_team_name": _team_display_name(
                    str(display_row["home_team_id"]),
                    str(display_row["home_team_name"]),
                ),
                "away_team_name": _team_display_name(
                    str(display_row["away_team_id"]),
                    str(display_row["away_team_name"]),
                ),
                "predicted_outcome": predicted_outcome,
                "actual_outcome": actual_outcome,
                "outcome_hit": predicted_outcome == actual_outcome,
                "score_hit": _is_exact_score_hit(display_row),
                "confidence_pct": _outcome_probability(display_row, predicted_outcome),
            },
        )
        analysis_rows.append(display_row)

    return sorted(analysis_rows, key=lambda row: int(row["match_number"]))


def _load_predicted_group_standings(db_path: str) -> dict[str, list[dict[str, Any]]]:
    query = """
        WITH team_match_rows AS (
            SELECT
                simulation_id,
                group_name,
                home_team_id AS team_id,
                home_team_name AS team_name,
                home_goals AS goals_for,
                away_goals AS goals_against,
                CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END AS wins,
                CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END AS draws,
                CASE WHEN home_goals < away_goals THEN 1 ELSE 0 END AS losses,
                CASE
                    WHEN home_goals > away_goals THEN 3
                    WHEN home_goals = away_goals THEN 1
                    ELSE 0
                END AS points
            FROM simulated_results
            WHERE round_name = 'group_stage'
            UNION ALL
            SELECT
                simulation_id,
                group_name,
                away_team_id AS team_id,
                away_team_name AS team_name,
                away_goals AS goals_for,
                home_goals AS goals_against,
                CASE WHEN away_goals > home_goals THEN 1 ELSE 0 END AS wins,
                CASE WHEN away_goals = home_goals THEN 1 ELSE 0 END AS draws,
                CASE WHEN away_goals < home_goals THEN 1 ELSE 0 END AS losses,
                CASE
                    WHEN away_goals > home_goals THEN 3
                    WHEN away_goals = home_goals THEN 1
                    ELSE 0
                END AS points
            FROM simulated_results
            WHERE round_name = 'group_stage'
        ),
        simulation_tables AS (
            SELECT
                simulation_id,
                group_name,
                team_id,
                ANY_VALUE(team_name) AS team_name,
                COUNT(*) AS played,
                SUM(wins) AS wins,
                SUM(draws) AS draws,
                SUM(losses) AS losses,
                SUM(goals_for) AS goals_for,
                SUM(goals_against) AS goals_against,
                SUM(goals_for) - SUM(goals_against) AS goal_diff,
                SUM(points) AS points
            FROM team_match_rows
            GROUP BY simulation_id, group_name, team_id
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY simulation_id, group_name
                    ORDER BY points DESC, goal_diff DESC, goals_for DESC, team_id ASC
                ) AS position
            FROM simulation_tables
        )
        SELECT
            group_name,
            team_id,
            ANY_VALUE(team_name) AS team_name,
            ROUND(AVG(played), 1) AS played,
            ROUND(AVG(wins), 1) AS wins,
            ROUND(AVG(draws), 1) AS draws,
            ROUND(AVG(losses), 1) AS losses,
            ROUND(AVG(goals_for), 1) AS goals_for,
            ROUND(AVG(goals_against), 1) AS goals_against,
            ROUND(AVG(goal_diff), 1) AS goal_diff,
            ROUND(AVG(points), 1) AS points,
            ROUND(AVG(position), 2) AS avg_position,
            ROUND(100.0 * AVG(CASE WHEN position = 1 THEN 1 ELSE 0 END), 1) AS first_pct,
            ROUND(100.0 * AVG(CASE WHEN position <= 2 THEN 1 ELSE 0 END), 1) AS top2_pct,
            ROUND(100.0 * AVG(CASE WHEN position = 3 THEN 1 ELSE 0 END), 1) AS third_pct
        FROM ranked
        GROUP BY group_name, team_id
        ORDER BY group_name ASC, avg_position ASC, points DESC, goal_diff DESC, goals_for DESC
    """
    with duckdb.connect(db_path, read_only=True) as con:
        _ensure_simulation_table(con)
        result = con.execute(query)
        rows = _query_dicts(result)
    return _group_rows_by_letter(rows)


def _load_global_projection(db_path: str) -> list[dict[str, Any]]:
    query = """
        WITH team_match_rows AS (
            SELECT
                simulation_id,
                group_name,
                home_team_id AS team_id,
                home_team_name AS team_name,
                home_goals AS goals_for,
                away_goals AS goals_against,
                CASE
                    WHEN home_goals > away_goals THEN 3
                    WHEN home_goals = away_goals THEN 1
                    ELSE 0
                END AS points
            FROM simulated_results
            WHERE round_name = 'group_stage'
            UNION ALL
            SELECT
                simulation_id,
                group_name,
                away_team_id AS team_id,
                away_team_name AS team_name,
                away_goals AS goals_for,
                home_goals AS goals_against,
                CASE
                    WHEN away_goals > home_goals THEN 3
                    WHEN away_goals = home_goals THEN 1
                    ELSE 0
                END AS points
            FROM simulated_results
            WHERE round_name = 'group_stage'
        ),
        simulation_tables AS (
            SELECT
                simulation_id,
                group_name,
                team_id,
                ANY_VALUE(team_name) AS team_name,
                SUM(goals_for) AS goals_for,
                SUM(goals_against) AS goals_against,
                SUM(goals_for) - SUM(goals_against) AS goal_diff,
                SUM(points) AS points
            FROM team_match_rows
            GROUP BY simulation_id, group_name, team_id
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY simulation_id, group_name
                    ORDER BY points DESC, goal_diff DESC, goals_for DESC, team_id ASC
                ) AS position
            FROM simulation_tables
        ),
        group_position AS (
            SELECT
                team_id,
                ANY_VALUE(team_name) AS team_name,
                ANY_VALUE(group_name) AS group_name,
                ROUND(AVG(position), 2) AS avg_group_position,
                ROUND(100.0 * AVG(CASE WHEN position = 1 THEN 1 ELSE 0 END), 1) AS group_1_pct,
                ROUND(100.0 * AVG(CASE WHEN position = 2 THEN 1 ELSE 0 END), 1) AS group_2_pct,
                ROUND(100.0 * AVG(CASE WHEN position = 3 THEN 1 ELSE 0 END), 1) AS group_3_pct,
                ROUND(100.0 * AVG(CASE WHEN position = 4 THEN 1 ELSE 0 END), 1) AS group_4_pct
            FROM ranked
            GROUP BY team_id
        ),
        round_participants AS (
            SELECT simulation_id, round_name, home_team_id AS team_id
            FROM simulated_results
            WHERE round_name != 'group_stage'
            UNION ALL
            SELECT simulation_id, round_name, away_team_id AS team_id
            FROM simulated_results
            WHERE round_name != 'group_stage'
        ),
        simulation_count AS (
            SELECT COUNT(DISTINCT simulation_id) AS total_simulations
            FROM simulated_results
        ),
        round_reach AS (
            SELECT
                team_id,
                ROUND(100.0 * COUNT(DISTINCT CASE
                    WHEN round_name = 'round_of_32' THEN simulation_id
                END)
                    / (SELECT total_simulations FROM simulation_count), 1) AS round_of_32_pct,
                ROUND(100.0 * COUNT(DISTINCT CASE
                    WHEN round_name = 'round_of_16' THEN simulation_id
                END)
                    / (SELECT total_simulations FROM simulation_count), 1) AS round_of_16_pct,
                ROUND(100.0 * COUNT(DISTINCT CASE
                    WHEN round_name = 'quarterfinal' THEN simulation_id
                END)
                    / (SELECT total_simulations FROM simulation_count), 1) AS quarterfinal_pct,
                ROUND(100.0 * COUNT(DISTINCT CASE
                    WHEN round_name = 'semifinal' THEN simulation_id
                END)
                    / (SELECT total_simulations FROM simulation_count), 1) AS semifinal_pct,
                ROUND(100.0 * COUNT(DISTINCT CASE
                    WHEN round_name = 'final' THEN simulation_id
                END)
                    / (SELECT total_simulations FROM simulation_count), 1) AS final_pct
            FROM round_participants
            GROUP BY team_id
        ),
        finalists AS (
            SELECT
                simulation_id,
                winner_team_id AS champion_team_id,
                runner_up_team_id AS runner_up_team_id
            FROM simulated_results
            WHERE match_number = 104
        ),
        champion AS (
            SELECT
                champion_team_id AS team_id,
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM finalists), 1) AS champion_pct
            FROM finalists
            GROUP BY champion_team_id
        ),
        runner_up AS (
            SELECT
                runner_up_team_id AS team_id,
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM finalists), 1) AS runner_up_pct
            FROM finalists
            GROUP BY runner_up_team_id
        )
        SELECT
            gp.team_id,
            gp.team_name,
            gp.group_name,
            gp.avg_group_position,
            gp.group_1_pct,
            gp.group_2_pct,
            gp.group_3_pct,
            gp.group_4_pct,
            COALESCE(rr.round_of_32_pct, 0) AS round_of_32_pct,
            COALESCE(rr.round_of_16_pct, 0) AS round_of_16_pct,
            COALESCE(rr.quarterfinal_pct, 0) AS quarterfinal_pct,
            COALESCE(rr.semifinal_pct, 0) AS semifinal_pct,
            COALESCE(rr.final_pct, 0) AS final_pct,
            COALESCE(ru.runner_up_pct, 0) AS runner_up_pct,
            COALESCE(c.champion_pct, 0) AS champion_pct
        FROM group_position AS gp
        LEFT JOIN round_reach AS rr USING (team_id)
        LEFT JOIN runner_up AS ru USING (team_id)
        LEFT JOIN champion AS c USING (team_id)
        ORDER BY champion_pct DESC, final_pct DESC, semifinal_pct DESC, team_name ASC
    """
    with duckdb.connect(db_path, read_only=True) as con:
        _ensure_simulation_table(con)
        result = con.execute(query)
        rows = _query_dicts(result)
    for row in rows:
        row["team_name"] = _team_display_name(str(row["team_id"]), str(row["team_name"]))
        row["group_letter"] = str(row["group_name"]).rsplit(" ", maxsplit=1)[-1]
    return rows


def _load_knockout_bracket(db_path: str) -> list[dict[str, Any]]:
    match_numbers = KNOCKOUT_MATCH_NUMBERS
    placeholders = ", ".join("?" for _ in match_numbers)
    query = f"""
        WITH selected AS (
            SELECT *
            FROM simulated_results
            WHERE match_number IN ({placeholders})
        ),
        match_totals AS (
            SELECT match_number, COUNT(*) AS match_rows
            FROM selected
            GROUP BY match_number
        ),
        pair_stats AS (
            SELECT
                s.match_number,
                ANY_VALUE(s.round_name) AS round_name,
                ANY_VALUE(s.match_date) AS match_date,
                s.home_team_id,
                ANY_VALUE(s.home_team_name) AS home_team_name,
                s.away_team_id,
                ANY_VALUE(s.away_team_name) AS away_team_name,
                COUNT(*) AS appearances,
                ROUND(100.0 * COUNT(*) / MAX(mt.match_rows), 1) AS occurrence_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY s.match_number
                    ORDER BY COUNT(*) DESC, s.home_team_id ASC, s.away_team_id ASC
                ) AS pair_rank
            FROM selected AS s
            JOIN match_totals AS mt USING (match_number)
            GROUP BY s.match_number, s.home_team_id, s.away_team_id
        ),
        winner_stats AS (
            SELECT
                match_number,
                home_team_id,
                away_team_id,
                winner_team_id,
                COUNT(*) AS winner_count,
                ROW_NUMBER() OVER (
                    PARTITION BY match_number, home_team_id, away_team_id
                    ORDER BY COUNT(*) DESC, winner_team_id ASC
                ) AS winner_rank
            FROM selected
            GROUP BY match_number, home_team_id, away_team_id, winner_team_id
        ),
        score_stats AS (
            SELECT
                match_number,
                home_team_id,
                away_team_id,
                home_goals AS predicted_home_goals,
                away_goals AS predicted_away_goals,
                COUNT(*) AS score_count,
                ROW_NUMBER() OVER (
                    PARTITION BY match_number, home_team_id, away_team_id
                    ORDER BY COUNT(*) DESC, home_goals + away_goals ASC, home_goals DESC
                ) AS score_rank
            FROM selected
            GROUP BY match_number, home_team_id, away_team_id, home_goals, away_goals
        )
        SELECT
            p.*,
            ws.winner_team_id,
            ROUND(100.0 * ws.winner_count / NULLIF(p.appearances, 0), 1) AS winner_pct,
            ss.predicted_home_goals,
            ss.predicted_away_goals
        FROM pair_stats AS p
        LEFT JOIN winner_stats AS ws
            ON ws.match_number = p.match_number
            AND ws.home_team_id = p.home_team_id
            AND ws.away_team_id = p.away_team_id
            AND ws.winner_rank = 1
        LEFT JOIN score_stats AS ss
            ON ss.match_number = p.match_number
            AND ss.home_team_id = p.home_team_id
            AND ss.away_team_id = p.away_team_id
            AND ss.score_rank = 1
        WHERE p.pair_rank = 1
        ORDER BY p.match_number ASC
    """
    with duckdb.connect(db_path, read_only=True) as con:
        _ensure_simulation_table(con)
        result = con.execute(query, match_numbers)
        rows = _query_dicts(result)
    for row in rows:
        row["home_team_name"] = _team_display_name(
            str(row["home_team_id"]),
            str(row["home_team_name"]),
        )
        row["away_team_name"] = _team_display_name(
            str(row["away_team_id"]),
            str(row["away_team_name"]),
        )
        winner_id = str(row["winner_team_id"])
        row["winner_name"] = (
            row["home_team_name"] if winner_id == row["home_team_id"] else row["away_team_name"]
        )
    return rows


def _query_dicts(result: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [column[0] for column in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _ensure_simulation_table(con: duckdb.DuckDBPyConnection) -> None:
    table_exists = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'simulated_results'
        """,
    ).fetchone()[0]
    if int(table_exists or 0) == 0:
        raise RuntimeError("A tabela simulated_results nao existe. Rode `uv run simulate`.")

    row_count = con.execute("SELECT COUNT(*) FROM simulated_results").fetchone()[0]
    if int(row_count or 0) == 0:
        raise RuntimeError("A tabela simulated_results esta vazia. Rode `uv run simulate`.")


def _outcome_predictions_available(con: duckdb.DuckDBPyConnection) -> bool:
    table_exists = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'outcome_predictions'
        """,
    ).fetchone()[0]
    if int(table_exists or 0) == 0:
        return False
    row_count = con.execute("SELECT COUNT(*) FROM outcome_predictions").fetchone()[0]
    return int(row_count or 0) > 0


def _outcome_predictions_available_for_path(db_path: str) -> bool:
    with duckdb.connect(db_path, read_only=True) as con:
        return _outcome_predictions_available(con)
