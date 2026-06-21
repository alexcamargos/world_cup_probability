"""Streamlit dashboard for World Cup simulation probabilities."""

from __future__ import annotations

import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import streamlit as st

try:
    from .world_cup_2026_schedule import WorldCupFixture, world_cup_2026_fixtures
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`.
    from world_cup_2026_schedule import WorldCupFixture, world_cup_2026_fixtures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "world_cup.duckdb"

ROUND_ORDER = {
    "round_of_32": 4,
    "round_of_16": 5,
    "quarterfinal": 6,
    "semifinal": 7,
    "third_place": 8,
    "final": 9,
}

ROUND_LABELS = {
    "round_of_32": "16 avos de final",
    "round_of_16": "Oitavas de final",
    "quarterfinal": "Quartas de final",
    "semifinal": "Semifinais",
    "third_place": "Disputa de 3o lugar",
    "final": "Final",
}

TEAM_NAMES_PT_BR = {
    "ALG": "Argelia",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgica",
    "BIH": "Bosnia e Herzegovina",
    "BRA": "Brasil",
    "CAN": "Canada",
    "CIV": "Costa do Marfim",
    "COD": "RD Congo",
    "COL": "Colombia",
    "CPV": "Cabo Verde",
    "CRO": "Croacia",
    "CUW": "Curacao",
    "CZE": "Republica Tcheca",
    "ECU": "Equador",
    "EGY": "Egito",
    "ENG": "Inglaterra",
    "ESP": "Espanha",
    "FRA": "Franca",
    "GER": "Alemanha",
    "GHA": "Gana",
    "HAI": "Haiti",
    "IRN": "Ira",
    "IRQ": "Iraque",
    "JOR": "Jordania",
    "JPN": "Japao",
    "KOR": "Coreia do Sul",
    "KSA": "Arabia Saudita",
    "MAR": "Marrocos",
    "MEX": "Mexico",
    "NED": "Paises Baixos",
    "NOR": "Noruega",
    "NZL": "Nova Zelandia",
    "PAN": "Panama",
    "PAR": "Paraguai",
    "POR": "Portugal",
    "QAT": "Catar",
    "RSA": "Africa do Sul",
    "SCO": "Escocia",
    "SEN": "Senegal",
    "SUI": "Suica",
    "SWE": "Suecia",
    "TUN": "Tunisia",
    "TUR": "Turquia",
    "URU": "Uruguai",
    "USA": "Estados Unidos",
    "UZB": "Uzbequistao",
}


@dataclass(frozen=True, slots=True)
class RoundOption:
    """Dashboard option for one displayable tournament round."""

    key: str
    label: str
    match_numbers: tuple[int, ...]
    first_column_header: str
    dynamic_matchups: bool = False


def cli() -> None:
    """Launch the Streamlit app from the project script."""
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Probabilidades da Copa",
        layout="wide",
    )
    _inject_styles()

    st.title("Probabilidades da Copa do Mundo")
    st.caption("Consulta dos resultados agregados das simulações Monte Carlo por rodada.")

    db_path = Path(
        st.sidebar.text_input(
            "Arquivo DuckDB",
            value=str(DB_PATH),
            help="Banco local preenchido pelo comando `uv run simulate`.",
        ),
    )

    if not db_path.exists():
        st.error(f"Banco de dados nao encontrado: {db_path}")
        st.stop()

    try:
        summary = _load_summary(str(db_path))
    except Exception as exc:  # noqa: BLE001 - Streamlit should show operational failures.
        st.error(_friendly_db_error(exc, db_path))
        st.stop()

    round_options = _build_round_options(world_cup_2026_fixtures())
    selected_round = st.sidebar.selectbox(
        "Rodada",
        round_options,
        format_func=lambda option: option.label,
    )
    assert selected_round is not None

    if selected_round.dynamic_matchups:
        show_all_matchups = st.sidebar.checkbox(
            "Mostrar todos os confrontos simulados",
            value=False,
            help="Quando desmarcado, mostra apenas o confronto mais frequente de cada jogo.",
        )
    else:
        show_all_matchups = True

    st.sidebar.divider()
    st.sidebar.metric("Simulacoes", f"{summary['simulation_count']:,}".replace(",", "."))
    st.sidebar.metric("Jogos simulados", f"{summary['row_count']:,}".replace(",", "."))

    metric_cols = st.columns(3)
    metric_cols[0].metric("Rodada selecionada", selected_round.label)
    metric_cols[1].metric("Jogos na rodada", len(selected_round.match_numbers))
    metric_cols[2].metric("Ultima simulacao", summary["last_created_at"] or "-")

    try:
        rows = _load_round_probabilities(
            str(db_path),
            selected_round.match_numbers,
            show_all_matchups=show_all_matchups,
            dynamic_matchups=selected_round.dynamic_matchups,
        )
    except Exception as exc:  # noqa: BLE001 - Streamlit should show operational failures.
        st.error(_friendly_db_error(exc, db_path))
        st.stop()

    if not rows:
        st.warning("Nao ha resultados de simulacao para a rodada selecionada.")
        st.stop()

    display_rows = _localize_rows(rows, selected_round)
    st.markdown(
        _probability_table_html(
            display_rows,
            first_column_header=selected_round.first_column_header,
            include_occurrence=selected_round.dynamic_matchups,
        ),
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
def _load_round_probabilities(
    db_path: str,
    match_numbers: tuple[int, ...],
    *,
    show_all_matchups: bool,
    dynamic_matchups: bool,
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


def _build_round_options(fixtures: tuple[WorldCupFixture, ...]) -> list[RoundOption]:
    group_rounds = _group_round_match_numbers(fixtures)
    options = [
        RoundOption(
            key=f"group_stage_{round_number}",
            label=f"Fase de grupos - Rodada {round_number}",
            match_numbers=tuple(match_numbers),
            first_column_header="Grupo",
        )
        for round_number, match_numbers in sorted(group_rounds.items())
    ]

    knockout_fixtures = sorted(
        (fixture for fixture in fixtures if fixture.round_name != "group_stage"),
        key=lambda fixture: (ROUND_ORDER[fixture.round_name], fixture.match_number),
    )
    by_round: dict[str, list[int]] = {}
    for fixture in knockout_fixtures:
        by_round.setdefault(fixture.round_name, []).append(fixture.match_number)

    for round_name, match_numbers in sorted(
        by_round.items(), key=lambda item: ROUND_ORDER[item[0]]
    ):
        options.append(
            RoundOption(
                key=round_name,
                label=ROUND_LABELS[round_name],
                match_numbers=tuple(match_numbers),
                first_column_header="Jogo",
                dynamic_matchups=True,
            ),
        )
    return options


def _group_round_match_numbers(fixtures: tuple[WorldCupFixture, ...]) -> dict[int, list[int]]:
    by_group: dict[str, list[WorldCupFixture]] = {}
    for fixture in fixtures:
        if fixture.round_name != "group_stage" or fixture.group_name is None:
            continue
        group_letter = fixture.group_name.rsplit(" ", maxsplit=1)[-1]
        by_group.setdefault(group_letter, []).append(fixture)

    round_matches: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for group_letter in sorted(by_group):
        ordered_group_fixtures = sorted(
            by_group[group_letter],
            key=lambda fixture: (fixture.match_date, fixture.match_number),
        )
        for index, fixture in enumerate(ordered_group_fixtures):
            round_number = index // 2 + 1
            round_matches[round_number].append(fixture.match_number)

    for match_numbers in round_matches.values():
        match_numbers.sort()
    return round_matches


def _localize_rows(rows: list[dict[str, Any]], selected_round: RoundOption) -> list[dict[str, Any]]:
    fixtures_by_match = {fixture.match_number: fixture for fixture in world_cup_2026_fixtures()}
    localized = []
    for row in rows:
        display_row = dict(row)
        display_row["home_team_name"] = _team_display_name(
            row["home_team_id"], row["home_team_name"]
        )
        display_row["away_team_name"] = _team_display_name(
            row["away_team_id"], row["away_team_name"]
        )
        display_row["bucket"] = _bucket_label(row, selected_round)
        _add_played_fixture_result(display_row, fixtures_by_match)
        localized.append(display_row)
    if selected_round.first_column_header == "Grupo":
        localized.sort(key=lambda row: (str(row["bucket"]), int(row["match_number"])))
    return localized


def _bucket_label(row: dict[str, Any], selected_round: RoundOption) -> str:
    if selected_round.first_column_header == "Grupo" and row.get("group_name"):
        return str(row["group_name"]).rsplit(" ", maxsplit=1)[-1]
    return str(row["match_number"])


def _team_display_name(team_id: str, fallback: str) -> str:
    return TEAM_NAMES_PT_BR.get(team_id, fallback)


def _add_played_fixture_result(
    row: dict[str, Any],
    fixtures_by_match: dict[int, WorldCupFixture],
) -> None:
    fixture = fixtures_by_match.get(int(row["match_number"]))
    row["actual_home_goals"] = None
    row["actual_away_goals"] = None
    row["prediction_result"] = None
    if (
        fixture is None
        or fixture.played_home_goals is None
        or fixture.played_away_goals is None
        or fixture.home_slot != row["home_team_id"]
        or fixture.away_slot != row["away_team_id"]
    ):
        return

    row["actual_home_goals"] = fixture.played_home_goals
    row["actual_away_goals"] = fixture.played_away_goals
    row["prediction_result"] = _prediction_result(row)


def _prediction_result(row: dict[str, Any]) -> str | None:
    predicted_home_goals = row.get("predicted_home_goals")
    predicted_away_goals = row.get("predicted_away_goals")
    actual_home_goals = row.get("actual_home_goals")
    actual_away_goals = row.get("actual_away_goals")
    if (
        predicted_home_goals is None
        or predicted_away_goals is None
        or actual_home_goals is None
        or actual_away_goals is None
    ):
        return None

    predicted_outcome = _score_outcome(int(predicted_home_goals), int(predicted_away_goals))
    actual_outcome = _score_outcome(int(actual_home_goals), int(actual_away_goals))
    return "Acerto" if predicted_outcome == actual_outcome else "Erro"


def _score_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def _probability_table_html(
    rows: list[dict[str, Any]],
    *,
    first_column_header: str,
    include_occurrence: bool,
) -> str:
    header_cells = [
        _header_cell(first_column_header),
        _header_cell("Time 1"),
    ]
    if include_occurrence:
        header_cells.append(_header_cell("Ocorr. (%)"))
    header_cells.extend(
        [
            _header_cell("Placar<br>modelo"),
            _header_cell("Placar<br>real"),
            _header_cell("Modelo<br>vs real"),
            _header_cell("Vitoria<br>Time 1 (%)"),
            _header_cell("Empate (%)"),
            _header_cell("Vitoria<br>Time 2 (%)"),
            _header_cell("Time 2"),
        ],
    )

    rowspans = _bucket_rowspans(rows)
    emitted_buckets: set[str] = set()
    body_rows = []
    for index, row in enumerate(rows):
        classes = "even-row" if index % 2 else "odd-row"
        cells = []
        bucket = str(row["bucket"])
        if bucket not in emitted_buckets:
            cells.append(
                f'<td class="bucket-cell" rowspan="{rowspans[bucket]}">{html.escape(bucket)}</td>',
            )
            emitted_buckets.add(bucket)
        cells.append(f'<td class="team-cell">{html.escape(str(row["home_team_name"]))}</td>')
        if include_occurrence:
            cells.append(f'<td class="numeric-cell">{_format_pct(row["occurrence_pct"])}</td>')
        cells.extend(
            [
                f'<td class="score-cell">{html.escape(_format_score(row))}</td>',
                (
                    '<td class="score-cell actual-score">'
                    f"{html.escape(_format_actual_score(row))}</td>"
                ),
                _prediction_result_cell(row),
                f'<td class="numeric-cell">{_format_pct(row["home_win_pct"])}</td>',
                f'<td class="numeric-cell">{_format_pct(row["draw_pct"])}</td>',
                f'<td class="numeric-cell">{_format_pct(row["away_win_pct"])}</td>',
                f'<td class="team-cell">{html.escape(str(row["away_team_name"]))}</td>',
            ],
        )
        body_rows.append(f'<tr class="{classes}">{"".join(cells)}</tr>')

    return f"""
    <div class="prob-table-wrap">
        <table class="prob-table">
            <thead><tr>{"".join(header_cells)}</tr></thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
    </div>
    """


def _header_cell(label: str) -> str:
    return f"<th><span>{label.upper()}</span></th>"


def _bucket_rowspans(rows: list[dict[str, Any]]) -> dict[str, int]:
    rowspans: dict[str, int] = {}
    for row in rows:
        bucket = str(row["bucket"])
        rowspans[bucket] = rowspans.get(bucket, 0) + 1
    return rowspans


def _format_pct(value: Any) -> str:
    return f"{float(value or 0):.2f}"


def _format_score(row: dict[str, Any]) -> str:
    home_goals = row.get("predicted_home_goals")
    away_goals = row.get("predicted_away_goals")
    if home_goals is None or away_goals is None:
        return "-"
    return f"{int(home_goals)} x {int(away_goals)}"


def _format_actual_score(row: dict[str, Any]) -> str:
    home_goals = row.get("actual_home_goals")
    away_goals = row.get("actual_away_goals")
    if home_goals is None or away_goals is None:
        return "-"
    return f"{int(home_goals)} x {int(away_goals)}"


def _prediction_result_cell(row: dict[str, Any]) -> str:
    result = row.get("prediction_result")
    if result is None:
        return '<td class="result-cell result-pending">-</td>'
    css_class = "result-hit" if result == "Acerto" else "result-miss"
    return f'<td class="result-cell {css_class}">{html.escape(str(result))}</td>'


def _friendly_db_error(exc: Exception, db_path: Path) -> str:
    message = str(exc)
    if "used by another process" in message or "sendo usado por outro processo" in message:
        return (
            f"Nao foi possivel abrir {db_path}. O arquivo DuckDB esta em uso por outro processo; "
            "aguarde a simulacao terminar ou feche o processo que esta com o banco aberto."
        )
    return message


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
        }
        .prob-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin-top: 1rem;
        }
        .prob-table {
            width: 100%;
            min-width: 1180px;
            border-collapse: collapse;
            font-size: 16px;
            line-height: 1.25;
        }
        .prob-table th {
            background: #1d2127;
            color: #ffffff;
            border: 1px solid #ffffff;
            padding: 7px 10px;
            text-align: center;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .prob-table th span {
            display: inline-block;
            background: #000000;
            padding: 2px 4px;
        }
        .prob-table td {
            border: 1px solid #ffffff;
            padding: 9px 10px;
            color: #53617d;
            background: #f8f8f8;
        }
        .prob-table tr.even-row td:not(.bucket-cell) {
            background: #e9e9e9;
        }
        .prob-table .bucket-cell {
            width: 72px;
            min-width: 72px;
            text-align: center;
            vertical-align: middle;
            background: #2d4053;
            color: #ffffff;
            font-weight: 800;
            font-size: 18px;
        }
        .prob-table .team-cell {
            min-width: 190px;
            font-weight: 500;
        }
        .prob-table .numeric-cell {
            min-width: 120px;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }
        .prob-table .score-cell {
            min-width: 110px;
            text-align: center;
            color: #1d2127;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
        }
        .prob-table .actual-score {
            color: #38445c;
        }
        .prob-table .result-cell {
            min-width: 118px;
            text-align: center;
            font-weight: 800;
        }
        .prob-table .result-hit {
            color: #11683b;
        }
        .prob-table .result-miss {
            color: #a93535;
        }
        .prob-table .result-pending {
            color: #8993a5;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
