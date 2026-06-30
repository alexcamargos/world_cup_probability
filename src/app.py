"""Streamlit dashboard for World Cup simulation probabilities."""

from __future__ import annotations

import html
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import duckdb
import polars as pl
import streamlit as st

try:
    from .live_results import LiveFixtureSnapshot, load_world_cup_fixture_snapshot
    from .settings import DB_PATH
    from .world_cup_2026_schedule import WorldCupFixture
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`.
    from live_results import LiveFixtureSnapshot, load_world_cup_fixture_snapshot
    from settings import DB_PATH
    from world_cup_2026_schedule import WorldCupFixture

ROUND_ORDER = {
    "round_of_32": 4,
    "round_of_16": 5,
    "quarterfinal": 6,
    "semifinal": 7,
    "third_place": 8,
    "final": 9,
}

ROUND_LABELS = {
    "group_stage": "Fase de grupos",
    "round_of_32": "16 avos de final",
    "round_of_16": "Oitavas de final",
    "quarterfinal": "Quartas de final",
    "semifinal": "Semifinais",
    "third_place": "Disputa de 3o lugar",
    "final": "Final",
}

ROUND_DISPLAY_ORDER = {
    "group_stage": 1,
    "round_of_32": 2,
    "round_of_16": 3,
    "quarterfinal": 4,
    "semifinal": 5,
    "third_place": 6,
    "final": 7,
}

KNOCKOUT_MATCH_NUMBERS = tuple(range(73, 105))

PREDICTION_SOURCE_SIMULATION = "simulation"
PREDICTION_SOURCE_OUTCOME_MODEL = "outcome_model"
PREDICTION_SOURCE_LABELS = {
    PREDICTION_SOURCE_SIMULATION: "Simulacao",
    PREDICTION_SOURCE_OUTCOME_MODEL: "Modelo V/E/D",
}

BRACKET_ROUNDS = (
    ("round_of_32", "16 avos", tuple(range(73, 89))),
    ("round_of_16", "Oitavas", tuple(range(89, 97))),
    ("quarterfinal", "Quartas", tuple(range(97, 101))),
    ("semifinal", "Semifinais", (101, 102)),
    ("final", "Final", (104,)),
    ("third_place", "3o lugar", (103,)),
)

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


def _load_dashboard_fixture_snapshot() -> LiveFixtureSnapshot:
    st.sidebar.subheader("Dados reais")
    auto_refresh = st.sidebar.checkbox(
        "Buscar placares ausentes",
        value=True,
        help=(
            "Consulta a API publica da FIFA quando ja existem jogos passados sem placar "
            "no cache local."
        ),
    )
    force_refresh = st.sidebar.button("Buscar dados atualizados")

    try:
        return load_world_cup_fixture_snapshot(
            fetch_if_needed=auto_refresh,
            force=force_refresh,
        )
    except Exception as exc:  # noqa: BLE001 - dashboard should keep the static fallback usable.
        st.sidebar.warning(
            "Nao foi possivel buscar placares atualizados. Usando o cache local ou "
            f"calendario estatico. Detalhe: {exc}",
        )
        return load_world_cup_fixture_snapshot(fetch_if_needed=False)


def _render_fixture_refresh_status(snapshot: LiveFixtureSnapshot) -> None:
    if snapshot.fetched_at is None:
        source_label = "Calendario estatico"
        updated_at = "-"
    else:
        source_label = "FIFA API" if snapshot.refreshed else "Cache FIFA"
        updated_at = snapshot.fetched_at.strftime("%Y-%m-%d %H:%M UTC")

    status_rows = [
        {"label": "Fonte placares", "value": source_label},
        {"label": "Jogos com placar", "value": snapshot.scored_fixture_count},
        {"label": "Atualizado em", "value": updated_at},
    ]
    if snapshot.stale_missing_fixture_count:
        status_rows.append(
            {"label": "Passados sem placar", "value": snapshot.stale_missing_fixture_count}
        )

    st.sidebar.markdown(
        _metric_cards_html(status_rows, class_name="sidebar-metric-stack"),
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Analise do Modelo - Copa 2026",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.markdown(
        """
        <section class="dashboard-hero">
            <div>
                <p class="eyebrow">World Cup Probability</p>
                <h1>Analise das previsoes da Copa do Mundo 2026</h1>
                <p>
                    Painel para acompanhar probabilidades simuladas, acuracia do vencedor
                    e aderencia dos placares conforme os jogos reais sao preenchidos.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

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

    fixture_snapshot = _load_dashboard_fixture_snapshot()
    fixtures = fixture_snapshot.fixtures
    _render_fixture_refresh_status(fixture_snapshot)

    try:
        summary = _load_summary(str(db_path))
    except Exception as exc:  # noqa: BLE001 - Streamlit should show operational failures.
        st.error(_friendly_db_error(exc, db_path))
        st.stop()

    prediction_source = _render_prediction_source_selector(db_path)

    try:
        played_analysis = _load_played_match_analysis(
            str(db_path),
            fixtures,
            prediction_source=prediction_source,
        )
    except Exception as exc:  # noqa: BLE001 - Streamlit should show operational failures.
        st.error(_friendly_db_error(exc, db_path))
        st.stop()

    try:
        group_projection = _load_predicted_group_standings(str(db_path))
        global_projection = _load_global_projection(str(db_path))
        bracket_projection = _load_knockout_bracket(str(db_path))
    except Exception as exc:  # noqa: BLE001 - Streamlit should show operational failures.
        st.error(_friendly_db_error(exc, db_path))
        st.stop()

    actual_group_standings = _actual_group_standings(fixtures)

    round_options = _build_round_options(fixtures)
    selected_round = st.sidebar.selectbox(
        "Rodada",
        round_options,
        format_func=lambda option: option.label,
    )
    assert selected_round is not None
    scoped_played_analysis = _filter_played_analysis_by_round(played_analysis, selected_round)

    if selected_round.key != "overall" and selected_round.dynamic_matchups:
        show_all_matchups = st.sidebar.checkbox(
            "Mostrar todos os confrontos simulados",
            value=False,
            help="Quando desmarcado, mostra apenas o confronto mais frequente de cada jogo.",
        )
    else:
        show_all_matchups = True

    st.sidebar.divider()
    st.sidebar.markdown(
        _metric_cards_html(
            [
                {
                    "label": "Simulacoes",
                    "value": f"{summary['simulation_count']:,}".replace(",", "."),
                },
                {
                    "label": "Jogos simulados",
                    "value": f"{summary['row_count']:,}".replace(",", "."),
                },
                {"label": "Jogos reais avaliados", "value": len(scoped_played_analysis)},
            ],
            class_name="sidebar-metric-stack",
        ),
        unsafe_allow_html=True,
    )

    rows: list[dict[str, Any]] = []
    if selected_round.key != "overall":
        try:
            rows = _load_round_probabilities(
                str(db_path),
                selected_round.match_numbers,
                show_all_matchups=show_all_matchups,
                dynamic_matchups=selected_round.dynamic_matchups,
                prediction_source=prediction_source,
            )
        except Exception as exc:  # noqa: BLE001 - Streamlit should show operational failures.
            st.error(_friendly_db_error(exc, db_path))
            st.stop()

    _render_accuracy_overview(summary, scoped_played_analysis, selected_round, prediction_source)

    tab_overview, tab_groups, tab_global, tab_bracket, tab_probabilities, tab_matches = st.tabs(
        [
            "Visao geral",
            "Grupos",
            "Prognostico",
            "Chaveamento",
            "Probabilidades",
            "Jogos avaliados",
        ],
    )

    with tab_overview:
        _render_accuracy_sections(scoped_played_analysis)

    with tab_groups:
        _render_groups_section(group_projection, actual_group_standings)

    with tab_global:
        _render_global_projection_section(global_projection, summary)

    with tab_bracket:
        _render_bracket_section(bracket_projection)

    with tab_probabilities:
        _render_round_probability_section(
            selected_round=selected_round,
            rows=rows,
            fixtures=fixtures,
        )

    with tab_matches:
        _render_played_matches_section(scoped_played_analysis)


def _render_accuracy_overview(
    summary: dict[str, Any],
    played_analysis: list[dict[str, Any]],
    selected_round: RoundOption,
    prediction_source: str,
) -> None:
    totals = _accuracy_summary(played_analysis)
    st.markdown(
        _metric_cards_html(
            [
                {
                    "label": "Fonte analisada",
                    "value": PREDICTION_SOURCE_LABELS[prediction_source],
                },
                {
                    "label": "Jogos reais avaliados",
                    "value": totals["played_count"],
                    "delta": selected_round.label,
                },
                {
                    "label": "Acerto de resultado",
                    "value": _format_metric_pct(totals["outcome_accuracy_pct"]),
                    "delta": f"{totals['outcome_hits']} acertos",
                },
                {
                    "label": "Placar exato",
                    "value": _format_metric_pct(totals["score_accuracy_pct"]),
                    "delta": f"{totals['score_hits']} acertos",
                },
                {
                    "label": "Confianca media",
                    "value": _format_metric_pct(totals["avg_confidence_pct"]),
                },
                {"label": "Ultima simulacao", "value": summary["last_created_at"] or "-"},
            ],
        ),
        unsafe_allow_html=True,
    )

    if not played_analysis:
        st.info(
            "Ainda nao ha jogos com placar real e simulacao correspondente para avaliar o modelo.",
        )


def _render_prediction_source_selector(db_path: Path) -> str:
    st.sidebar.subheader("Modelo")
    outcome_available = _outcome_predictions_available_for_path(str(db_path))
    options = [PREDICTION_SOURCE_SIMULATION]
    if outcome_available:
        options.append(PREDICTION_SOURCE_OUTCOME_MODEL)

    default_index = (
        options.index(PREDICTION_SOURCE_OUTCOME_MODEL)
        if PREDICTION_SOURCE_OUTCOME_MODEL in options
        else 0
    )
    prediction_source = st.sidebar.radio(
        "Fonte das probabilidades",
        options,
        index=default_index,
        format_func=lambda option: PREDICTION_SOURCE_LABELS[option],
        help=(
            "Simulacao usa as frequencias gravadas em simulated_results. Modelo V/E/D usa "
            "as probabilidades diretas da tabela outcome_predictions quando disponivel."
        ),
    )
    if not outcome_available:
        st.sidebar.caption(
            "A tabela outcome_predictions nao foi encontrada. Rode `uv run predict-outcomes` "
            "para habilitar o modelo V/E/D no dashboard.",
        )
    return str(prediction_source)


def _render_accuracy_sections(played_analysis: list[dict[str, Any]]) -> None:
    group_rows = _group_round_accuracy_rows(played_analysis)
    phase_rows = _phase_accuracy_rows(played_analysis)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Fase de grupos")
        _render_accuracy_chart(group_rows, label_column="rodada")
        _render_accuracy_table(group_rows, label_column="rodada")

    with right_col:
        st.subheader("Fases da Copa")
        _render_accuracy_chart(phase_rows, label_column="fase")
        _render_accuracy_table(phase_rows, label_column="fase")


def _render_accuracy_chart(rows: list[dict[str, Any]], *, label_column: str) -> None:
    chart_df = (
        pl.DataFrame(rows)
        .select(
            pl.col(label_column),
            pl.col("resultado_pct").alias("Resultado"),
            pl.col("placar_pct").alias("Placar exato"),
        )
        .unpivot(index=label_column, variable_name="metrica", value_name="taxa")
        .to_pandas()
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{label_column}:N", title=None, axis=alt.Axis(labelColor="#344054")),
            y=alt.Y("taxa:Q", title="Taxa (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "metrica:N",
                title=None,
                scale=alt.Scale(range=["#0f6b5f", "#d65f4c"]),
            ),
            xOffset="metrica:N",
            tooltip=[
                alt.Tooltip(f"{label_column}:N", title="Corte"),
                alt.Tooltip("metrica:N", title="Metrica"),
                alt.Tooltip("taxa:Q", title="Taxa", format=".1f"),
            ],
        )
        .properties(height=300)
        .configure(background="#ffffff")
        .configure_view(stroke=None)
        .configure_axis(gridColor="#e5e7eb", labelColor="#344054", titleColor="#475467")
        .configure_legend(labelColor="#344054", titleColor="#344054")
    )
    st.altair_chart(chart, use_container_width=True)


def _render_groups_section(
    predicted_groups: dict[str, list[dict[str, Any]]],
    actual_groups: dict[str, list[dict[str, Any]]],
) -> None:
    st.subheader("Classificacao dos grupos")
    st.caption(
        "A previsao usa medias das simulacoes para a tabela final. A classificacao real usa "
        "os placares ja preenchidos no calendario do projeto.",
    )
    mode = st.segmented_control(
        "Tabela exibida",
        ["Previsao do modelo", "Real atual"],
        default="Previsao do modelo",
    )
    source = predicted_groups if mode == "Previsao do modelo" else actual_groups
    st.markdown(
        _group_cards_html(source, predicted=mode == "Previsao do modelo"),
        unsafe_allow_html=True,
    )

    st.subheader("Terceiros colocados")
    third_rows = _third_place_rows(source, predicted=mode == "Previsao do modelo")
    column_config = {}
    if mode == "Previsao do modelo":
        column_config["Chance 3o"] = st.column_config.ProgressColumn(
            "Chance 3o",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        )
    st.dataframe(
        third_rows,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
    )


def _render_global_projection_section(
    global_projection: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    st.subheader("Prognostico global do torneio")
    st.caption(
        "Distribuicao agregada das simulacoes: posicao provavel no grupo, probabilidade de "
        "alcancar cada fase e chance de titulo.",
    )
    top_team = global_projection[0] if global_projection else None
    st.markdown(
        _metric_cards_html(
            [
                {"label": "Selecoes avaliadas", "value": len(global_projection)},
                {
                    "label": "Simulacoes",
                    "value": f"{summary['simulation_count']:,}".replace(",", "."),
                },
                {
                    "label": "Maior chance de titulo",
                    "value": top_team["team_name"] if top_team else "-",
                },
                {
                    "label": "Probabilidade lider",
                    "value": _format_metric_pct(top_team["champion_pct"] if top_team else 0),
                },
            ],
            class_name="metric-grid metric-grid-four",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(_global_projection_table_html(global_projection), unsafe_allow_html=True)

    contenders = global_projection[:12]
    chart_df = pl.DataFrame(contenders).select(
        pl.col("team_name"),
        pl.col("champion_pct").alias("Titulo"),
        pl.col("final_pct").alias("Final"),
        pl.col("semifinal_pct").alias("Semifinal"),
    )
    st.subheader("Principais candidatos")
    st.altair_chart(_stage_probability_chart(chart_df), use_container_width=True)


def _render_bracket_section(bracket_projection: list[dict[str, Any]]) -> None:
    st.subheader("Chaveamento mais provavel do mata-mata")
    st.caption(
        "Cada card mostra o confronto mais frequente nas simulacoes para aquele jogo e o "
        "vencedor mais provavel dentro desse confronto.",
    )
    st.markdown(_bracket_html(bracket_projection), unsafe_allow_html=True)


def _stage_probability_chart(frame: pl.DataFrame) -> alt.Chart:
    chart_df = frame.unpivot(
        index="team_name",
        variable_name="fase",
        value_name="probabilidade",
    ).to_pandas()
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("team_name:N", title=None, sort="-x", axis=alt.Axis(labelColor="#344054")),
            x=alt.X("probabilidade:Q", title="Probabilidade (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "fase:N",
                title=None,
                scale=alt.Scale(range=["#d65f4c", "#246baf", "#0f6b5f"]),
            ),
            tooltip=[
                alt.Tooltip("team_name:N", title="Selecao"),
                alt.Tooltip("fase:N", title="Fase"),
                alt.Tooltip("probabilidade:Q", title="Probabilidade", format=".1f"),
            ],
        )
        .properties(height=380)
        .configure(background="#ffffff")
        .configure_view(stroke=None)
        .configure_axis(gridColor="#e5e7eb", labelColor="#344054", titleColor="#475467")
        .configure_legend(labelColor="#344054", titleColor="#344054")
    )


def _render_accuracy_table(rows: list[dict[str, Any]], *, label_column: str) -> None:
    display_rows = [
        {
            label_column.capitalize(): row[label_column],
            "Jogos": row["jogos"],
            "Acertos resultado": row["resultado_acertos"],
            "Taxa resultado": row["resultado_pct"],
            "Placares exatos": row["placar_acertos"],
            "Taxa placar": row["placar_pct"],
        }
        for row in rows
    ]
    st.dataframe(
        display_rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Taxa resultado": st.column_config.ProgressColumn(
                "Taxa resultado",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Taxa placar": st.column_config.ProgressColumn(
                "Taxa placar",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )


def _render_round_probability_section(
    *,
    selected_round: RoundOption,
    rows: list[dict[str, Any]],
    fixtures: tuple[WorldCupFixture, ...],
) -> None:
    if selected_round.key == "overall":
        st.info("Selecione uma rodada no filtro lateral para ver as probabilidades jogo a jogo.")
        return

    st.markdown(
        _metric_cards_html(
            [
                {"label": "Rodada selecionada", "value": selected_round.label},
                {"label": "Jogos na rodada", "value": len(selected_round.match_numbers)},
                {"label": "Confrontos exibidos", "value": len(rows)},
            ],
            class_name="metric-grid metric-grid-three",
        ),
        unsafe_allow_html=True,
    )

    if not rows:
        st.warning("Nao ha resultados de simulacao para a rodada selecionada.")
        return

    display_rows = _localize_rows(rows, selected_round, fixtures)
    st.markdown(
        _probability_table_html(
            display_rows,
            first_column_header=selected_round.first_column_header,
            include_occurrence=selected_round.dynamic_matchups,
        ),
        unsafe_allow_html=True,
    )


def _render_played_matches_section(played_analysis: list[dict[str, Any]]) -> None:
    if not played_analysis:
        st.info("Nenhum jogo real foi avaliado ate o momento.")
        return

    st.subheader("Comparacao jogo a jogo")
    st.dataframe(
        _played_matches_table_rows(played_analysis),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Confianca": st.column_config.ProgressColumn(
                "Confianca",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


def _accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    played_count = len(rows)
    outcome_hits = sum(1 for row in rows if row["outcome_hit"])
    score_hits = sum(1 for row in rows if row["score_hit"])
    avg_confidence = (
        sum(float(row["confidence_pct"] or 0) for row in rows) / played_count
        if played_count
        else 0.0
    )
    return {
        "played_count": played_count,
        "outcome_hits": outcome_hits,
        "score_hits": score_hits,
        "outcome_accuracy_pct": _safe_pct(outcome_hits, played_count),
        "score_accuracy_pct": _safe_pct(score_hits, played_count),
        "avg_confidence_pct": avg_confidence,
    }


def _group_round_accuracy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _accuracy_bucket_row(
            label=f"Rodada {round_number}",
            rows=[row for row in rows if row.get("group_round") == round_number],
            key_name="rodada",
            order=round_number,
        )
        for round_number in (1, 2, 3)
    ]


def _phase_accuracy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_rows = []
    for round_name, phase_order in sorted(ROUND_DISPLAY_ORDER.items(), key=lambda item: item[1]):
        phase_rows.append(
            _accuracy_bucket_row(
                label=ROUND_LABELS.get(round_name, round_name),
                rows=[row for row in rows if row.get("round_name") == round_name],
                key_name="fase",
                order=phase_order,
            ),
        )
    return phase_rows


def _accuracy_bucket_row(
    *,
    label: str,
    rows: list[dict[str, Any]],
    key_name: str,
    order: int,
) -> dict[str, Any]:
    summary = _accuracy_summary(rows)
    return {
        key_name: label,
        "ordem": order,
        "jogos": summary["played_count"],
        "resultado_acertos": summary["outcome_hits"],
        "resultado_pct": round(float(summary["outcome_accuracy_pct"]), 1),
        "placar_acertos": summary["score_hits"],
        "placar_pct": round(float(summary["score_accuracy_pct"]), 1),
    }


def _played_matches_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Jogo": int(row["match_number"]),
            "Fase": row["phase_label"],
            "Rodada": row["group_round_label"],
            "Time 1": row["home_team_name"],
            "Time 2": row["away_team_name"],
            "Placar modelo": _format_score(row),
            "Resultado modelo": _outcome_display(row, str(row["predicted_outcome"])),
            "Fonte prob.": _probability_source_label(row),
            "Placar real": _format_actual_score(row),
            "Resultado real": _outcome_display(row, str(row["actual_outcome"])),
            "Acertou resultado": "Sim" if row["outcome_hit"] else "Nao",
            "Acertou placar": "Sim" if row["score_hit"] else "Nao",
            "Confianca": round(float(row["confidence_pct"] or 0), 1),
        }
        for row in rows
    ]


def _metric_cards_html(
    cards: list[dict[str, Any]],
    *,
    class_name: str = "metric-grid",
) -> str:
    rendered_cards = []
    for card in cards:
        label = html.escape(str(card["label"]))
        value = html.escape(str(card["value"]))
        delta = card.get("delta")
        rendered_delta = (
            f'<div class="metric-delta">{html.escape(str(delta))}</div>'
            if delta is not None
            else ""
        )
        rendered_cards.append(
            (
                '<article class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f"{rendered_delta}"
                "</article>"
            ),
        )
    return f'<div class="{html.escape(class_name)}">{"".join(rendered_cards)}</div>'


def _filter_played_analysis_by_round(
    rows: list[dict[str, Any]],
    selected_round: RoundOption,
) -> list[dict[str, Any]]:
    if selected_round.key == "overall":
        return rows
    selected_matches = set(selected_round.match_numbers)
    return [row for row in rows if int(row["match_number"]) in selected_matches]


def _actual_group_standings(
    fixtures: tuple[WorldCupFixture, ...],
) -> dict[str, list[dict[str, Any]]]:
    standings: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for fixture in fixtures:
        if fixture.round_name != "group_stage" or fixture.group_name is None:
            continue
        group_letter = fixture.group_name.rsplit(" ", maxsplit=1)[-1]
        for team_id in (fixture.home_slot, fixture.away_slot):
            standings[group_letter].setdefault(
                team_id,
                {
                    "group_letter": group_letter,
                    "team_id": team_id,
                    "team_name": _team_display_name(team_id, team_id),
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goal_diff": 0,
                    "points": 0,
                },
            )

        if fixture.played_home_goals is None or fixture.played_away_goals is None:
            continue

        home = standings[group_letter][fixture.home_slot]
        away = standings[group_letter][fixture.away_slot]
        home_goals = fixture.played_home_goals
        away_goals = fixture.played_away_goals
        _apply_actual_result(home, home_goals, away_goals)
        _apply_actual_result(away, away_goals, home_goals)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for group_letter, rows_by_team in standings.items():
        rows = list(rows_by_team.values())
        rows.sort(
            key=lambda row: (
                -int(row["points"]),
                -int(row["goal_diff"]),
                -int(row["goals_for"]),
                str(row["team_name"]),
            ),
        )
        for index, row in enumerate(rows, start=1):
            row["position"] = index
        grouped[group_letter] = rows
    return dict(sorted(grouped.items()))


def _apply_actual_result(row: dict[str, Any], goals_for: int, goals_against: int) -> None:
    row["played"] += 1
    row["goals_for"] += goals_for
    row["goals_against"] += goals_against
    row["goal_diff"] = row["goals_for"] - row["goals_against"]
    if goals_for > goals_against:
        row["wins"] += 1
        row["points"] += 3
    elif goals_for == goals_against:
        row["draws"] += 1
        row["points"] += 1
    else:
        row["losses"] += 1


def _group_rows_by_letter(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_letter = str(row["group_name"]).rsplit(" ", maxsplit=1)[-1]
        display_row = dict(row)
        display_row["group_letter"] = group_letter
        display_row["team_name"] = _team_display_name(str(row["team_id"]), str(row["team_name"]))
        display_row["position"] = len(grouped[group_letter]) + 1
        grouped[group_letter].append(display_row)
    return dict(sorted(grouped.items()))


def _third_place_rows(
    groups: dict[str, list[dict[str, Any]]],
    *,
    predicted: bool,
) -> list[dict[str, Any]]:
    third_rows = []
    for group_letter, rows in groups.items():
        if len(rows) < 3:
            continue
        row = rows[2]
        base = {
            "Grupo": group_letter,
            "Selecao": row["team_name"],
            "J": row["played"],
            "GP": row["goals_for"],
            "GC": row["goals_against"],
            "SG": row["goal_diff"],
            "PTS": row["points"],
        }
        if predicted:
            base["Chance 3o"] = float(row.get("third_pct") or 0)
        third_rows.append(base)
    third_rows.sort(key=lambda row: (-float(row["PTS"]), -float(row["SG"]), -float(row["GP"])))
    return third_rows


def _group_cards_html(groups: dict[str, list[dict[str, Any]]], *, predicted: bool) -> str:
    cards = []
    for group_letter, rows in groups.items():
        body_rows = []
        for row in rows:
            position = int(row["position"])
            team_name = html.escape(str(row["team_name"]))
            status_class = "qualified" if position <= 2 else "third" if position == 3 else "out"
            body_rows.append(
                (
                    "<tr>"
                    f'<td><span class="rank {status_class}">{position}</span></td>'
                    f'<td class="team-name">{team_name}</td>'
                    f"<td>{_format_table_number(row['played'])}</td>"
                    f"<td>{_format_table_number(row['wins'])}</td>"
                    f"<td>{_format_table_number(row['draws'])}</td>"
                    f"<td>{_format_table_number(row['losses'])}</td>"
                    f"<td>{_format_table_number(row['goals_for'])}</td>"
                    f"<td>{_format_table_number(row['goals_against'])}</td>"
                    f"<td>{_format_signed_number(row['goal_diff'])}</td>"
                    f'<td class="points">{_format_table_number(row["points"])}</td>'
                    f"{'<td>' + _format_pct(row.get('top2_pct', 0)) + '</td>' if predicted else ''}"
                    "</tr>"
                ),
            )
        top2_header = "<th>Top 2</th>" if predicted else ""
        cards.append(
            (
                '<article class="group-card">'
                '<div class="group-card-title">'
                f"<span>{html.escape(group_letter)}</span>"
                f"<strong>Grupo {html.escape(group_letter)}</strong>"
                "</div>"
                '<div class="group-table-wrap">'
                f'<table class="group-table {"predicted" if predicted else "actual"}">'
                "<thead><tr>"
                "<th>#</th><th>Selecao</th><th>J</th><th>V</th><th>E</th><th>D</th>"
                f"<th>GP</th><th>GC</th><th>SG</th><th>PTS</th>{top2_header}"
                "</tr></thead>"
                f"<tbody>{''.join(body_rows)}</tbody>"
                "</table>"
                "</div>"
                "</article>"
            ),
        )
    legend = (
        '<div class="group-legend"><span class="dot qualified"></span>Avanca direto '
        '<span class="dot third"></span>3o colocado '
        '<span class="dot out"></span>Eliminado/risco</div>'
    )
    return f'<div class="group-grid">{"".join(cards)}</div>{legend}'


def _global_projection_table_html(rows: list[dict[str, Any]]) -> str:
    columns = (
        ("group_1_pct", "1o grupo"),
        ("group_2_pct", "2o grupo"),
        ("group_3_pct", "3o grupo"),
        ("round_of_32_pct", "32 avos"),
        ("round_of_16_pct", "Oitavas"),
        ("quarterfinal_pct", "Quartas"),
        ("semifinal_pct", "Semis"),
        ("final_pct", "Final"),
        ("runner_up_pct", "Vice"),
        ("champion_pct", "Titulo"),
    )
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = float(row.get(key) or 0)
            cells.append(
                f'<td style="{_heat_cell_style(value)}">{_format_pct(value)}</td>',
            )
        body.append(
            (
                "<tr>"
                '<td class="team-sticky">'
                f'<span class="team-code">{html.escape(str(row["team_id"]))}</span>'
                f"{html.escape(str(row['team_name']))}"
                "</td>"
                f"<td>{html.escape(str(row['group_letter']))}</td>"
                f"<td>{float(row['avg_group_position']):.2f}</td>"
                f"{''.join(cells)}"
                "</tr>"
            ),
        )
    return (
        '<div class="projection-table-wrap">'
        '<table class="projection-table">'
        "<thead><tr>"
        '<th class="team-sticky">Selecao</th>'
        "<th>Grupo</th><th>Pos. media</th>"
        f"{header}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
        "</div>"
    )


def _bracket_html(rows: list[dict[str, Any]]) -> str:
    rows_by_match = {int(row["match_number"]): row for row in rows}
    columns = []
    for round_name, round_label, match_numbers in BRACKET_ROUNDS:
        cards = []
        for match_number in match_numbers:
            row = rows_by_match.get(match_number)
            if row is None:
                continue
            cards.append(_bracket_match_card(row, highlight=round_name == "final"))
        columns.append(
            (
                '<section class="bracket-column">'
                f"<h4>{html.escape(round_label)}</h4>"
                f'<div class="bracket-stack">{"".join(cards)}</div>'
                "</section>"
            ),
        )
    return f'<div class="bracket-board">{"".join(columns)}</div>'


def _bracket_match_card(row: dict[str, Any], *, highlight: bool) -> str:
    card_class = "bracket-card final-card" if highlight else "bracket-card"
    score = _format_score(row)
    winner_name = html.escape(str(row["winner_name"]))
    return (
        f'<article class="{card_class}">'
        '<div class="match-meta">'
        f"<span>Jogo {int(row['match_number'])}</span>"
        f"<span>{_format_pct(row['occurrence_pct'])} ocorr.</span>"
        "</div>"
        '<div class="match-teams">'
        f"<span>{html.escape(str(row['home_team_name']))}</span>"
        f"<strong>{html.escape(score)}</strong>"
        f"<span>{html.escape(str(row['away_team_name']))}</span>"
        "</div>"
        '<div class="winner-line">'
        f"Avanca: <strong>{winner_name}</strong> ({_format_pct(row['winner_pct'])})"
        "</div>"
        "</article>"
    )


def _heat_cell_style(value: float) -> str:
    alpha = 0.08 + min(max(value, 0), 100) / 100 * 0.56
    return f"background: rgba(15, 107, 95, {alpha:.3f}); color: #10231f;"


def _query_dicts(result: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [column[0] for column in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _format_table_number(value: Any) -> str:
    numeric = float(value or 0)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.1f}"


def _format_signed_number(value: Any) -> str:
    numeric = float(value or 0)
    if numeric.is_integer():
        return f"{int(numeric):+d}"
    return f"{numeric:+.1f}"


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


@st.cache_data(ttl=300)
def _outcome_predictions_available_for_path(db_path: str) -> bool:
    with duckdb.connect(db_path, read_only=True) as con:
        return _outcome_predictions_available(con)


def _build_round_options(fixtures: tuple[WorldCupFixture, ...]) -> list[RoundOption]:
    group_rounds = _group_round_match_numbers(fixtures)
    options = [
        RoundOption(
            key="overall",
            label="Geral",
            match_numbers=tuple(fixture.match_number for fixture in fixtures),
            first_column_header="Jogo",
        ),
    ]
    options.extend(
        [
            RoundOption(
                key=f"group_stage_{round_number}",
                label=f"Fase de grupos - Rodada {round_number}",
                match_numbers=tuple(match_numbers),
                first_column_header="Grupo",
            )
            for round_number, match_numbers in sorted(group_rounds.items())
        ],
    )

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


def _group_round_lookup(fixtures: tuple[WorldCupFixture, ...]) -> dict[int, int]:
    lookup = {}
    for round_number, match_numbers in _group_round_match_numbers(fixtures).items():
        for match_number in match_numbers:
            lookup[match_number] = round_number
    return lookup


def _localize_rows(
    rows: list[dict[str, Any]],
    selected_round: RoundOption,
    fixtures: tuple[WorldCupFixture, ...],
) -> list[dict[str, Any]]:
    fixtures_by_match = {fixture.match_number: fixture for fixture in fixtures}
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
    actual_home_goals = row.get("actual_home_goals")
    actual_away_goals = row.get("actual_away_goals")
    if actual_home_goals is None or actual_away_goals is None:
        return None

    if _has_outcome_probabilities(row):
        predicted_outcome = _most_likely_outcome(row)
    else:
        predicted_home_goals = row.get("predicted_home_goals")
        predicted_away_goals = row.get("predicted_away_goals")
        if predicted_home_goals is None or predicted_away_goals is None:
            return None
        predicted_outcome = _score_outcome(int(predicted_home_goals), int(predicted_away_goals))
    actual_outcome = _score_outcome(int(actual_home_goals), int(actual_away_goals))
    return "Acerto" if predicted_outcome == actual_outcome else "Erro"


def _has_outcome_probabilities(row: dict[str, Any]) -> bool:
    return all(key in row for key in ("home_win_pct", "draw_pct", "away_win_pct"))


def _most_likely_outcome(row: dict[str, Any]) -> str:
    probabilities = {
        "home": float(row.get("home_win_pct") or 0),
        "draw": float(row.get("draw_pct") or 0),
        "away": float(row.get("away_win_pct") or 0),
    }
    modal_score_outcome = None
    if row.get("predicted_home_goals") is not None and row.get("predicted_away_goals") is not None:
        modal_score_outcome = _score_outcome(
            int(row["predicted_home_goals"]),
            int(row["predicted_away_goals"]),
        )
    return max(
        probabilities,
        key=lambda outcome: (
            probabilities[outcome],
            outcome == modal_score_outcome,
            outcome == "home",
        ),
    )


def _outcome_probability(row: dict[str, Any], outcome: str) -> float:
    probability_columns = {
        "home": "home_win_pct",
        "draw": "draw_pct",
        "away": "away_win_pct",
    }
    return float(row.get(probability_columns[outcome]) or 0)


def _score_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def _is_exact_score_hit(row: dict[str, Any]) -> bool:
    return (
        row.get("predicted_home_goals") is not None
        and row.get("predicted_away_goals") is not None
        and row.get("actual_home_goals") is not None
        and row.get("actual_away_goals") is not None
        and int(row["predicted_home_goals"]) == int(row["actual_home_goals"])
        and int(row["predicted_away_goals"]) == int(row["actual_away_goals"])
    )


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _format_metric_pct(value: Any) -> str:
    return f"{float(value or 0):.1f}%"


def _outcome_display(row: dict[str, Any], outcome: str) -> str:
    if outcome == "home":
        return str(row["home_team_name"])
    if outcome == "away":
        return str(row["away_team_name"])
    return "Empate"


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
            _header_cell("Fonte"),
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
                f'<td class="source-cell">{html.escape(_probability_source_label(row))}</td>',
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


def _probability_source_label(row: dict[str, Any]) -> str:
    if row.get("probability_source") == "outcome_model":
        return "V/E/D"
    return "Sim."


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
        :root {
            --dashboard-bg: #f6f7f9;
            --panel-border: #d8dde6;
            --ink: #18202f;
            --muted: #667085;
            --accent: #0f6b5f;
            --accent-soft: #e6f4f1;
        }
        .stApp {
            background: var(--dashboard-bg);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        .dashboard-hero {
            background:
                linear-gradient(135deg, rgba(15, 107, 95, 0.94), rgba(24, 32, 47, 0.96)),
                radial-gradient(circle at 78% 20%, rgba(255, 255, 255, 0.22), transparent 32%);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            color: #ffffff;
            margin-bottom: 1.25rem;
            padding: 1.35rem 1.5rem;
        }
        .dashboard-hero .eyebrow {
            color: #c8f1e8;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            margin: 0 0 0.4rem;
            text-transform: uppercase;
        }
        .dashboard-hero h1 {
            color: #ffffff;
            font-size: 2rem;
            line-height: 1.1;
            margin: 0;
            letter-spacing: 0;
        }
        .dashboard-hero p:last-child {
            color: rgba(255, 255, 255, 0.84);
            margin: 0.65rem 0 0;
            max-width: 760px;
        }
        .metric-grid {
            display: grid;
            gap: 0.85rem;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            margin: 1rem 0 1.25rem;
        }
        .metric-grid-four {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .metric-grid-three {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .sidebar-metric-stack {
            display: grid;
            gap: 0.8rem;
            margin-top: 1rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-width: 0;
            padding: 0.85rem 1rem;
        }
        .metric-label {
            color: #344054;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.55rem;
            opacity: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .metric-value {
            color: #101828;
            font-size: 1.75rem;
            font-weight: 900;
            line-height: 1.1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .metric-delta {
            color: #067647;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 0.45rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .sidebar-metric-stack .metric-card {
            min-height: 96px;
        }
        .sidebar-metric-stack .metric-value {
            font-size: 1.65rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            min-height: 108px;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        section[data-testid="stSidebar"] div[data-testid="stMetricLabel"],
        section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] * {
            color: #344054 !important;
            font-size: 0.82rem;
            font-weight: 700;
            opacity: 1 !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-weight: 800;
        }
        div[data-testid="stMetricDelta"] {
            color: #067647 !important;
            font-weight: 700;
            opacity: 1 !important;
        }
        div[data-testid="stTabs"] button {
            color: #344054;
            font-weight: 700;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #0f6b5f;
        }
        input, textarea, select {
            color: #101828 !important;
            background: #ffffff !important;
        }
        button:focus-visible,
        [role="tab"]:focus-visible,
        input:focus-visible {
            outline: 3px solid rgba(15, 107, 95, 0.35) !important;
            outline-offset: 2px;
        }
        h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        p, label, span {
            text-wrap: pretty;
        }
        .group-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .group-card {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            overflow: hidden;
        }
        .group-table-wrap {
            overflow-x: auto;
            width: 100%;
        }
        .group-card-title {
            align-items: center;
            background: #18202f;
            color: #ffffff;
            display: flex;
            gap: 0.55rem;
            padding: 0.7rem 0.85rem;
        }
        .group-card-title span {
            align-items: center;
            background: #ffffff;
            border-radius: 999px;
            color: #18202f;
            display: inline-flex;
            font-weight: 800;
            height: 1.45rem;
            justify-content: center;
            width: 1.45rem;
        }
        .group-card table,
        .projection-table {
            border-collapse: collapse;
            width: 100%;
        }
        .group-card .group-table.predicted {
            min-width: 520px;
        }
        .group-card .group-table.actual {
            min-width: 470px;
        }
        .group-card th,
        .group-card td,
        .projection-table th,
        .projection-table td {
            border-bottom: 1px solid #edf0f5;
            color: #344054;
            font-size: 0.78rem;
            padding: 0.48rem 0.45rem;
            text-align: right;
            white-space: nowrap;
        }
        .group-card th,
        .projection-table th {
            background: #f8fafc;
            color: #667085;
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .group-card .team-name,
        .projection-table .team-sticky {
            font-weight: 700;
            max-width: 180px;
            min-width: 140px;
            overflow: hidden;
            text-align: left;
            text-overflow: ellipsis;
        }
        .group-card .rank {
            border-radius: 999px;
            display: inline-block;
            font-weight: 800;
            min-width: 1.45rem;
            padding: 0.1rem 0.25rem;
            text-align: center;
        }
        .group-card .qualified {
            background: #e6f4f1;
            color: #0f6b5f;
        }
        .group-card .third {
            background: #fff4df;
            color: #9a5b00;
        }
        .group-card .out {
            background: #fee4e2;
            color: #b42318;
        }
        .group-card .points {
            color: var(--ink);
            font-weight: 900;
        }
        .group-legend {
            color: #475467;
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: 0.85rem 0 1.25rem;
        }
        .dot {
            border-radius: 999px;
            display: inline-block;
            height: 0.62rem;
            margin-right: 0.35rem;
            width: 0.62rem;
        }
        .dot.qualified {
            background: #0f6b5f;
        }
        .dot.third {
            background: #d68b00;
        }
        .dot.out {
            background: #d65f4c;
        }
        .projection-table-wrap,
        .bracket-board {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            margin-top: 1rem;
            overflow-x: auto;
        }
        .projection-table {
            min-width: 1180px;
        }
        .projection-table .team-sticky {
            background: inherit;
            left: 0;
            position: sticky;
            z-index: 1;
        }
        .projection-table thead .team-sticky {
            background: #f8fafc;
            z-index: 2;
        }
        .team-code {
            color: #667085;
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 800;
            margin-right: 0.45rem;
            min-width: 2.2rem;
        }
        .bracket-board {
            align-items: stretch;
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(6, minmax(210px, 1fr));
            padding: 1rem;
        }
        .bracket-column h4 {
            color: #18202f;
            font-size: 0.9rem;
            margin: 0 0 0.75rem;
            text-align: center;
        }
        .bracket-stack {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
        }
        .bracket-card {
            background: #f8fafc;
            border: 1px solid #d8dde6;
            border-radius: 8px;
            padding: 0.65rem;
        }
        .final-card {
            background: #fff8e8;
            border-color: #d68b00;
            box-shadow: 0 0 0 2px rgba(214, 139, 0, 0.14);
        }
        .match-meta,
        .winner-line {
            color: #667085;
            font-size: 0.72rem;
        }
        .match-meta {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
        }
        .match-teams {
            display: grid;
            gap: 0.25rem;
            margin: 0.45rem 0;
        }
        .match-teams span,
        .winner-line strong {
            color: #18202f;
            font-weight: 800;
            overflow-wrap: anywhere;
        }
        .match-teams strong {
            color: #0f6b5f;
            font-variant-numeric: tabular-nums;
        }
        @media (max-width: 1100px) {
            .metric-grid,
            .metric-grid-four {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .metric-grid-three {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .group-grid {
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr));
            }
            .bracket-board {
                grid-template-columns: repeat(3, minmax(210px, 1fr));
            }
        }
        @media (max-width: 760px) {
            .metric-grid,
            .metric-grid-four,
            .metric-grid-three {
                grid-template-columns: 1fr;
            }
            .group-grid {
                grid-template-columns: 1fr;
            }
            .bracket-board {
                grid-template-columns: repeat(2, minmax(210px, 1fr));
            }
        }
        .prob-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin-top: 1rem;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: #ffffff;
        }
        .prob-table {
            width: 100%;
            min-width: 1180px;
            border-collapse: collapse;
            font-size: 15px;
            line-height: 1.25;
        }
        .prob-table th {
            background: #18202f;
            color: #ffffff;
            border-bottom: 1px solid #2f3949;
            padding: 10px 12px;
            text-align: center;
            font-weight: 700;
            letter-spacing: 0;
        }
        .prob-table th span {
            display: inline-block;
        }
        .prob-table td {
            border-bottom: 1px solid #edf0f5;
            padding: 10px 12px;
            color: #344054;
            background: #ffffff;
        }
        .prob-table tr.even-row td:not(.bucket-cell) {
            background: #f8fafc;
        }
        .prob-table .bucket-cell {
            width: 72px;
            min-width: 72px;
            text-align: center;
            vertical-align: middle;
            background: var(--accent);
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
            color: var(--ink);
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
        .prob-table .source-cell {
            color: #475467;
            font-size: 0.82rem;
            font-weight: 800;
            min-width: 72px;
            text-align: center;
        }
        .prob-table .result-hit {
            color: #067647;
        }
        .prob-table .result-miss {
            color: #b42318;
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
