"""Streamlit view functions for dashboard sections and tabs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import polars as pl
import streamlit as st

try:
    from ..live_results import LiveFixtureSnapshot, load_world_cup_fixture_snapshot
    from ..world_cup_2026_schedule import WorldCupFixture
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`.
    from live_results import LiveFixtureSnapshot, load_world_cup_fixture_snapshot
    from world_cup_2026_schedule import WorldCupFixture

from .analysis import (
    _accuracy_summary,
    _group_round_accuracy_rows,
    _localize_rows,
    _phase_accuracy_rows,
    _third_place_rows,
)
from .components import (
    _bracket_html,
    _global_projection_table_html,
    _group_cards_html,
    _home_projection_showcase_html,
    _match_prediction_cards_html,
    _metric_cards_html,
    _played_matches_table_rows,
    _probability_table_html,
)
from .constants import (
    PREDICTION_SOURCE_LABELS,
    PREDICTION_SOURCE_OUTCOME_MODEL,
    PREDICTION_SOURCE_SIMULATION,
    RoundOption,
)
from .data import _outcome_predictions_available_for_path
from .formatting import _format_metric_pct


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


def _render_home_projection_showcase(
    global_projection: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if not global_projection:
        st.info("Ainda nao ha prognostico global para montar o destaque do modelo.")
        return

    st.markdown(
        _home_projection_showcase_html(global_projection, summary),
        unsafe_allow_html=True,
    )


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
                scale=alt.Scale(range=["#08786c", "#c85547"]),
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
        .configure_axis(gridColor="#e8eef4", labelColor="#344054", titleColor="#667085")
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
                scale=alt.Scale(range=["#c85547", "#2563a8", "#08786c"]),
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
        .configure_axis(gridColor="#e8eef4", labelColor="#344054", titleColor="#667085")
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


def _render_match_prediction_section(
    *,
    selected_round: RoundOption,
    rows: list[dict[str, Any]],
    fixtures: tuple[WorldCupFixture, ...],
) -> None:
    if selected_round.key == "overall":
        st.info("Selecione uma rodada no filtro lateral para ver a previsao visual por jogo.")
        return

    if not rows:
        st.warning("Nao ha resultados de simulacao para a rodada selecionada.")
        return

    display_rows = _localize_rows(rows, selected_round, fixtures)
    st.subheader("Como o modelo via cada jogo")
    st.caption(
        "Cada card mostra a probabilidade de vitoria do mandante, empate e vitoria do "
        "visitante. Quando ha placar real, o card compara a previsao mais provavel com o "
        "resultado observado."
    )
    st.markdown(
        _match_prediction_cards_html(
            display_rows,
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
