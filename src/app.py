"""Streamlit dashboard entrypoint for World Cup simulation probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

try:
    from .dashboard.analysis import (
        _accuracy_summary,
        _actual_group_standings,
        _add_played_fixture_result,
        _build_round_options,
        _filter_played_analysis_by_round,
        _group_round_accuracy_rows,
        _phase_accuracy_rows,
        _prediction_result,
        _third_place_rows,
    )
    from .dashboard.components import (
        _home_projection_showcase_html,
        _match_prediction_cards_html,
        _metric_cards_html,
    )
    from .dashboard.constants import (
        PREDICTION_SOURCE_OUTCOME_MODEL,
        PREDICTION_SOURCE_SIMULATION,
        RoundOption,
    )
    from .dashboard.data import (
        _load_global_projection,
        _load_knockout_bracket,
        _load_played_match_analysis,
        _load_predicted_group_standings,
        _load_round_probabilities,
        _load_summary,
    )
    from .dashboard.formatting import _format_actual_score, _format_score, _friendly_db_error
    from .dashboard.styles import _inject_styles
    from .dashboard.views import (
        _load_dashboard_fixture_snapshot,
        _render_accuracy_overview,
        _render_accuracy_sections,
        _render_bracket_section,
        _render_fixture_refresh_status,
        _render_global_projection_section,
        _render_groups_section,
        _render_home_projection_showcase,
        _render_match_prediction_section,
        _render_played_matches_section,
        _render_prediction_source_selector,
        _render_round_probability_section,
    )
    from .settings import DB_PATH
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`.
    from dashboard.analysis import (
        _accuracy_summary,
        _actual_group_standings,
        _add_played_fixture_result,
        _build_round_options,
        _filter_played_analysis_by_round,
        _group_round_accuracy_rows,
        _phase_accuracy_rows,
        _prediction_result,
        _third_place_rows,
    )
    from dashboard.components import (
        _home_projection_showcase_html,
        _match_prediction_cards_html,
        _metric_cards_html,
    )
    from dashboard.constants import (
        PREDICTION_SOURCE_OUTCOME_MODEL,
        PREDICTION_SOURCE_SIMULATION,
        RoundOption,
    )
    from dashboard.data import (
        _load_global_projection,
        _load_knockout_bracket,
        _load_played_match_analysis,
        _load_predicted_group_standings,
        _load_round_probabilities,
        _load_summary,
    )
    from dashboard.formatting import _format_actual_score, _format_score, _friendly_db_error
    from dashboard.styles import _inject_styles
    from dashboard.views import (
        _load_dashboard_fixture_snapshot,
        _render_accuracy_overview,
        _render_accuracy_sections,
        _render_bracket_section,
        _render_fixture_refresh_status,
        _render_global_projection_section,
        _render_groups_section,
        _render_home_projection_showcase,
        _render_match_prediction_section,
        _render_played_matches_section,
        _render_prediction_source_selector,
        _render_round_probability_section,
    )
    from settings import DB_PATH

__all__ = [
    "PREDICTION_SOURCE_OUTCOME_MODEL",
    "PREDICTION_SOURCE_SIMULATION",
    "RoundOption",
    "_accuracy_summary",
    "_actual_group_standings",
    "_add_played_fixture_result",
    "_build_round_options",
    "_filter_played_analysis_by_round",
    "_format_actual_score",
    "_format_score",
    "_group_round_accuracy_rows",
    "_home_projection_showcase_html",
    "_load_round_probabilities",
    "_match_prediction_cards_html",
    "_metric_cards_html",
    "_phase_accuracy_rows",
    "_prediction_result",
    "_third_place_rows",
    "cli",
    "main",
]


def cli() -> None:
    """Launch the Streamlit app from the project script."""
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Analise do Modelo - Copa 2026",
        layout="wide",
        initial_sidebar_state="auto",
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

    rows = []
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

    (
        tab_overview,
        tab_groups,
        tab_global,
        tab_bracket,
        tab_match_predictions,
        tab_probabilities,
        tab_matches,
    ) = st.tabs(
        [
            "Visao geral",
            "Grupos",
            "Prognostico",
            "Chaveamento",
            "Previsao por jogo",
            "Probabilidades",
            "Jogos avaliados",
        ],
    )

    with tab_overview:
        _render_home_projection_showcase(global_projection, summary)
        _render_accuracy_sections(scoped_played_analysis)

    with tab_groups:
        _render_groups_section(group_projection, actual_group_standings)

    with tab_global:
        _render_global_projection_section(global_projection, summary)

    with tab_bracket:
        _render_bracket_section(bracket_projection)

    with tab_match_predictions:
        _render_match_prediction_section(
            selected_round=selected_round,
            rows=rows,
            fixtures=fixtures,
        )

    with tab_probabilities:
        _render_round_probability_section(
            selected_round=selected_round,
            rows=rows,
            fixtures=fixtures,
        )

    with tab_matches:
        _render_played_matches_section(scoped_played_analysis)


if __name__ == "__main__":
    main()
