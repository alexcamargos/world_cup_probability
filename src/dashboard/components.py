"""HTML component builders used by the Streamlit dashboard."""

from __future__ import annotations

import html
import textwrap
from typing import Any

from .analysis import _actual_outcome, _most_likely_outcome, _outcome_probability
from .constants import BRACKET_ROUNDS, TEAM_FLAG_EMOJI
from .formatting import (
    _format_actual_score,
    _format_compact_pct,
    _format_metric_pct,
    _format_one_decimal,
    _format_pct,
    _format_score,
    _format_signed_number,
    _format_table_number,
    _probability_source_label,
)


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


def _home_projection_showcase_html(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    if not rows:
        return ""

    leader = rows[0]
    challengers = rows[1:12]
    return f"""
    <section class="home-projection-showcase">
        <div class="favorite-panel">
            <div class="favorite-copy">
                <p>Favorita do modelo</p>
                <div class="favorite-title-line">
                    {_team_flag_badge_html(str(leader["team_id"]))}
                    <strong>{html.escape(str(leader["team_name"]))}</strong>
                    <span>{_format_metric_pct(leader["champion_pct"])}</span>
                </div>
                <em>Vai erguer a taca</em>
                <small>{html.escape(_showcase_summary_line(summary))}</small>
            </div>
        </div>
        <div class="stage-metric-panel">
            {_stage_metric_tile_html("Taca", leader.get("champion_pct"))}
            {_stage_metric_tile_html("FNL", leader.get("final_pct"))}
            {_stage_metric_tile_html("SF", leader.get("semifinal_pct"))}
            {_stage_metric_tile_html("R16", leader.get("round_of_16_pct"))}
        </div>
    </section>
    {_challenger_ranking_html(challengers, start_rank=2)}
    """


def _showcase_summary_line(summary: dict[str, Any]) -> str:
    simulation_count = int(summary.get("simulation_count") or 0)
    simulations = f"{simulation_count:,}".replace(",", ".")
    last_created_at = summary.get("last_created_at") or "-"
    return f"{simulations} simulacoes - modelo de simulacao - ultima execucao {last_created_at}"


def _stage_metric_tile_html(label: str, value: Any) -> str:
    return (
        '<article class="stage-metric-tile">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{_format_metric_pct(value)}</strong>"
        "</article>"
    )


def _challenger_ranking_html(rows: list[dict[str, Any]], *, start_rank: int) -> str:
    if not rows:
        return ""

    rendered_rows = [
        _challenger_row_html(row, rank=start_rank + index) for index, row in enumerate(rows)
    ]
    return f"""
    <section class="challenger-ranking">
        <div class="section-kicker">Candidatas seguintes</div>
        <div class="challenger-list">{"".join(rendered_rows)}</div>
    </section>
    """


def _challenger_row_html(row: dict[str, Any], *, rank: int) -> str:
    team_id = str(row["team_id"])
    stage_line = (
        f"FNL {_format_compact_pct(row.get('final_pct'))} - "
        f"SF {_format_compact_pct(row.get('semifinal_pct'))}"
    )
    return f"""
    <article class="challenger-row">
        <div class="challenger-rank">{rank:02d}</div>
        {_team_flag_badge_html(team_id)}
        <div class="challenger-team">
            <strong>{html.escape(str(row["team_name"]))}</strong>
            <span>{html.escape(stage_line)}</span>
        </div>
        <div class="challenger-title-prob">
            <strong>{_format_one_decimal(row.get("champion_pct"))}</strong>
            <span>Taca</span>
        </div>
    </article>
    """


def _team_flag_badge_html(team_id: str) -> str:
    flag = TEAM_FLAG_EMOJI.get(team_id)
    if flag:
        return f'<span class="flag-badge" aria-label="{html.escape(team_id)}">{flag}</span>'
    return f'<span class="flag-badge flag-fallback">{html.escape(team_id)}</span>'


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


def _outcome_display(row: dict[str, Any], outcome: str) -> str:
    if outcome == "home":
        return str(row["home_team_name"])
    if outcome == "away":
        return str(row["away_team_name"])
    return "Empate"


def _match_prediction_cards_html(
    rows: list[dict[str, Any]],
    *,
    include_occurrence: bool,
) -> str:
    cards = [
        _match_prediction_card_html(row, include_occurrence=include_occurrence) for row in rows
    ]
    rendered_cards = "\n".join(cards)
    return f'<div class="match-prediction-grid">{rendered_cards}</div>'


def _match_prediction_card_html(row: dict[str, Any], *, include_occurrence: bool) -> str:
    predicted_outcome = _most_likely_outcome(row)
    actual_outcome = _actual_outcome(row)
    prediction_result = row.get("prediction_result")
    predicted_label = _outcome_display(row, predicted_outcome)
    actual_label = _outcome_display(row, actual_outcome) if actual_outcome else "-"
    confidence_pct = _outcome_probability(row, predicted_outcome)
    status_class = "pending"
    status_label = "Aguardando real"
    if prediction_result == "Acerto":
        status_class = "hit"
        status_label = "Acerto"
    elif prediction_result == "Erro":
        status_class = "miss"
        status_label = "Erro"

    meta_parts = [f"Jogo {int(row['match_number'])}"]
    if row.get("bucket"):
        meta_parts.append(f"{row['bucket']}")
    if include_occurrence:
        meta_parts.append(f"Ocorr. {_format_pct(row.get('occurrence_pct'))}%")
    meta = " - ".join(meta_parts)

    score_value = _format_actual_score(row)
    score_class = "is-pending" if score_value == "-" else ""
    score_label = "Placar real" if score_value != "-" else "A jogar"
    status_badge = (
        f'<span class="prediction-status {status_class}">{html.escape(status_label)}</span>'
    )

    return textwrap.dedent(
        f"""
    <article class="match-prediction-card">
        <div class="match-prediction-topline">
            <span>{html.escape(meta)}</span>
            <span class="source-pill">{html.escape(_probability_source_label(row))}</span>
        </div>
        <div class="match-scoreboard">
            <div class="match-team match-team-home">
                <span class="team-code">{html.escape(str(row["home_team_id"]))}</span>
                <strong>{html.escape(str(row["home_team_name"]))}</strong>
            </div>
            <div class="score-block {score_class}">
                <strong>{html.escape(score_value if score_value != "-" else "vs")}</strong>
                <span>{html.escape(score_label)}</span>
            </div>
            <div class="match-team match-team-away">
                <span class="team-code">{html.escape(str(row["away_team_id"]))}</span>
                <strong>{html.escape(str(row["away_team_name"]))}</strong>
            </div>
        </div>
        <div class="prediction-summary">
            <span>Modelo previa</span>
            <strong>{html.escape(predicted_label)}</strong>
            <em>{_format_metric_pct(confidence_pct)}</em>
        </div>
        {_probability_bar_html(row)}
        <dl class="prediction-details">
            <div>
                <dt>Placar modelo</dt>
                <dd>{html.escape(_format_score(row))}</dd>
            </div>
            <div>
                <dt>Resultado real</dt>
                <dd>{html.escape(actual_label)}</dd>
            </div>
            <div>
                <dt>Modelo vs real</dt>
                <dd>{status_badge}</dd>
            </div>
        </dl>
    </article>
    """,
    ).strip()


def _probability_bar_html(row: dict[str, Any]) -> str:
    probabilities = [
        ("home", row["home_team_name"], float(row.get("home_win_pct") or 0), "home"),
        ("draw", "Empate", float(row.get("draw_pct") or 0), "draw"),
        ("away", row["away_team_name"], float(row.get("away_win_pct") or 0), "away"),
    ]
    segments = []
    legend_items = []
    for outcome, label, value, css_class in probabilities:
        width = max(0.0, min(100.0, value))
        is_favorite = outcome == _most_likely_outcome(row)
        favorite_class = " is-favorite" if is_favorite else ""
        segments.append(
            (
                f'<span class="probability-segment {css_class}{favorite_class}" '
                f'style="width: {width:.4f}%"></span>'
            ),
        )
        legend_items.append(
            (
                f'<span class="{css_class}{favorite_class}">'
                f"<strong>{html.escape(str(label))}</strong>"
                f"<em>{_format_metric_pct(value)}</em>"
                "</span>"
            ),
        )
    return (
        '<div class="probability-visual">'
        f'<div class="probability-bar">{"".join(segments)}</div>'
        f'<div class="probability-legend">{"".join(legend_items)}</div>'
        "</div>"
    )


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


def _prediction_result_cell(row: dict[str, Any]) -> str:
    result = row.get("prediction_result")
    if result is None:
        return '<td class="result-cell result-pending">-</td>'
    css_class = "result-hit" if result == "Acerto" else "result-miss"
    return f'<td class="result-cell {css_class}">{html.escape(str(result))}</td>'
