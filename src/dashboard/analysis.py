"""Pure dashboard analysis helpers independent from Streamlit rendering."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

try:
    from ..world_cup_2026_schedule import WorldCupFixture
except ImportError:  # pragma: no cover - supports `streamlit run src/app.py`.
    from world_cup_2026_schedule import WorldCupFixture

from .constants import ROUND_DISPLAY_ORDER, ROUND_LABELS, ROUND_ORDER, RoundOption
from .formatting import _team_display_name


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


def _actual_outcome(row: dict[str, Any]) -> str | None:
    actual_home_goals = row.get("actual_home_goals")
    actual_away_goals = row.get("actual_away_goals")
    if actual_home_goals is None or actual_away_goals is None:
        return None
    return _score_outcome(int(actual_home_goals), int(actual_away_goals))
