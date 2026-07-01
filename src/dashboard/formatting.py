"""Formatting helpers shared by dashboard queries and views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import TEAM_NAMES_PT_BR


def _team_display_name(team_id: str, fallback: str) -> str:
    return TEAM_NAMES_PT_BR.get(team_id, fallback)


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


def _format_metric_pct(value: Any) -> str:
    return f"{float(value or 0):.1f}%"


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


def _format_compact_pct(value: Any) -> str:
    return f"{float(value or 0):.0f}%"


def _format_one_decimal(value: Any) -> str:
    return f"{float(value or 0):.1f}"


def _probability_source_label(row: dict[str, Any]) -> str:
    if row.get("probability_source") == "outcome_model":
        return "V/E/D"
    return "Sim."


def _friendly_db_error(exc: Exception, db_path: Path) -> str:
    message = str(exc)
    if "used by another process" in message or "sendo usado por outro processo" in message:
        return (
            f"Nao foi possivel abrir {db_path}. O arquivo DuckDB esta em uso por outro processo; "
            "aguarde a simulacao terminar ou feche o processo que esta com o banco aberto."
        )
    return message
