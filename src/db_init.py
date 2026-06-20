"""Initialize the local DuckDB warehouse and load the first raw files.

This module creates a small star schema in ``data/warehouse/world_cup.duckdb``
and loads any compatible files found under ``data/raw``.

Parquet files are ingested with ``read_parquet()``. CSV files are supported as a
fallback via ``read_csv_auto()`` so the bootstrap routine can accept either raw
format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
WAREHOUSE_DIR = PROJECT_ROOT / "data" / "warehouse"
DB_PATH = WAREHOUSE_DIR / "world_cup.duckdb"

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS d_teams (
        team_id VARCHAR PRIMARY KEY,
        team_name VARCHAR NOT NULL,
        country VARCHAR,
        confederation VARCHAR,
        market_value_eur DOUBLE,
        market_value_currency VARCHAR,
        squad_size INTEGER,
        coach_name VARCHAR,
        source_file VARCHAR,
        loaded_at TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS d_squads (
        squad_id VARCHAR PRIMARY KEY,
        team_id VARCHAR NOT NULL,
        player_id VARCHAR,
        player_name VARCHAR NOT NULL,
        position VARCHAR,
        birth_date DATE,
        age INTEGER,
        nationality VARCHAR,
        height_cm DOUBLE,
        weight_kg DOUBLE,
        preferred_foot VARCHAR,
        market_value_eur DOUBLE,
        jersey_number INTEGER,
        source_file VARCHAR,
        loaded_at TIMESTAMP,
        FOREIGN KEY (team_id) REFERENCES d_teams(team_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS f_matches (
        match_id VARCHAR PRIMARY KEY,
        match_date DATE NOT NULL,
        competition VARCHAR,
        season VARCHAR,
        stage VARCHAR,
        home_team_id VARCHAR NOT NULL,
        away_team_id VARCHAR NOT NULL,
        home_team_score INTEGER,
        away_team_score INTEGER,
        home_xg DOUBLE,
        away_xg DOUBLE,
        venue VARCHAR,
        city VARCHAR,
        country VARCHAR,
        attendance INTEGER,
        neutral_site BOOLEAN,
        source_file VARCHAR,
        loaded_at TIMESTAMP,
        FOREIGN KEY (home_team_id) REFERENCES d_teams(team_id),
        FOREIGN KEY (away_team_id) REFERENCES d_teams(team_id)
    );
    """,
)

TABLE_LOADERS: tuple["TableLoader", ...] = ()


@dataclass(frozen=True, slots=True)
class TableLoader:
    """Describe how to project a raw file into a warehouse table."""

    table_name: str
    required_any: tuple[tuple[str, ...], ...]
    column_map: tuple[tuple[str, tuple[str, ...], str], ...]

    def can_load(self, available_columns: set[str]) -> bool:
        """Return whether the source has enough columns for this table."""
        if self.required_any and not any(
            any(alias in available_columns for alias in alias_group)
            for alias_group in self.required_any
        ):
            return False

        return True


TABLE_LOADERS = (
    TableLoader(
        table_name="f_matches",
        required_any=(
            ("match_date", "date"),
            ("home_team_id", "home_team"),
            ("away_team_id", "away_team"),
        ),
        column_map=(
            ("match_id", ("match_id", "id"), "CAST(coalesce(match_id, id) AS VARCHAR)"),
            (
                "match_date",
                ("match_date", "date"),
                "CAST(coalesce(match_date, date) AS DATE)",
            ),
            ("competition", ("competition", "tournament", "league"), "CAST(coalesce(competition, tournament, league) AS VARCHAR)"),
            ("season", ("season", "year"), "CAST(coalesce(season, year) AS VARCHAR)"),
            ("stage", ("stage", "round", "phase"), "CAST(coalesce(stage, round, phase) AS VARCHAR)"),
            ("home_team_id", ("home_team_id", "home_team", "home"), "CAST(coalesce(home_team_id, home_team, home) AS VARCHAR)"),
            ("away_team_id", ("away_team_id", "away_team", "away"), "CAST(coalesce(away_team_id, away_team, away) AS VARCHAR)"),
            ("home_team_score", ("home_team_score", "home_score", "score_home", "goals_home"), "CAST(coalesce(home_team_score, home_score, score_home, goals_home) AS INTEGER)"),
            ("away_team_score", ("away_team_score", "away_score", "score_away", "goals_away"), "CAST(coalesce(away_team_score, away_score, score_away, goals_away) AS INTEGER)"),
            ("home_xg", ("home_xg",), "CAST(home_xg AS DOUBLE)"),
            ("away_xg", ("away_xg",), "CAST(away_xg AS DOUBLE)"),
            ("venue", ("venue", "stadium"), "CAST(coalesce(venue, stadium) AS VARCHAR)"),
            ("city", ("city",), "CAST(city AS VARCHAR)"),
            ("country", ("country",), "CAST(country AS VARCHAR)"),
            ("attendance", ("attendance",), "CAST(attendance AS INTEGER)"),
            ("neutral_site", ("neutral_site", "neutral"), "CAST(coalesce(neutral_site, neutral) AS BOOLEAN)"),
            ("source_file", tuple(), "CAST(source_file AS VARCHAR)"),
            ("loaded_at", tuple(), "current_timestamp"),
        ),
    ),
    TableLoader(
        table_name="d_teams",
        required_any=(
            ("team_id", "team_name", "club", "squad"),
            ("market_value_eur", "market_value", "squad_value"),
        ),
        column_map=(
            ("team_id", ("team_id", "id", "club_id"), "CAST(coalesce(team_id, id, club_id) AS VARCHAR)"),
            ("team_name", ("team_name", "team", "club", "name"), "CAST(coalesce(team_name, team, club, name) AS VARCHAR)"),
            ("country", ("country", "nation"), "CAST(coalesce(country, nation) AS VARCHAR)"),
            ("confederation", ("confederation",), "CAST(confederation AS VARCHAR)"),
            ("market_value_eur", ("market_value_eur", "market_value", "squad_value"), "CAST(coalesce(market_value_eur, market_value, squad_value) AS DOUBLE)"),
            ("market_value_currency", ("market_value_currency", "currency"), "CAST(coalesce(market_value_currency, currency) AS VARCHAR)"),
            ("squad_size", ("squad_size", "players"), "CAST(coalesce(squad_size, players) AS INTEGER)"),
            ("coach_name", ("coach_name", "manager"), "CAST(coalesce(coach_name, manager) AS VARCHAR)"),
            ("source_file", tuple(), "CAST(source_file AS VARCHAR)"),
            ("loaded_at", tuple(), "current_timestamp"),
        ),
    ),
    TableLoader(
        table_name="d_squads",
        required_any=(
            ("player_id", "player_name", "player"),
            ("team_id", "team_name", "club"),
        ),
        column_map=(
            ("squad_id", ("squad_id", "id"), "CAST(coalesce(squad_id, id) AS VARCHAR)"),
            ("team_id", ("team_id", "team_name", "club"), "CAST(coalesce(team_id, team_name, club) AS VARCHAR)"),
            ("player_id", ("player_id", "id_player"), "CAST(coalesce(player_id, id_player) AS VARCHAR)"),
            ("player_name", ("player_name", "player", "name"), "CAST(coalesce(player_name, player, name) AS VARCHAR)"),
            ("position", ("position", "role"), "CAST(coalesce(position, role) AS VARCHAR)"),
            ("birth_date", ("birth_date", "dob", "date_of_birth"), "CAST(coalesce(birth_date, dob, date_of_birth) AS DATE)"),
            ("age", ("age",), "CAST(age AS INTEGER)"),
            ("nationality", ("nationality", "country"), "CAST(coalesce(nationality, country) AS VARCHAR)"),
            ("height_cm", ("height_cm", "height"), "CAST(coalesce(height_cm, height) AS DOUBLE)"),
            ("weight_kg", ("weight_kg", "weight"), "CAST(coalesce(weight_kg, weight) AS DOUBLE)"),
            ("preferred_foot", ("preferred_foot", "foot"), "CAST(coalesce(preferred_foot, foot) AS VARCHAR)"),
            ("market_value_eur", ("market_value_eur", "market_value"), "CAST(coalesce(market_value_eur, market_value) AS DOUBLE)"),
            ("jersey_number", ("jersey_number", "shirt_number", "number"), "CAST(coalesce(jersey_number, shirt_number, number) AS INTEGER)"),
            ("source_file", tuple(), "CAST(source_file AS VARCHAR)"),
            ("loaded_at", tuple(), "current_timestamp"),
        ),
    ),
)


def initialize_database(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR) -> Path:
    """Create the warehouse schema and load compatible raw inputs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        _create_schema(con)
        _load_raw_files(con, raw_dir)
    finally:
        con.close()

    return db_path


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:
    for statement in DDL_STATEMENTS:
        con.execute(statement)


def _load_raw_files(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    raw_files = sorted(
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".parquet", ".csv"}
    )

    for raw_file in raw_files:
        columns = _read_source_columns(con, raw_file)
        if not columns:
            continue

        available_columns = {column.lower() for column in columns}
        source_sql = _source_sql(raw_file)
        loaded_any_table = False

        for loader in TABLE_LOADERS:
            if not loader.can_load(available_columns):
                continue

            select_list = _build_select_list(columns, loader, raw_file)
            if select_list is None:
                continue

            con.execute(f"DELETE FROM {loader.table_name} WHERE source_file = ?", [str(raw_file)])
            con.execute(
                f"""
                INSERT INTO {loader.table_name}
                SELECT {select_list}
                FROM {source_sql}
                """,
            )
            loaded_any_table = True

        if not loaded_any_table:
            continue


def _read_source_columns(con: duckdb.DuckDBPyConnection, raw_file: Path) -> list[str]:
    relation_sql = _source_sql(raw_file)
    rows = con.execute(f"DESCRIBE SELECT * FROM {relation_sql}").fetchall()
    return [str(row[0]) for row in rows]


def _source_sql(raw_file: Path) -> str:
    file_path = raw_file.as_posix().replace("'", "''")
    if raw_file.suffix.lower() == ".parquet":
        return f"read_parquet('{file_path}')"
    if raw_file.suffix.lower() == ".csv":
        return f"read_csv_auto('{file_path}', header=true)"
    raise ValueError(f"Unsupported raw file extension: {raw_file.suffix}")


def _build_select_list(
    columns: Iterable[str],
    loader: TableLoader,
    raw_file: Path,
) -> str | None:
    column_lookup = {column.lower(): column for column in columns}
    select_items: list[str] = []

    for target_column, aliases, expression in loader.column_map:
        if target_column == "source_file":
            continue
        if target_column == "loaded_at":
            continue

        if not aliases:
            select_items.append(f"{expression} AS {target_column}")
            continue

        matching_alias = next((alias for alias in aliases if alias.lower() in column_lookup), None)
        if matching_alias is None:
            if target_column in {"source_file", "loaded_at"}:
                select_items.append(f"{expression} AS {target_column}")
            else:
                select_items.append(f"NULL::{_sql_type_for_column(target_column)} AS {target_column}")
            continue

        resolved_name = column_lookup[matching_alias.lower()]
        select_items.append(
            f"CAST({_quote_identifier(resolved_name)} AS {_sql_type_for_column(target_column)}) AS {target_column}"
        )

    if not select_items:
        return None

    # The table-specific required-any check already filters weak matches; a
    # fully padded projection keeps the star schema types strict.
    select_items.append(f"CAST('{str(raw_file)}' AS VARCHAR) AS source_file")
    select_items.append("current_timestamp AS loaded_at")
    return ", ".join(_dedupe_selected_columns(select_items))


def _dedupe_selected_columns(select_items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in select_items:
        target_name = item.rsplit(" AS ", 1)[-1]
        if target_name in seen:
            continue
        seen.add(target_name)
        deduped.append(item)
    return deduped


def _sql_type_for_column(column_name: str) -> str:
    column_types: dict[str, str] = {
        "match_id": "VARCHAR",
        "match_date": "DATE",
        "competition": "VARCHAR",
        "season": "VARCHAR",
        "stage": "VARCHAR",
        "home_team_id": "VARCHAR",
        "away_team_id": "VARCHAR",
        "home_team_score": "INTEGER",
        "away_team_score": "INTEGER",
        "home_xg": "DOUBLE",
        "away_xg": "DOUBLE",
        "venue": "VARCHAR",
        "city": "VARCHAR",
        "country": "VARCHAR",
        "attendance": "INTEGER",
        "neutral_site": "BOOLEAN",
        "team_id": "VARCHAR",
        "team_name": "VARCHAR",
        "confederation": "VARCHAR",
        "market_value_eur": "DOUBLE",
        "market_value_currency": "VARCHAR",
        "squad_size": "INTEGER",
        "coach_name": "VARCHAR",
        "squad_id": "VARCHAR",
        "player_id": "VARCHAR",
        "player_name": "VARCHAR",
        "position": "VARCHAR",
        "birth_date": "DATE",
        "age": "INTEGER",
        "nationality": "VARCHAR",
        "height_cm": "DOUBLE",
        "weight_kg": "DOUBLE",
        "preferred_foot": "VARCHAR",
        "jersey_number": "INTEGER",
        "source_file": "VARCHAR",
        "loaded_at": "TIMESTAMP",
    }
    return column_types[column_name]


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def main() -> int:
    """Command-line entrypoint."""
    db_file = initialize_database()
    print(f"DuckDB initialized at: {db_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
