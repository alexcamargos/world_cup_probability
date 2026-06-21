"""Initialize the local DuckDB warehouse and load raw files.

This module creates the DuckDB star schema in ``data/warehouse`` and ingests
compatible CSV/Parquet files found under ``data/raw``. Each raw file is mapped
through an explicit table loader so source validation and load ordering stay
centralized.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb

LOGGER = logging.getLogger(__name__)

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
        total_market_value_eur DOUBLE,
        market_value_currency VARCHAR,
        market_value_source_url VARCHAR,
        market_value_scraped_at TIMESTAMP,
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
    """
    CREATE TABLE IF NOT EXISTS f_match_stats (
        match_id VARCHAR NOT NULL,
        match_date DATE NOT NULL,
        team_id VARCHAR NOT NULL,
        opponent_team_id VARCHAR,
        tournament VARCHAR,
        xg DOUBLE,
        possession_pct DOUBLE,
        source VARCHAR NOT NULL,
        source_file VARCHAR,
        loaded_at TIMESTAMP,
        PRIMARY KEY (match_id, team_id, source),
        FOREIGN KEY (team_id) REFERENCES d_teams(team_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS d_squad_attributes (
        team_id VARCHAR NOT NULL,
        source_season VARCHAR NOT NULL,
        avg_overall DOUBLE,
        avg_pace DOUBLE,
        avg_stamina DOUBLE,
        sampled_player_count INTEGER,
        source_dataset VARCHAR,
        source_file VARCHAR,
        loaded_at TIMESTAMP,
        PRIMARY KEY (team_id, source_season),
        FOREIGN KEY (team_id) REFERENCES d_teams(team_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS d_world_football_elo_ratings (
        world_football_team_code VARCHAR PRIMARY KEY,
        team_name VARCHAR NOT NULL,
        elo_rank INTEGER NOT NULL,
        elo_rating DOUBLE NOT NULL,
        rating_date DATE,
        source_url VARCHAR,
        source_file VARCHAR,
        loaded_at TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS d_world_football_elo_team_aliases (
        team_alias_key VARCHAR PRIMARY KEY,
        team_alias VARCHAR NOT NULL,
        normalized_team_alias VARCHAR NOT NULL,
        world_football_team_code VARCHAR NOT NULL,
        source_file VARCHAR,
        loaded_at TIMESTAMP,
        FOREIGN KEY (world_football_team_code)
            REFERENCES d_world_football_elo_ratings(world_football_team_code)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS d_fifa_world_ranking (
        fifa_country_code VARCHAR PRIMARY KEY,
        fifa_team_id VARCHAR,
        team_name VARCHAR NOT NULL,
        confederation VARCHAR,
        fifa_rank INTEGER NOT NULL,
        previous_rank INTEGER,
        ranking_points DOUBLE NOT NULL,
        previous_points DOUBLE,
        ranking_movement INTEGER,
        matches INTEGER,
        ranking_date DATE NOT NULL,
        next_update_date DATE,
        source_url VARCHAR,
        api_url VARCHAR,
        source_file VARCHAR,
        loaded_at TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS d_fifa_world_ranking_team_aliases (
        team_alias_key VARCHAR PRIMARY KEY,
        team_alias VARCHAR NOT NULL,
        normalized_team_alias VARCHAR NOT NULL,
        fifa_country_code VARCHAR NOT NULL,
        source_file VARCHAR,
        loaded_at TIMESTAMP,
        FOREIGN KEY (fifa_country_code)
            REFERENCES d_fifa_world_ranking(fifa_country_code)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS data_collection_runs (
        run_id VARCHAR PRIMARY KEY,
        source_name VARCHAR NOT NULL,
        source_kind VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        raw_path VARCHAR,
        rows_loaded INTEGER,
        cutoff_date DATE,
        started_at TIMESTAMP NOT NULL,
        finished_at TIMESTAMP NOT NULL,
        error_message VARCHAR
    );
    """,
)


@dataclass(frozen=True, slots=True)
class ColumnProjection:
    """Map one warehouse column to accepted raw aliases."""

    target_column: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableLoader:
    """Describe how to project a raw file into one warehouse table."""

    table_name: str
    primary_key: str
    required_alias_groups: tuple[tuple[str, ...], ...]
    column_map: tuple[ColumnProjection, ...]

    @property
    def target_columns(self) -> tuple[str, ...]:
        """Return target columns in insertion order."""
        return tuple(projection.target_column for projection in self.column_map)

    def can_load(self, available_columns: set[str]) -> bool:
        """Return whether the source has all required alias groups."""
        return all(
            any(alias in available_columns for alias in alias_group)
            for alias_group in self.required_alias_groups
        )


TEAM_LOADER = TableLoader(
    table_name="d_teams",
    primary_key="team_id",
    required_alias_groups=(
        ("team_id", "id", "club_id", "team_name", "team", "club", "name", "squad"),
    ),
    column_map=(
        ColumnProjection(
            "team_id", ("team_id", "id", "club_id", "team_name", "team", "club", "name", "squad")
        ),
        ColumnProjection(
            "team_name", ("team_name", "team", "club", "name", "squad", "team_id", "id", "club_id")
        ),
        ColumnProjection("country", ("country", "nation")),
        ColumnProjection("confederation", ("confederation",)),
        ColumnProjection("market_value_eur", ("market_value_eur", "market_value", "squad_value")),
        ColumnProjection("market_value_currency", ("market_value_currency", "currency")),
        ColumnProjection("squad_size", ("squad_size", "players")),
        ColumnProjection("coach_name", ("coach_name", "manager")),
        ColumnProjection("source_file", ()),
        ColumnProjection("loaded_at", ()),
    ),
)

SQUAD_LOADER = TableLoader(
    table_name="d_squads",
    primary_key="squad_id",
    required_alias_groups=(
        ("player_id", "player_name", "player"),
        ("team_id", "team_name", "club"),
    ),
    column_map=(
        ColumnProjection("squad_id", ("squad_id", "id")),
        ColumnProjection("team_id", ("team_id", "team_name", "club")),
        ColumnProjection("player_id", ("player_id", "id_player")),
        ColumnProjection(
            "player_name", ("player_name", "player", "name", "player_id", "id_player")
        ),
        ColumnProjection("position", ("position", "role")),
        ColumnProjection("birth_date", ("birth_date", "dob", "date_of_birth")),
        ColumnProjection("age", ("age",)),
        ColumnProjection("nationality", ("nationality", "country")),
        ColumnProjection("height_cm", ("height_cm", "height")),
        ColumnProjection("weight_kg", ("weight_kg", "weight")),
        ColumnProjection("preferred_foot", ("preferred_foot", "foot")),
        ColumnProjection("market_value_eur", ("market_value_eur", "market_value")),
        ColumnProjection("jersey_number", ("jersey_number", "shirt_number", "number")),
        ColumnProjection("source_file", ()),
        ColumnProjection("loaded_at", ()),
    ),
)

MATCH_LOADER = TableLoader(
    table_name="f_matches",
    primary_key="match_id",
    required_alias_groups=(
        ("match_date", "date"),
        ("home_team_id", "home_team", "home"),
        ("away_team_id", "away_team", "away"),
    ),
    column_map=(
        ColumnProjection("match_id", ("match_id", "id")),
        ColumnProjection("match_date", ("match_date", "date")),
        ColumnProjection("competition", ("competition", "tournament", "league")),
        ColumnProjection("season", ("season", "year")),
        ColumnProjection("stage", ("stage", "round", "phase")),
        ColumnProjection("home_team_id", ("home_team_id", "home_team", "home")),
        ColumnProjection("away_team_id", ("away_team_id", "away_team", "away")),
        ColumnProjection(
            "home_team_score", ("home_team_score", "home_score", "score_home", "goals_home")
        ),
        ColumnProjection(
            "away_team_score", ("away_team_score", "away_score", "score_away", "goals_away")
        ),
        ColumnProjection("home_xg", ("home_xg",)),
        ColumnProjection("away_xg", ("away_xg",)),
        ColumnProjection("venue", ("venue", "stadium")),
        ColumnProjection("city", ("city",)),
        ColumnProjection("country", ("country",)),
        ColumnProjection("attendance", ("attendance",)),
        ColumnProjection("neutral_site", ("neutral_site", "neutral")),
        ColumnProjection("source_file", ()),
        ColumnProjection("loaded_at", ()),
    ),
)

TABLE_LOADERS: tuple[TableLoader, ...] = (TEAM_LOADER, SQUAD_LOADER, MATCH_LOADER)


def initialize_database(
    db_path: Path = DB_PATH,
    raw_dir: Path = RAW_DIR,
    *,
    load_raw_files: bool = True,
) -> Path:
    """Create the warehouse schema and optionally load compatible raw inputs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    with duckdb.connect(str(db_path)) as con:
        _create_schema(con)
        _migrate_schema(con)
        if load_raw_files:
            _load_raw_files(con, raw_dir)

    return db_path


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:
    for statement in DDL_STATEMENTS:
        con.execute(statement)


def _migrate_schema(con: duckdb.DuckDBPyConnection) -> None:
    migration_statements = (
        "ALTER TABLE d_teams ADD COLUMN IF NOT EXISTS total_market_value_eur DOUBLE",
        "ALTER TABLE d_teams ADD COLUMN IF NOT EXISTS market_value_source_url VARCHAR",
        "ALTER TABLE d_teams ADD COLUMN IF NOT EXISTS market_value_scraped_at TIMESTAMP",
    )
    for statement in migration_statements:
        con.execute(statement)


def _load_raw_files(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    raw_files = _discover_raw_files(raw_dir)

    for loader in TABLE_LOADERS:
        for raw_file in raw_files:
            columns = _read_source_columns(con, raw_file)
            if not columns:
                continue

            available_columns = {column.lower() for column in columns}
            if not loader.can_load(available_columns):
                continue

            _load_raw_file_into_table(con, raw_file, columns, loader)


def _discover_raw_files(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".parquet", ".csv"}
    )


def _load_raw_file_into_table(
    con: duckdb.DuckDBPyConnection,
    raw_file: Path,
    columns: Iterable[str],
    loader: TableLoader,
) -> None:
    source_sql = _source_sql(raw_file)
    column_lookup = {column.lower(): column for column in columns}

    if loader.table_name == "d_squads":
        _ensure_teams_from_source(
            con, source_sql, column_lookup, raw_file, ("team_id", "team_name", "club")
        )
    elif loader.table_name == "f_matches":
        _ensure_teams_from_source(
            con,
            source_sql,
            column_lookup,
            raw_file,
            ("home_team_id", "home_team", "home"),
        )
        _ensure_teams_from_source(
            con,
            source_sql,
            column_lookup,
            raw_file,
            ("away_team_id", "away_team", "away"),
        )

    select_list = _build_select_list(column_lookup, loader, raw_file)
    con.execute(f"DELETE FROM {loader.table_name} WHERE source_file = ?", [str(raw_file)])
    con.execute(_insert_sql(loader, select_list, source_sql))
    LOGGER.info("Loaded %s into %s.", raw_file, loader.table_name)


def _read_source_columns(con: duckdb.DuckDBPyConnection, raw_file: Path) -> list[str]:
    relation_sql = _source_sql(raw_file)
    rows = con.execute(f"DESCRIBE SELECT * FROM {relation_sql}").fetchall()
    return [str(row[0]) for row in rows]


def _source_sql(raw_file: Path) -> str:
    file_path = _sql_string_literal(raw_file.as_posix())
    if raw_file.suffix.lower() == ".parquet":
        return f"read_parquet({file_path})"
    if raw_file.suffix.lower() == ".csv":
        return f"read_csv_auto({file_path}, header=true)"
    raise ValueError(f"Unsupported raw file extension: {raw_file.suffix}")


def _build_select_list(
    column_lookup: dict[str, str],
    loader: TableLoader,
    raw_file: Path,
) -> str:
    select_items: list[str] = []

    for projection in loader.column_map:
        target_column = projection.target_column
        expression = _projection_expression(
            target_column, projection.aliases, column_lookup, raw_file
        )
        select_items.append(f"{expression} AS {target_column}")

    return ", ".join(select_items)


def _projection_expression(
    target_column: str,
    aliases: tuple[str, ...],
    column_lookup: dict[str, str],
    raw_file: Path,
) -> str:
    if target_column == "source_file":
        return _sql_string_literal(str(raw_file))
    if target_column == "loaded_at":
        return "current_timestamp"

    matching_alias = next((alias for alias in aliases if alias.lower() in column_lookup), None)
    if matching_alias is None:
        return _generated_or_null_expression(target_column, raw_file)

    resolved_name = column_lookup[matching_alias.lower()]
    return f"CAST({_quote_identifier(resolved_name)} AS {_sql_type_for_column(target_column)})"


def _generated_or_null_expression(target_column: str, raw_file: Path) -> str:
    if target_column in {"match_id", "squad_id"}:
        source_literal = _sql_string_literal(str(raw_file))
        return f"md5({source_literal} || ':' || CAST(row_number() OVER () AS VARCHAR))"
    return f"NULL::{_sql_type_for_column(target_column)}"


def _insert_sql(loader: TableLoader, select_list: str, source_sql: str) -> str:
    columns = ", ".join(loader.target_columns)
    updates = ", ".join(
        f"{column} = excluded.{column}"
        for column in loader.target_columns
        if column != loader.primary_key
    )
    return f"""
        INSERT INTO {loader.table_name} ({columns})
        SELECT {select_list}
        FROM {source_sql}
        ON CONFLICT ({loader.primary_key}) DO UPDATE SET {updates}
    """


def _ensure_teams_from_source(
    con: duckdb.DuckDBPyConnection,
    source_sql: str,
    column_lookup: dict[str, str],
    raw_file: Path,
    aliases: tuple[str, ...],
) -> None:
    team_column = _first_matching_column(column_lookup, aliases)
    if team_column is None:
        return

    team_expression = f"CAST({_quote_identifier(team_column)} AS VARCHAR)"
    source_file = _sql_string_literal(f"{raw_file}#auto-team")
    con.execute(
        f"""
        INSERT INTO d_teams (
            team_id,
            team_name,
            country,
            confederation,
            market_value_eur,
            market_value_currency,
            squad_size,
            coach_name,
            source_file,
            loaded_at
        )
        SELECT DISTINCT
            {team_expression} AS team_id,
            {team_expression} AS team_name,
            NULL::VARCHAR AS country,
            NULL::VARCHAR AS confederation,
            NULL::DOUBLE AS market_value_eur,
            NULL::VARCHAR AS market_value_currency,
            NULL::INTEGER AS squad_size,
            NULL::VARCHAR AS coach_name,
            {source_file} AS source_file,
            current_timestamp AS loaded_at
        FROM {source_sql}
        WHERE {team_expression} IS NOT NULL
        ON CONFLICT (team_id) DO NOTHING
        """,
    )


def _first_matching_column(column_lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    matching_alias = next((alias for alias in aliases if alias.lower() in column_lookup), None)
    if matching_alias is None:
        return None
    return column_lookup[matching_alias.lower()]


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
        "total_market_value_eur": "DOUBLE",
        "market_value_currency": "VARCHAR",
        "market_value_source_url": "VARCHAR",
        "market_value_scraped_at": "TIMESTAMP",
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


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    """Command-line entrypoint."""
    db_file = initialize_database()
    LOGGER.info("DuckDB initialized at: %s", db_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
