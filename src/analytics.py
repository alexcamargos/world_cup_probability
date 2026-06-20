"""Analytical SQL views over the Monte Carlo simulation outputs.

This module summarizes ``simulated_results`` with DuckDB views and exports the
results to CSV via Polars for external visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "world_cup.duckdb"
EXPORT_DIR = PROJECT_ROOT / "reports" / "analytics"


def create_analytics_views(db_path: Path = DB_PATH) -> None:
    """Create analytical views for simulation summaries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB warehouse not found: {db_path}")

    with duckdb.connect(str(db_path)) as con:
        _validate_source(con)
        con.execute(_view_semifinal_reach_sql())
        con.execute(_view_title_probability_sql())
        con.execute(_view_final_matchup_sql())

    LOGGER.info("Analytical views created successfully.")


def export_analytics(db_path: Path = DB_PATH, output_dir: Path = EXPORT_DIR) -> dict[str, Path]:
    """Export analytical views to CSV files using Polars."""
    create_analytics_views(db_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path), read_only=True) as con:
        semifinal_df = con.sql(
            """
            SELECT *
            FROM v_semifinal_reach
            ORDER BY semifinal_pct DESC, team_name ASC
            """,
        ).pl()
        title_df = con.sql(
            """
            SELECT *
            FROM v_title_probability
            ORDER BY title_probability DESC, team_name ASC
            """,
        ).pl()
        final_matchup_df = con.sql("SELECT * FROM v_most_probable_final_matchup").pl()

    semifinal_path = output_dir / "semifinal_reach_probability.csv"
    title_path = output_dir / "title_probability.csv"
    final_matchup_path = output_dir / "most_probable_final_matchup.csv"

    semifinal_df.write_csv(semifinal_path)
    title_df.write_csv(title_path)
    final_matchup_df.write_csv(final_matchup_path)

    LOGGER.info("Exported analytical CSVs to %s", output_dir)
    return {
        "semifinal_reach": semifinal_path,
        "title_probability": title_path,
        "most_probable_final_matchup": final_matchup_path,
    }


def build_analytics_frame(db_path: Path = DB_PATH) -> pl.DataFrame:
    """Return a combined Polars frame for external inspection."""
    create_analytics_views(db_path)

    with duckdb.connect(str(db_path), read_only=True) as con:
        semifinal_df = con.sql(
            """
            SELECT
                'semifinal_reach' AS metric,
                team_id,
                team_name,
                semifinal_pct AS value
            FROM v_semifinal_reach
            """,
        ).pl()
        title_df = con.sql(
            """
            SELECT
                'title_probability' AS metric,
                team_id,
                team_name,
                title_probability AS value
            FROM v_title_probability
            """,
        ).pl()
        final_df = con.sql(
            """
            SELECT
                'most_probable_final_matchup' AS metric,
                matchup_id AS team_id,
                matchup_label AS team_name,
                matchup_probability AS value
            FROM v_most_probable_final_matchup
            """,
        ).pl()

    return pl.concat([semifinal_df, title_df, final_df], how="diagonal_relaxed")


def _validate_source(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure simulation results are available."""
    table_exists = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'simulated_results'
        """,
    ).fetchone()[0]
    if int(table_exists) == 0:
        raise RuntimeError("simulated_results table not found. Run the simulator first.")

    row_count = con.execute("SELECT COUNT(*) FROM simulated_results").fetchone()[0]
    if int(row_count) == 0:
        raise RuntimeError("simulated_results is empty. Run the simulator first.")


def _view_semifinal_reach_sql() -> str:
    return """
    CREATE OR REPLACE VIEW v_semifinal_reach AS
    WITH semifinal_participants AS (
        SELECT simulation_id, home_team_id AS team_id, home_team_name AS team_name
        FROM simulated_results
        WHERE round_name = 'semifinal'
        UNION ALL
        SELECT simulation_id, away_team_id AS team_id, away_team_name AS team_name
        FROM simulated_results
        WHERE round_name = 'semifinal'
    ),
    total_sims AS (
        SELECT COUNT(DISTINCT simulation_id) AS total_simulations
        FROM simulated_results
    )
    SELECT
        sp.team_id,
        sp.team_name,
        ROUND(
            100.0 * COUNT(DISTINCT sp.simulation_id) / NULLIF(ts.total_simulations, 0),
            2
        ) AS semifinal_pct,
        COUNT(DISTINCT sp.simulation_id) AS semifinal_appearances,
        ts.total_simulations
    FROM semifinal_participants AS sp
    CROSS JOIN total_sims AS ts
    GROUP BY sp.team_id, sp.team_name, ts.total_simulations
    ORDER BY semifinal_pct DESC, sp.team_name ASC
    """


def _view_title_probability_sql() -> str:
    return """
    CREATE OR REPLACE VIEW v_title_probability AS
    WITH champions AS (
        SELECT
            simulation_id,
            winner_team_id AS team_id,
            CASE
                WHEN winner_team_id = home_team_id THEN home_team_name
                ELSE away_team_name
            END AS team_name
        FROM simulated_results
        WHERE round_name = 'final'
    ),
    total_sims AS (
        SELECT COUNT(DISTINCT simulation_id) AS total_simulations
        FROM simulated_results
    )
    SELECT
        c.team_id,
        c.team_name,
        ROUND(100.0 * COUNT(*) / NULLIF(ts.total_simulations, 0), 2) AS title_probability,
        COUNT(*) AS title_wins,
        ts.total_simulations
    FROM champions AS c
    CROSS JOIN total_sims AS ts
    GROUP BY c.team_id, c.team_name, ts.total_simulations
    ORDER BY title_probability DESC, c.team_name ASC
    """


def _view_final_matchup_sql() -> str:
    return """
    CREATE OR REPLACE VIEW v_most_probable_final_matchup AS
    WITH finals AS (
        SELECT
            simulation_id,
            CASE
                WHEN home_team_id <= away_team_id THEN home_team_id
                ELSE away_team_id
            END AS team_a_id,
            CASE
                WHEN home_team_id <= away_team_id THEN home_team_name
                ELSE away_team_name
            END AS team_a_name,
            CASE
                WHEN home_team_id <= away_team_id THEN away_team_id
                ELSE home_team_id
            END AS team_b_id,
            CASE
                WHEN home_team_id <= away_team_id THEN away_team_name
                ELSE home_team_name
            END AS team_b_name
        FROM simulated_results
        WHERE round_name = 'final'
    ),
    pair_counts AS (
        SELECT
            team_a_id,
            team_b_id,
            team_a_name,
            team_b_name,
            COUNT(*) AS matchup_count
        FROM finals
        GROUP BY team_a_id, team_b_id, team_a_name, team_b_name
    ),
    total_sims AS (
        SELECT COUNT(DISTINCT simulation_id) AS total_simulations
        FROM simulated_results
    )
    SELECT
        matchup_id,
        matchup_label,
        matchup_probability,
        matchup_count,
        total_simulations
    FROM (
        SELECT
            CONCAT(team_a_id, '_vs_', team_b_id) AS matchup_id,
            CONCAT(team_a_name, ' vs ', team_b_name) AS matchup_label,
            ROUND(
                100.0 * matchup_count / NULLIF(ts.total_simulations, 0),
                2
            ) AS matchup_probability,
            matchup_count,
            ts.total_simulations,
            ROW_NUMBER() OVER (ORDER BY matchup_count DESC, team_a_name ASC, team_b_name ASC) AS rn
        FROM pair_counts
        CROSS JOIN total_sims AS ts
    ) ranked
    WHERE rn = 1
    """


def main() -> int:
    """CLI entrypoint."""
    output_paths = export_analytics()
    for name, path in output_paths.items():
        LOGGER.info("%s exported to %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
