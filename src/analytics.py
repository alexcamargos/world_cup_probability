"""Analytical SQL views over the Monte Carlo simulation outputs.

This module summarizes ``simulated_results`` with DuckDB views and exports the
results to CSV via Polars for external visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

try:
    from .settings import ANALYTICS_EXPORT_DIR as EXPORT_DIR
    from .settings import DB_PATH
    from .sql_loader import read_sql
except ImportError:  # pragma: no cover - supports direct script execution.
    from settings import ANALYTICS_EXPORT_DIR as EXPORT_DIR
    from settings import DB_PATH
    from sql_loader import read_sql

LOGGER = logging.getLogger(__name__)


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
    return read_sql("analytics/semifinal_reach.sql")


def _view_title_probability_sql() -> str:
    return read_sql("analytics/title_probability.sql")


def _view_final_matchup_sql() -> str:
    return read_sql("analytics/most_probable_final_matchup.sql")


def main() -> int:
    """CLI entrypoint."""
    output_paths = export_analytics()
    for name, path in output_paths.items():
        LOGGER.info("%s exported to %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
