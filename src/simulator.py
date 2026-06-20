"""Monte Carlo simulator for the World Cup knockout bracket.

The simulator consumes Poisson goal lambdas produced by the XGBoost model,
generates exact integer scores with ``numpy.random.poisson(lam)``, resolves
ties with a simple penalty rule when a winner is required, and persists every
simulated match to DuckDB in batches to keep memory usage bounded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "warehouse" / "world_cup.duckdb"


@dataclass(frozen=True, slots=True)
class TeamLambda:
    """Predicted scoring intensity for one team."""

    team_id: str
    team_name: str
    lambda_goals: float


@dataclass(frozen=True, slots=True)
class SimulatedMatch:
    """Single simulated match result."""

    simulation_id: int
    round_name: str
    match_number: int
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    home_lambda: float
    away_lambda: float
    home_goals: int
    away_goals: int
    require_winner: bool
    decided_by_penalties: bool
    penalty_winner_team_id: str | None
    winner_team_id: str
    created_at: datetime


def initialize_simulated_results(db_path: Path = DB_PATH) -> None:
    """Create the simulation results table if it does not exist."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS simulated_results (
                simulation_id INTEGER NOT NULL,
                round_name VARCHAR NOT NULL,
                match_number INTEGER NOT NULL,
                home_team_id VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL,
                home_team_name VARCHAR,
                away_team_name VARCHAR,
                home_lambda DOUBLE NOT NULL,
                away_lambda DOUBLE NOT NULL,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL,
                require_winner BOOLEAN NOT NULL,
                decided_by_penalties BOOLEAN NOT NULL,
                penalty_winner_team_id VARCHAR,
                winner_team_id VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL
            );
            """,
        )


def simulate_match(
    home_team: TeamLambda,
    away_team: TeamLambda,
    *,
    require_winner: bool,
    rng: np.random.Generator,
) -> SimulatedMatch:
    """Simulate a single match using Poisson scoring and optional penalties."""
    home_lambda = max(float(home_team.lambda_goals), 0.0)
    away_lambda = max(float(away_team.lambda_goals), 0.0)

    home_goals = int(rng.poisson(home_lambda))
    away_goals = int(rng.poisson(away_lambda))

    decided_by_penalties = False
    penalty_winner_team_id: str | None = None

    if home_goals > away_goals:
        winner_team_id = home_team.team_id
    elif away_goals > home_goals:
        winner_team_id = away_team.team_id
    elif require_winner:
        decided_by_penalties = True
        penalty_winner_team_id = _resolve_penalties(home_team.team_id, away_team.team_id, rng)
        winner_team_id = penalty_winner_team_id
    else:
        winner_team_id = home_team.team_id

    return SimulatedMatch(
        simulation_id=-1,
        round_name="",
        match_number=-1,
        home_team_id=home_team.team_id,
        away_team_id=away_team.team_id,
        home_team_name=home_team.team_name,
        away_team_name=away_team.team_name,
        home_lambda=home_lambda,
        away_lambda=away_lambda,
        home_goals=home_goals,
        away_goals=away_goals,
        require_winner=require_winner,
        decided_by_penalties=decided_by_penalties,
        penalty_winner_team_id=penalty_winner_team_id,
        winner_team_id=winner_team_id,
        created_at=datetime.now(UTC),
    )


def simulate_world_cup(
    teams: Sequence[TeamLambda],
    *,
    iterations: int = 100_000,
    batch_size: int = 2_500,
    db_path: Path = DB_PATH,
    seed: int | None = None,
) -> None:
    """Run a Monte Carlo simulation for the knockout bracket.

    Args:
        teams: Ordered teams for the initial bracket.
        iterations: Number of tournament runs to execute.
        batch_size: How many simulated matches to persist per insert batch.
        db_path: Local DuckDB warehouse path.
        seed: Optional random seed for reproducibility.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    if len(teams) < 2 or not _is_power_of_two(len(teams)):
        raise ValueError("teams must contain a power-of-two number of entries.")

    initialize_simulated_results(db_path)
    rng = np.random.default_rng(seed)

    with duckdb.connect(str(db_path)) as con:
        con.execute("DELETE FROM simulated_results")

        buffer: list[tuple[object, ...]] = []
        for simulation_id in range(1, iterations + 1):
            tournament_rows = _simulate_tournament(teams, simulation_id, rng)
            buffer.extend(tournament_rows)

            if len(buffer) >= batch_size:
                _flush_rows(con, buffer)
                buffer.clear()

        if buffer:
            _flush_rows(con, buffer)

    LOGGER.info("Finished %d Monte Carlo simulations.", iterations)


def _simulate_tournament(
    teams: Sequence[TeamLambda],
    simulation_id: int,
    rng: np.random.Generator,
) -> list[tuple[object, ...]]:
    """Simulate one complete knockout bracket and return the rows to persist."""
    current_round = list(teams)
    round_index = 1
    rows: list[tuple[object, ...]] = []

    while len(current_round) >= 2:
        round_name = _round_name(len(current_round), round_index)
        require_winner = True
        next_round: list[TeamLambda] = []

        for match_number, pair in enumerate(_pairwise(current_round), start=1):
            home_team, away_team = pair
            result = simulate_match(
                home_team,
                away_team,
                require_winner=require_winner,
                rng=rng,
            )
            rows.append(
                (
                    simulation_id,
                    round_name,
                    match_number,
                    result.home_team_id,
                    result.away_team_id,
                    result.home_team_name,
                    result.away_team_name,
                    result.home_lambda,
                    result.away_lambda,
                    result.home_goals,
                    result.away_goals,
                    result.require_winner,
                    result.decided_by_penalties,
                    result.penalty_winner_team_id,
                    result.winner_team_id,
                    result.created_at,
                )
            )

            winner = home_team if result.winner_team_id == home_team.team_id else away_team
            next_round.append(winner)

        current_round = next_round
        round_index += 1

    return rows


def _flush_rows(con: duckdb.DuckDBPyConnection, rows: list[tuple[object, ...]]) -> None:
    """Persist a batch of rows into DuckDB."""
    con.executemany(
        """
        INSERT INTO simulated_results (
            simulation_id,
            round_name,
            match_number,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name,
            home_lambda,
            away_lambda,
            home_goals,
            away_goals,
            require_winner,
            decided_by_penalties,
            penalty_winner_team_id,
            winner_team_id,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _resolve_penalties(
    home_team_id: str,
    away_team_id: str,
    rng: np.random.Generator,
) -> str:
    """Resolve a tie with a simple 50/50 penalty shootout."""
    return home_team_id if int(rng.integers(0, 2)) == 0 else away_team_id


def _pairwise(items: Sequence[TeamLambda]) -> Iterable[tuple[TeamLambda, TeamLambda]]:
    """Yield bracket pairs in order."""
    for index in range(0, len(items), 2):
        yield items[index], items[index + 1]


def _round_name(participants: int, round_index: int) -> str:
    """Generate a human-readable knockout round label."""
    labels = {
        32: "round_of_32",
        16: "round_of_16",
        8: "quarterfinal",
        4: "semifinal",
        2: "final",
    }
    return labels.get(participants, f"round_{round_index}")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def main() -> int:
    """CLI entrypoint."""
    LOGGER.info("Simulator module loaded. Import and call simulate_world_cup().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
