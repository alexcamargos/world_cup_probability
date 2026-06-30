"""Monte Carlo simulator for the FIFA World Cup 2026 tournament path.

The simulator uses the real 104-match schedule as the target bracket, plays the
group stage, selects the top two teams from each group plus the eight best
third-place teams, and resolves the knockout slots through the final. Real match
scores from the current tournament are never used as simulation inputs.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import xgboost as xgb

try:
    from .analytics import export_analytics
    from .outcome_model import OUTCOME_CALIBRATION_PATH, OUTCOME_MODEL_PATH
    from .outcome_predictions import (
        build_current_world_cup_team_features,
        load_calibration_temperature,
        load_outcome_model_artifact,
        predict_match_probabilities,
    )
    from .settings import (
        DB_PATH,
        DEFAULT_BATCH_SIZE,
        DEFAULT_ITERATIONS,
        DEFAULT_SEED,
        MODEL_PATH,
    )
    from .world_cup_2026_schedule import (
        TEAM_COUNTRIES,
        TEAM_NAMES,
        WorldCupFixture,
        world_cup_2026_fixtures,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from analytics import export_analytics
    from outcome_model import OUTCOME_CALIBRATION_PATH, OUTCOME_MODEL_PATH
    from outcome_predictions import (
        build_current_world_cup_team_features,
        load_calibration_temperature,
        load_outcome_model_artifact,
        predict_match_probabilities,
    )
    from settings import DB_PATH, DEFAULT_BATCH_SIZE, DEFAULT_ITERATIONS, DEFAULT_SEED, MODEL_PATH
    from world_cup_2026_schedule import (  # type: ignore[no-redef]
        TEAM_COUNTRIES,
        TEAM_NAMES,
        WorldCupFixture,
        world_cup_2026_fixtures,
    )

LOGGER = logging.getLogger(__name__)

HOST_ADVANTAGE_MULTIPLIER = 1.10
REST_DAY_EDGE_MULTIPLIER = 0.03
MAX_REST_DAY_EDGE = 3.0
EXPECTED_TEAM_COUNT = 48
EXPECTED_MATCH_COUNT = 104
POISSON_SCORE_ENGINE = "poisson"
OUTCOME_HYBRID_SCORE_ENGINE = "outcome_hybrid"
DIXON_COLES_POISSON_SCORE_ENGINE = "poisson_dixon_coles"
DIXON_COLES_OUTCOME_HYBRID_SCORE_ENGINE = "outcome_hybrid_dixon_coles"
DEFAULT_DIXON_COLES_RHO = -0.10
MIN_DIXON_COLES_RHO = -0.95
MAX_DIXON_COLES_RHO = 0.95
MAX_DIXON_COLES_SCORE_ATTEMPTS = 64
MAX_CONDITIONAL_SCORE_ATTEMPTS = 256


@dataclass(frozen=True, slots=True)
class TeamLambda:
    """Predicted scoring intensity for one team."""

    team_id: str
    team_name: str
    lambda_goals: float
    country_code: str | None = None
    fair_play_penalty_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class OutcomeModelContext:
    """Loaded V/E/D model artifacts used by hybrid Monte Carlo scoring."""

    model: xgb.XGBClassifier
    team_features: dict[str, dict[str, float]]
    calibration_temperature: float | None = None


@dataclass(frozen=True, slots=True)
class SimulatedMatch:
    """Single simulated match result."""

    simulation_id: int
    round_name: str
    group_name: str | None
    match_number: int
    match_date: datetime
    stadium: str
    city: str
    country: str
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
    runner_up_team_id: str
    home_rest_days: float | None
    away_rest_days: float | None
    home_advantage: bool
    away_advantage: bool
    score_engine: str
    sampled_outcome: str | None
    outcome_home_win_pct: float | None
    outcome_draw_pct: float | None
    outcome_away_win_pct: float | None
    created_at: datetime


@dataclass(slots=True)
class GroupStanding:
    """Mutable group table row."""

    team: TeamLambda
    group_letter: str
    played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    fair_play_points: int = 0
    lot_draw_order: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True, slots=True)
class GroupMatchResult:
    """Completed group-stage match used for ranking tie-breakers."""

    group_letter: str
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int


def initialize_simulated_results(db_path: Path = DB_PATH) -> None:
    """Create the simulation results table if it does not exist."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS simulated_results (
                simulation_id INTEGER NOT NULL,
                round_name VARCHAR NOT NULL,
                group_name VARCHAR,
                match_number INTEGER NOT NULL,
                match_date TIMESTAMP,
                stadium VARCHAR,
                city VARCHAR,
                country VARCHAR,
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
                runner_up_team_id VARCHAR,
                home_rest_days DOUBLE,
                away_rest_days DOUBLE,
                home_advantage BOOLEAN,
                away_advantage BOOLEAN,
                score_engine VARCHAR,
                sampled_outcome VARCHAR,
                outcome_home_win_pct DOUBLE,
                outcome_draw_pct DOUBLE,
                outcome_away_win_pct DOUBLE,
                created_at TIMESTAMP NOT NULL
            );
            """,
        )
        _migrate_simulated_results(con)


def simulate_match(
    home_team: TeamLambda,
    away_team: TeamLambda,
    *,
    require_winner: bool,
    rng: np.random.Generator,
    venue_country: str | None = None,
    home_rest_days: float | None = None,
    away_rest_days: float | None = None,
    outcome_probabilities: tuple[float, float, float] | None = None,
    dixon_coles_rho: float = DEFAULT_DIXON_COLES_RHO,
) -> SimulatedMatch:
    """Simulate a single match using Dixon-Coles adjusted Poisson scoring."""
    _validate_dixon_coles_rho(dixon_coles_rho)
    home_advantage = _has_host_advantage(home_team, venue_country)
    away_advantage = _has_host_advantage(away_team, venue_country)
    home_lambda = _adjust_lambda(
        home_team.lambda_goals,
        has_host_advantage=home_advantage,
        own_rest_days=home_rest_days,
        opponent_rest_days=away_rest_days,
    )
    away_lambda = _adjust_lambda(
        away_team.lambda_goals,
        has_host_advantage=away_advantage,
        own_rest_days=away_rest_days,
        opponent_rest_days=home_rest_days,
    )

    score_engine = _poisson_score_engine(dixon_coles_rho)
    sampled_outcome: str | None = None
    outcome_home_win_pct: float | None = None
    outcome_draw_pct: float | None = None
    outcome_away_win_pct: float | None = None
    if outcome_probabilities is None:
        home_goals, away_goals = _sample_score(
            home_lambda,
            away_lambda,
            rng,
            dixon_coles_rho=dixon_coles_rho,
        )
    else:
        normalized_probabilities = _normalize_outcome_probabilities(outcome_probabilities)
        sampled_outcome = _sample_outcome(normalized_probabilities, rng)
        home_goals, away_goals = _sample_goals_for_outcome(
            home_lambda,
            away_lambda,
            sampled_outcome,
            rng,
            dixon_coles_rho=dixon_coles_rho,
        )
        score_engine = _outcome_hybrid_score_engine(dixon_coles_rho)
        outcome_home_win_pct = 100.0 * normalized_probabilities[0]
        outcome_draw_pct = 100.0 * normalized_probabilities[1]
        outcome_away_win_pct = 100.0 * normalized_probabilities[2]

    decided_by_penalties = False
    penalty_winner_team_id: str | None = None

    if home_goals > away_goals:
        winner_team_id = home_team.team_id
        runner_up_team_id = away_team.team_id
    elif away_goals > home_goals:
        winner_team_id = away_team.team_id
        runner_up_team_id = home_team.team_id
    elif require_winner:
        decided_by_penalties = True
        penalty_winner_team_id = _resolve_penalties(home_team.team_id, away_team.team_id, rng)
        winner_team_id = penalty_winner_team_id
        runner_up_team_id = (
            away_team.team_id if winner_team_id == home_team.team_id else home_team.team_id
        )
    else:
        winner_team_id = home_team.team_id
        runner_up_team_id = away_team.team_id

    return SimulatedMatch(
        simulation_id=-1,
        round_name="",
        group_name=None,
        match_number=-1,
        match_date=datetime.now(UTC),
        stadium="",
        city="",
        country=venue_country or "",
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
        runner_up_team_id=runner_up_team_id,
        home_rest_days=home_rest_days,
        away_rest_days=away_rest_days,
        home_advantage=home_advantage,
        away_advantage=away_advantage,
        score_engine=score_engine,
        sampled_outcome=sampled_outcome,
        outcome_home_win_pct=outcome_home_win_pct,
        outcome_draw_pct=outcome_draw_pct,
        outcome_away_win_pct=outcome_away_win_pct,
        created_at=datetime.now(UTC),
    )


def simulate_world_cup(
    teams: Sequence[TeamLambda],
    *,
    iterations: int = 100_000,
    batch_size: int = 2_500,
    db_path: Path = DB_PATH,
    seed: int | None = None,
    fixtures: Sequence[WorldCupFixture] | None = None,
    outcome_model_context: OutcomeModelContext | None = None,
    dixon_coles_rho: float = DEFAULT_DIXON_COLES_RHO,
) -> None:
    """Run a Monte Carlo simulation for the FIFA World Cup 2026 schedule."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    _validate_dixon_coles_rho(dixon_coles_rho)

    tournament_fixtures = tuple(fixtures or world_cup_2026_fixtures())
    _validate_tournament_shape(tournament_fixtures)
    teams_by_code = _teams_by_code(teams)
    _validate_schedule_coverage(teams_by_code, tournament_fixtures)

    initialize_simulated_results(db_path)
    rng = np.random.default_rng(seed)
    progress_interval = _simulation_progress_interval(iterations)
    LOGGER.info(
        "Starting %d Monte Carlo tournament simulations (%d matches per tournament).",
        iterations,
        len(tournament_fixtures),
    )
    LOGGER.info("Using Dixon-Coles rho %.3f for score sampling.", dixon_coles_rho)

    with duckdb.connect(str(db_path)) as con:
        con.execute("DELETE FROM simulated_results")

        buffer: list[tuple[object, ...]] = []
        for simulation_id in range(1, iterations + 1):
            tournament_rows = _simulate_tournament(
                teams_by_code,
                tournament_fixtures,
                simulation_id,
                rng,
                outcome_model_context=outcome_model_context,
                dixon_coles_rho=dixon_coles_rho,
            )
            buffer.extend(tournament_rows)
            if _should_log_simulation_progress(simulation_id, iterations, progress_interval):
                LOGGER.info(
                    "Simulation progress: tournament %d/%d completed (%.1f%% complete).",
                    simulation_id,
                    iterations,
                    100 * simulation_id / iterations,
                )

            if len(buffer) >= batch_size:
                _flush_rows(con, buffer)
                buffer.clear()

        if buffer:
            _flush_rows(con, buffer)

    LOGGER.info("Finished %d Monte Carlo simulations.", iterations)


def _simulation_progress_interval(iterations: int) -> int:
    return max(1, min(1_000, iterations // 100 or 1))


def _should_log_simulation_progress(
    simulation_id: int,
    iterations: int,
    progress_interval: int,
) -> bool:
    return (
        simulation_id == 1 or simulation_id == iterations or simulation_id % progress_interval == 0
    )


def _simulate_tournament(
    teams_by_code: dict[str, TeamLambda],
    fixtures: Sequence[WorldCupFixture],
    simulation_id: int,
    rng: np.random.Generator,
    *,
    outcome_model_context: OutcomeModelContext | None,
    dixon_coles_rho: float,
) -> list[tuple[object, ...]]:
    """Simulate one complete World Cup and return rows to persist."""
    rows: list[tuple[object, ...]] = []
    group_standings = _initial_group_standings(teams_by_code, fixtures, rng)
    group_match_results: list[GroupMatchResult] = []
    last_played_at: dict[str, datetime] = {}
    match_results: dict[int, SimulatedMatch] = {}

    for fixture in sorted(_group_fixtures(fixtures), key=lambda item: item.match_date):
        home_team = teams_by_code[fixture.home_slot]
        away_team = teams_by_code[fixture.away_slot]
        result = _play_fixture(
            fixture,
            home_team,
            away_team,
            simulation_id,
            rng,
            last_played_at,
            require_winner=False,
            outcome_model_context=outcome_model_context,
            dixon_coles_rho=dixon_coles_rho,
        )
        _apply_group_result(group_standings, fixture, result, rng)
        group_match_results.append(_group_match_result(fixture, result))
        match_results[fixture.match_number] = result
        rows.append(_row_tuple(result))

    slot_teams = _qualified_slot_teams(group_standings, group_match_results, fixtures)

    for fixture in sorted(_knockout_fixtures(fixtures), key=lambda item: item.match_date):
        home_team = _resolve_slot(fixture.home_slot, teams_by_code, slot_teams, match_results)
        away_team = _resolve_slot(fixture.away_slot, teams_by_code, slot_teams, match_results)
        result = _play_fixture(
            fixture,
            home_team,
            away_team,
            simulation_id,
            rng,
            last_played_at,
            require_winner=True,
            outcome_model_context=outcome_model_context,
            dixon_coles_rho=dixon_coles_rho,
        )
        match_results[fixture.match_number] = result
        rows.append(_row_tuple(result))

    return rows


def _play_fixture(
    fixture: WorldCupFixture,
    home_team: TeamLambda,
    away_team: TeamLambda,
    simulation_id: int,
    rng: np.random.Generator,
    last_played_at: dict[str, datetime],
    *,
    require_winner: bool,
    outcome_model_context: OutcomeModelContext | None,
    dixon_coles_rho: float,
) -> SimulatedMatch:
    home_rest_days = _rest_days(last_played_at.get(home_team.team_id), fixture.match_date)
    away_rest_days = _rest_days(last_played_at.get(away_team.team_id), fixture.match_date)
    result = simulate_match(
        home_team,
        away_team,
        require_winner=require_winner,
        rng=rng,
        venue_country=fixture.country,
        home_rest_days=home_rest_days,
        away_rest_days=away_rest_days,
        outcome_probabilities=_match_outcome_probabilities(
            outcome_model_context,
            home_team.team_id,
            away_team.team_id,
        ),
        dixon_coles_rho=dixon_coles_rho,
    )
    result = _with_fixture_metadata(result, fixture, simulation_id)
    last_played_at[home_team.team_id] = fixture.match_date
    last_played_at[away_team.team_id] = fixture.match_date
    return result


def _with_fixture_metadata(
    result: SimulatedMatch,
    fixture: WorldCupFixture,
    simulation_id: int,
) -> SimulatedMatch:
    return SimulatedMatch(
        simulation_id=simulation_id,
        round_name=fixture.round_name,
        group_name=fixture.group_name,
        match_number=fixture.match_number,
        match_date=fixture.match_date,
        stadium=fixture.stadium,
        city=fixture.city,
        country=fixture.country,
        home_team_id=result.home_team_id,
        away_team_id=result.away_team_id,
        home_team_name=result.home_team_name,
        away_team_name=result.away_team_name,
        home_lambda=result.home_lambda,
        away_lambda=result.away_lambda,
        home_goals=result.home_goals,
        away_goals=result.away_goals,
        require_winner=result.require_winner,
        decided_by_penalties=result.decided_by_penalties,
        penalty_winner_team_id=result.penalty_winner_team_id,
        winner_team_id=result.winner_team_id,
        runner_up_team_id=result.runner_up_team_id,
        home_rest_days=result.home_rest_days,
        away_rest_days=result.away_rest_days,
        home_advantage=result.home_advantage,
        away_advantage=result.away_advantage,
        score_engine=result.score_engine,
        sampled_outcome=result.sampled_outcome,
        outcome_home_win_pct=result.outcome_home_win_pct,
        outcome_draw_pct=result.outcome_draw_pct,
        outcome_away_win_pct=result.outcome_away_win_pct,
        created_at=result.created_at,
    )


def _initial_group_standings(
    teams_by_code: dict[str, TeamLambda],
    fixtures: Sequence[WorldCupFixture],
    rng: np.random.Generator,
) -> dict[str, dict[str, GroupStanding]]:
    standings: dict[str, dict[str, GroupStanding]] = defaultdict(dict)
    for fixture in _group_fixtures(fixtures):
        group_letter = _group_letter(fixture.group_name)
        for code in (fixture.home_slot, fixture.away_slot):
            if code not in standings[group_letter]:
                standings[group_letter][code] = GroupStanding(
                    team=teams_by_code[code],
                    group_letter=group_letter,
                )
    for group_rows in standings.values():
        lot_order = rng.permutation(len(group_rows))
        for order, code in zip(lot_order, sorted(group_rows), strict=True):
            group_rows[code].lot_draw_order = int(order)
    return standings


def _apply_group_result(
    standings: dict[str, dict[str, GroupStanding]],
    fixture: WorldCupFixture,
    result: SimulatedMatch,
    rng: np.random.Generator,
) -> None:
    group_letter = _group_letter(fixture.group_name)
    home = standings[group_letter][result.home_team_id]
    away = standings[group_letter][result.away_team_id]

    home.played += 1
    away.played += 1
    home.goals_for += result.home_goals
    home.goals_against += result.away_goals
    away.goals_for += result.away_goals
    away.goals_against += result.home_goals
    home.fair_play_points += _simulate_fair_play_penalty(home.team, rng)
    away.fair_play_points += _simulate_fair_play_penalty(away.team, rng)

    if result.home_goals > result.away_goals:
        home.points += 3
    elif result.away_goals > result.home_goals:
        away.points += 3
    else:
        home.points += 1
        away.points += 1


def _simulate_fair_play_penalty(team: TeamLambda, rng: np.random.Generator) -> int:
    """Sample positive fair-play penalty points from prior World Cup discipline."""
    return int(rng.poisson(max(float(team.fair_play_penalty_rate), 0.0)))


def _group_match_result(
    fixture: WorldCupFixture,
    result: SimulatedMatch,
) -> GroupMatchResult:
    return GroupMatchResult(
        group_letter=_group_letter(fixture.group_name),
        home_team_id=result.home_team_id,
        away_team_id=result.away_team_id,
        home_goals=result.home_goals,
        away_goals=result.away_goals,
    )


def _qualified_slot_teams(
    standings: dict[str, dict[str, GroupStanding]],
    group_match_results: Sequence[GroupMatchResult],
    fixtures: Sequence[WorldCupFixture],
) -> dict[str, TeamLambda]:
    slot_teams: dict[str, TeamLambda] = {}
    third_place_rows: list[GroupStanding] = []

    for group_letter, group_rows in sorted(standings.items()):
        ranked = _rank_group(group_rows.values(), group_match_results)
        slot_teams[f"1{group_letter}"] = ranked[0].team
        slot_teams[f"2{group_letter}"] = ranked[1].team
        third_place_rows.append(ranked[2])

    third_place_rows = _rank_third_place_teams(third_place_rows)[:8]
    slot_teams.update(_assign_third_place_slots(third_place_rows, fixtures))
    return slot_teams


def _assign_third_place_slots(
    third_place_rows: Sequence[GroupStanding],
    fixtures: Sequence[WorldCupFixture],
) -> dict[str, TeamLambda]:
    specs = [
        slot
        for fixture in sorted(_round_of_32_fixtures(fixtures), key=lambda item: item.match_number)
        for slot in (fixture.home_slot, fixture.away_slot)
        if _is_third_place_slot(slot)
    ]
    rank_by_group = {row.group_letter: index for index, row in enumerate(third_place_rows)}
    team_by_group = {row.group_letter: row.team for row in third_place_rows}
    assignment = _third_place_slot_assignment(specs, tuple(rank_by_group))
    return {spec: team_by_group[group_letter] for spec, group_letter in assignment.items()}


def _third_place_slot_assignment(
    specs: Sequence[str],
    available_groups: tuple[str, ...],
) -> dict[str, str]:
    groups_by_spec = {spec: tuple(spec[1:]) for spec in specs}

    def can_complete(index: int, used_groups: frozenset[str]) -> bool:
        remaining_specs = specs[index:]
        remaining_groups = tuple(group for group in available_groups if group not in used_groups)
        if len(remaining_groups) < len(remaining_specs):
            return False
        possible_specs = {
            spec
            for spec in remaining_specs
            if any(group in groups_by_spec[spec] for group in remaining_groups)
        }
        return len(possible_specs) == len(remaining_specs)

    def assign(index: int, used_groups: frozenset[str]) -> dict[str, str] | None:
        if index == len(specs):
            return {}
        spec = specs[index]
        candidates = [
            group
            for group in available_groups
            if group not in used_groups and group in groups_by_spec[spec]
        ]
        for group in candidates:
            next_used = used_groups | {group}
            if not can_complete(index + 1, next_used):
                continue
            tail = assign(index + 1, next_used)
            if tail is not None:
                return {spec: group, **tail}
        return None

    assignment = assign(0, frozenset())
    if assignment is None:
        raise RuntimeError("Could not assign third-place teams to Round of 32 slots.")
    return assignment


def _rank_group(
    rows: Iterable[GroupStanding],
    group_match_results: Sequence[GroupMatchResult],
) -> list[GroupStanding]:
    """Rank a group with FIFA-style table and head-to-head tie-breakers."""
    standings = list(rows)
    matches = [
        match
        for match in group_match_results
        if standings and match.group_letter == standings[0].group_letter
    ]

    return sorted(
        standings,
        key=lambda row: (
            -row.points,
            -row.goal_difference,
            -row.goals_for,
            _head_to_head_sort_key(row, standings, matches),
            row.fair_play_points,
            row.lot_draw_order,
        ),
    )


def _rank_third_place_teams(rows: Iterable[GroupStanding]) -> list[GroupStanding]:
    return sorted(
        rows,
        key=lambda row: (
            -row.points,
            -row.goal_difference,
            -row.goals_for,
            row.fair_play_points,
            row.lot_draw_order,
            row.group_letter,
        ),
    )


def _head_to_head_sort_key(
    row: GroupStanding,
    all_rows: Sequence[GroupStanding],
    group_match_results: Sequence[GroupMatchResult],
) -> tuple[int, int, int]:
    tied_team_ids = [
        candidate.team.team_id
        for candidate in all_rows
        if (
            candidate.points,
            candidate.goal_difference,
            candidate.goals_for,
        )
        == (row.points, row.goal_difference, row.goals_for)
    ]
    if len(tied_team_ids) < 2:
        return (0, 0, 0)

    tied_ids = set(tied_team_ids)
    points = 0
    goals_for = 0
    goals_against = 0

    for match in group_match_results:
        if match.home_team_id not in tied_ids or match.away_team_id not in tied_ids:
            continue
        if row.team.team_id == match.home_team_id:
            goals_for += match.home_goals
            goals_against += match.away_goals
            if match.home_goals > match.away_goals:
                points += 3
            elif match.home_goals == match.away_goals:
                points += 1
        elif row.team.team_id == match.away_team_id:
            goals_for += match.away_goals
            goals_against += match.home_goals
            if match.away_goals > match.home_goals:
                points += 3
            elif match.away_goals == match.home_goals:
                points += 1

    return (-points, -(goals_for - goals_against), -goals_for)


def _resolve_slot(
    slot: str,
    teams_by_code: dict[str, TeamLambda],
    qualified_slots: dict[str, TeamLambda],
    match_results: dict[int, SimulatedMatch],
) -> TeamLambda:
    if slot in teams_by_code:
        return teams_by_code[slot]
    if slot in qualified_slots:
        return qualified_slots[slot]

    winner_match = re.fullmatch(r"W(\d+)", slot)
    if winner_match is not None:
        match = match_results[int(winner_match.group(1))]
        return _team_from_result(match.winner_team_id, match, teams_by_code)

    runner_up_match = re.fullmatch(r"RU(\d+)", slot)
    if runner_up_match is not None:
        match = match_results[int(runner_up_match.group(1))]
        return _team_from_result(match.runner_up_team_id, match, teams_by_code)

    raise KeyError(f"Unresolved tournament slot: {slot}")


def _team_from_result(
    team_id: str,
    result: SimulatedMatch,
    teams_by_code: dict[str, TeamLambda],
) -> TeamLambda:
    if team_id in teams_by_code:
        return teams_by_code[team_id]
    if team_id == result.home_team_id:
        return TeamLambda(result.home_team_id, result.home_team_name, result.home_lambda)
    return TeamLambda(result.away_team_id, result.away_team_name, result.away_lambda)


def _row_tuple(result: SimulatedMatch) -> tuple[object, ...]:
    return (
        result.simulation_id,
        result.round_name,
        result.group_name,
        result.match_number,
        result.match_date,
        result.stadium,
        result.city,
        result.country,
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
        result.runner_up_team_id,
        result.home_rest_days,
        result.away_rest_days,
        result.home_advantage,
        result.away_advantage,
        result.score_engine,
        result.sampled_outcome,
        result.outcome_home_win_pct,
        result.outcome_draw_pct,
        result.outcome_away_win_pct,
        result.created_at,
    )


def _flush_rows(con: duckdb.DuckDBPyConnection, rows: list[tuple[object, ...]]) -> None:
    """Persist a batch of rows into DuckDB."""
    con.executemany(
        """
        INSERT INTO simulated_results (
            simulation_id,
            round_name,
            group_name,
            match_number,
            match_date,
            stadium,
            city,
            country,
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
            runner_up_team_id,
            home_rest_days,
            away_rest_days,
            home_advantage,
            away_advantage,
            score_engine,
            sampled_outcome,
            outcome_home_win_pct,
            outcome_draw_pct,
            outcome_away_win_pct,
            created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )


def _migrate_simulated_results(con: duckdb.DuckDBPyConnection) -> None:
    migration_statements = (
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS group_name VARCHAR",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS match_date TIMESTAMP",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS stadium VARCHAR",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS city VARCHAR",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS country VARCHAR",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS runner_up_team_id VARCHAR",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS home_rest_days DOUBLE",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS away_rest_days DOUBLE",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS home_advantage BOOLEAN",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS away_advantage BOOLEAN",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS score_engine VARCHAR",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS sampled_outcome VARCHAR",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS outcome_home_win_pct DOUBLE",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS outcome_draw_pct DOUBLE",
        "ALTER TABLE simulated_results ADD COLUMN IF NOT EXISTS outcome_away_win_pct DOUBLE",
    )
    for statement in migration_statements:
        con.execute(statement)


def _teams_by_code(teams: Sequence[TeamLambda]) -> dict[str, TeamLambda]:
    by_code: dict[str, TeamLambda] = {}
    lookup = {_normalize_key(team.team_id): team for team in teams}
    lookup.update({_normalize_key(team.team_name): team for team in teams})

    for code, official_name in TEAM_NAMES.items():
        team = next(
            (lookup[key] for key in _team_lookup_keys(code, official_name) if key in lookup),
            None,
        )
        if team is None:
            continue
        by_code[code] = TeamLambda(
            team_id=code,
            team_name=official_name,
            lambda_goals=team.lambda_goals,
            country_code=team.country_code or TEAM_COUNTRIES.get(code, code),
            fair_play_penalty_rate=team.fair_play_penalty_rate,
        )
    return by_code


def _team_lookup_keys(code: str, official_name: str) -> tuple[str, ...]:
    aliases = {
        "BIH": ("Bosnia", "Bosnia-Herzegovina", "Bosnia and Herzegovina"),
        "CIV": ("Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire"),
        "COD": ("DR Congo", "Congo DR", "Congo Democratic Republic"),
        "CPV": ("Cape Verde", "Cabo Verde"),
        "CZE": ("Czech Republic", "Czechia"),
        "IRN": ("Iran", "IR Iran"),
        "KOR": ("South Korea", "Korea Republic"),
        "RSA": ("South Africa",),
        "SUI": ("Switzerland", "Swiss"),
        "TUR": ("Turkey", "Türkiye"),
        "USA": ("United States", "USA", "USMNT"),
    }
    names = (code, official_name, *aliases.get(code, ()))
    return tuple(_normalize_key(name) for name in names)


def _validate_schedule_coverage(
    teams_by_code: dict[str, TeamLambda],
    fixtures: Sequence[WorldCupFixture],
) -> None:
    fixture_team_codes = {
        slot
        for fixture in _group_fixtures(fixtures)
        for slot in (fixture.home_slot, fixture.away_slot)
    }
    missing_codes = sorted(fixture_team_codes - set(teams_by_code))
    if missing_codes:
        missing_names = ", ".join(
            f"{code} ({TEAM_NAMES.get(code, code)})" for code in missing_codes
        )
        raise ValueError(f"teams is missing World Cup 2026 teams: {missing_names}")
    if len(fixture_team_codes) != EXPECTED_TEAM_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TEAM_COUNT} World Cup teams, got {len(fixture_team_codes)}."
        )


def _validate_tournament_shape(fixtures: Sequence[WorldCupFixture]) -> None:
    if len(fixtures) != EXPECTED_MATCH_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MATCH_COUNT} World Cup fixtures, got {len(fixtures)}."
        )

    match_numbers = sorted(fixture.match_number for fixture in fixtures)
    expected_match_numbers = list(range(1, EXPECTED_MATCH_COUNT + 1))
    if match_numbers != expected_match_numbers:
        raise ValueError("World Cup fixtures must have match numbers 1 through 104 exactly once.")

    round_counts = {
        "group_stage": 72,
        "round_of_32": 16,
        "round_of_16": 8,
        "quarterfinal": 4,
        "semifinal": 2,
        "third_place": 1,
        "final": 1,
    }
    for round_name, expected_count in round_counts.items():
        actual_count = sum(fixture.round_name == round_name for fixture in fixtures)
        if actual_count != expected_count:
            raise ValueError(
                f"expected {expected_count} {round_name} fixtures, got {actual_count}."
            )

    groups: dict[str, list[WorldCupFixture]] = defaultdict(list)
    for fixture in _group_fixtures(fixtures):
        groups[_group_letter(fixture.group_name)].append(fixture)

    if sorted(groups) != list("ABCDEFGHIJKL"):
        raise ValueError("World Cup group fixtures must contain groups A through L.")

    for group_letter, group_fixtures in groups.items():
        group_team_codes = {
            slot for fixture in group_fixtures for slot in (fixture.home_slot, fixture.away_slot)
        }
        if len(group_team_codes) != 4:
            raise ValueError(f"Group {group_letter} must contain exactly four teams.")
        pairings = {
            tuple(sorted((fixture.home_slot, fixture.away_slot))) for fixture in group_fixtures
        }
        if len(group_fixtures) != 6 or len(pairings) != 6:
            raise ValueError(f"Group {group_letter} must contain six unique pairings.")


def _group_fixtures(fixtures: Sequence[WorldCupFixture]) -> Iterable[WorldCupFixture]:
    return (fixture for fixture in fixtures if fixture.round_name == "group_stage")


def _knockout_fixtures(fixtures: Sequence[WorldCupFixture]) -> Iterable[WorldCupFixture]:
    return (fixture for fixture in fixtures if fixture.round_name != "group_stage")


def _round_of_32_fixtures(fixtures: Sequence[WorldCupFixture]) -> Iterable[WorldCupFixture]:
    return (fixture for fixture in fixtures if fixture.round_name == "round_of_32")


def _is_third_place_slot(slot: str) -> bool:
    return bool(re.fullmatch(r"3[A-L]+", slot))


def _group_letter(group_name: str | None) -> str:
    if not group_name:
        raise ValueError("Group fixture is missing group_name.")
    return group_name.rsplit(" ", maxsplit=1)[-1]


def _resolve_penalties(
    home_team_id: str,
    away_team_id: str,
    rng: np.random.Generator,
) -> str:
    """Resolve a tie with a simple 50/50 penalty shootout."""
    return home_team_id if int(rng.integers(0, 2)) == 0 else away_team_id


def _match_outcome_probabilities(
    outcome_model_context: OutcomeModelContext | None,
    home_team_id: str,
    away_team_id: str,
) -> tuple[float, float, float] | None:
    if outcome_model_context is None:
        return None
    try:
        probabilities = predict_match_probabilities(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            team_features=outcome_model_context.team_features,
            model=outcome_model_context.model,
            calibration_temperature=outcome_model_context.calibration_temperature,
        )
    except KeyError:
        return None
    return tuple(float(probability) for probability in probabilities)


def _normalize_outcome_probabilities(
    probabilities: tuple[float, float, float],
) -> tuple[float, float, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    total = float(clipped.sum())
    if total <= 0.0:
        return (1 / 3, 1 / 3, 1 / 3)
    normalized = clipped / total
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


def _sample_outcome(
    probabilities: tuple[float, float, float],
    rng: np.random.Generator,
) -> str:
    index = int(rng.choice(3, p=np.asarray(probabilities, dtype=float)))
    return ("home", "draw", "away")[index]


def _sample_goals_for_outcome(
    home_lambda: float,
    away_lambda: float,
    sampled_outcome: str,
    rng: np.random.Generator,
    *,
    dixon_coles_rho: float,
) -> tuple[int, int]:
    for _ in range(MAX_CONDITIONAL_SCORE_ATTEMPTS):
        home_goals, away_goals = _sample_score(
            home_lambda,
            away_lambda,
            rng,
            dixon_coles_rho=dixon_coles_rho,
        )
        if _score_matches_outcome(home_goals, away_goals, sampled_outcome):
            return home_goals, away_goals
    return _fallback_score_for_outcome(home_lambda, away_lambda, sampled_outcome, rng)


def _score_matches_outcome(home_goals: int, away_goals: int, sampled_outcome: str) -> bool:
    if sampled_outcome == "home":
        return home_goals > away_goals
    if sampled_outcome == "away":
        return away_goals > home_goals
    return home_goals == away_goals


def _fallback_score_for_outcome(
    home_lambda: float,
    away_lambda: float,
    sampled_outcome: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if sampled_outcome == "draw":
        draw_lambda = max((home_lambda + away_lambda) / 2.0, 0.0)
        goals = int(rng.poisson(draw_lambda))
        return goals, goals

    if sampled_outcome == "home":
        away_goals = int(rng.poisson(max(away_lambda, 0.0)))
        margin_lambda = max(home_lambda - away_lambda, 0.35)
        return away_goals + 1 + int(rng.poisson(margin_lambda)), away_goals

    home_goals = int(rng.poisson(max(home_lambda, 0.0)))
    margin_lambda = max(away_lambda - home_lambda, 0.35)
    return home_goals, home_goals + 1 + int(rng.poisson(margin_lambda))


def _sample_score(
    home_lambda: float,
    away_lambda: float,
    rng: np.random.Generator,
    *,
    dixon_coles_rho: float,
) -> tuple[int, int]:
    if math.isclose(dixon_coles_rho, 0.0, abs_tol=1e-12):
        return (
            int(rng.poisson(max(home_lambda, 0.0))),
            int(rng.poisson(max(away_lambda, 0.0))),
        )
    return _sample_dixon_coles_score(home_lambda, away_lambda, rng, rho=dixon_coles_rho)


def _sample_dixon_coles_score(
    home_lambda: float,
    away_lambda: float,
    rng: np.random.Generator,
    *,
    rho: float,
) -> tuple[int, int]:
    home_rate = max(float(home_lambda), 0.0)
    away_rate = max(float(away_lambda), 0.0)
    tau_upper_bound = _dixon_coles_tau_upper_bound(home_rate, away_rate, rho)

    for _ in range(MAX_DIXON_COLES_SCORE_ATTEMPTS):
        home_goals = int(rng.poisson(home_rate))
        away_goals = int(rng.poisson(away_rate))
        tau = _dixon_coles_tau(home_goals, away_goals, home_rate, away_rate, rho)
        if tau > 0.0 and float(rng.random()) <= tau / tau_upper_bound:
            return home_goals, away_goals

    return int(rng.poisson(home_rate)), int(rng.poisson(away_rate))


def _dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_lambda: float,
    away_lambda: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return max(0.0, 1.0 - home_lambda * away_lambda * rho)
    if home_goals == 0 and away_goals == 1:
        return max(0.0, 1.0 + home_lambda * rho)
    if home_goals == 1 and away_goals == 0:
        return max(0.0, 1.0 + away_lambda * rho)
    if home_goals == 1 and away_goals == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def _dixon_coles_tau_upper_bound(home_lambda: float, away_lambda: float, rho: float) -> float:
    candidates = [
        1.0,
        _dixon_coles_tau(0, 0, home_lambda, away_lambda, rho),
    ]
    if home_lambda > 0.0:
        candidates.append(_dixon_coles_tau(1, 0, home_lambda, away_lambda, rho))
    if away_lambda > 0.0:
        candidates.append(_dixon_coles_tau(0, 1, home_lambda, away_lambda, rho))
    if home_lambda > 0.0 and away_lambda > 0.0:
        candidates.append(_dixon_coles_tau(1, 1, home_lambda, away_lambda, rho))
    return max(candidates)


def _validate_dixon_coles_rho(rho: float) -> None:
    if not MIN_DIXON_COLES_RHO <= rho <= MAX_DIXON_COLES_RHO:
        raise ValueError(
            f"dixon_coles_rho must be between {MIN_DIXON_COLES_RHO} and {MAX_DIXON_COLES_RHO}."
        )


def _poisson_score_engine(dixon_coles_rho: float) -> str:
    if math.isclose(dixon_coles_rho, 0.0, abs_tol=1e-12):
        return POISSON_SCORE_ENGINE
    return DIXON_COLES_POISSON_SCORE_ENGINE


def _outcome_hybrid_score_engine(dixon_coles_rho: float) -> str:
    if math.isclose(dixon_coles_rho, 0.0, abs_tol=1e-12):
        return OUTCOME_HYBRID_SCORE_ENGINE
    return DIXON_COLES_OUTCOME_HYBRID_SCORE_ENGINE


def _adjust_lambda(
    lambda_goals: float,
    *,
    has_host_advantage: bool,
    own_rest_days: float | None,
    opponent_rest_days: float | None,
) -> float:
    adjusted = max(float(lambda_goals), 0.0)
    if has_host_advantage:
        adjusted *= HOST_ADVANTAGE_MULTIPLIER
    if own_rest_days is not None and opponent_rest_days is not None:
        rest_edge = max(
            -MAX_REST_DAY_EDGE,
            min(MAX_REST_DAY_EDGE, own_rest_days - opponent_rest_days),
        )
        adjusted *= max(0.75, 1.0 + rest_edge * REST_DAY_EDGE_MULTIPLIER)
    return adjusted


def _has_host_advantage(team: TeamLambda, venue_country: str | None) -> bool:
    return bool(venue_country and team.country_code == venue_country)


def _rest_days(previous_match: datetime | None, match_date: datetime) -> float | None:
    if previous_match is None:
        return None
    return (match_date - previous_match).total_seconds() / 86_400


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone tournament simulations."""
    parser = argparse.ArgumentParser(
        description="Run FIFA World Cup 2026 Monte Carlo simulations from project data.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of complete tournament simulations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used when inserting rows into DuckDB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Optional random seed for reproducible simulations.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="DuckDB warehouse path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Trained XGBoost Poisson model path.",
    )
    parser.add_argument(
        "--export-analytics",
        action="store_true",
        help="Export analytics CSVs after the simulation finishes.",
    )
    parser.add_argument(
        "--disable-outcome-model",
        action="store_true",
        help="Use the original Poisson-only score simulation even if the V/E/D model exists.",
    )
    parser.add_argument(
        "--dixon-coles-rho",
        type=float,
        default=DEFAULT_DIXON_COLES_RHO,
        help=(
            "Low-score Dixon-Coles dependence parameter. Use 0.0 for independent "
            "Poisson score sampling."
        ),
    )
    return parser


def run_simulation_from_project_data(
    *,
    db_path: Path = DB_PATH,
    model_path: Path = MODEL_PATH,
    iterations: int = DEFAULT_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = DEFAULT_SEED,
    export_reports: bool = False,
    use_outcome_model: bool = True,
    dixon_coles_rho: float = DEFAULT_DIXON_COLES_RHO,
) -> None:
    """Build team lambdas from the local warehouse/model and simulate the tournament."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB warehouse not found: {db_path}. "
            "Run `uv run pipeline` or `uv run db-init` first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run `uv run train-model` before `uv run simulate`."
        )

    model = _load_poisson_model(model_path)
    team_lambdas = _build_world_cup_team_lambdas_from_project_data(db_path=db_path, model=model)
    outcome_model_context = (
        load_outcome_model_context(db_path=db_path) if use_outcome_model else None
    )
    simulate_world_cup(
        team_lambdas,
        iterations=iterations,
        batch_size=batch_size,
        db_path=db_path,
        seed=seed,
        outcome_model_context=outcome_model_context,
        dixon_coles_rho=dixon_coles_rho,
    )
    if export_reports:
        export_analytics(db_path=db_path)


def _load_poisson_model(model_path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model


def load_outcome_model_context(
    *,
    db_path: Path,
    model_path: Path = OUTCOME_MODEL_PATH,
    calibration_path: Path = OUTCOME_CALIBRATION_PATH,
) -> OutcomeModelContext | None:
    if not model_path.exists():
        LOGGER.info("Outcome model not found at %s; using Poisson-only simulation.", model_path)
        return None

    outcome_model = load_outcome_model_artifact(model_path)
    team_features = build_current_world_cup_team_features(db_path)
    calibration_temperature = load_calibration_temperature(calibration_path)
    LOGGER.info(
        "Using calibrated V/E/D outcome model for hybrid score simulation (%d teams).",
        len(team_features),
    )
    return OutcomeModelContext(
        model=outcome_model,
        team_features=team_features,
        calibration_temperature=calibration_temperature,
    )


def _build_world_cup_team_lambdas_from_project_data(
    *,
    db_path: Path,
    model: xgb.XGBRegressor,
) -> list[TeamLambda]:
    try:
        from .orchestrator import _build_team_lambdas
    except ImportError:  # pragma: no cover - supports direct script execution.
        from orchestrator import _build_team_lambdas  # type: ignore[no-redef]

    return _build_team_lambdas(db_path=db_path, model=model)


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    run_simulation_from_project_data(
        db_path=args.db_path,
        model_path=args.model_path,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
        export_reports=args.export_analytics,
        use_outcome_model=not args.disable_outcome_model,
        dixon_coles_rho=args.dixon_coles_rho,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
