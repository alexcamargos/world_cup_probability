# ruff: noqa: E501
"""Official FIFA World Cup 2026 match schedule used by the simulator.

The rows were normalized from the public FIFA calendar endpoint on
2026-06-21:
https://api.fifa.com/api/v3/calendar/matches?from=2026-06-11&to=2026-07-20&language=en&count=200&idCompetition=17&idSeason=285023
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class WorldCupFixture:
    """One scheduled FIFA World Cup 2026 match."""

    match_number: int
    match_date: datetime
    round_name: str
    group_name: str | None
    home_slot: str
    away_slot: str
    stadium: str
    city: str
    country: str
    played_home_goals: int | None = None
    played_away_goals: int | None = None


_FIXTURE_ROWS: tuple[tuple[object, ...], ...] = (
    (1, "2026-06-11T19:00:00Z", "group_stage", "Group A", "MEX", "RSA", "Mexico City Stadium", "Mexico City", 2, 0),
    (2, "2026-06-12T02:00:00Z", "group_stage", "Group A", "KOR", "CZE", "Guadalajara Stadium", "Guadalajara", 2, 1),
    (3, "2026-06-12T19:00:00Z", "group_stage", "Group B", "CAN", "BIH", "Toronto Stadium", "Toronto", 1, 1),
    (4, "2026-06-13T01:00:00Z", "group_stage", "Group D", "USA", "PAR", "Los Angeles Stadium", "Los Angeles", 4, 1),
    (5, "2026-06-14T01:00:00Z", "group_stage", "Group C", "HAI", "SCO", "Boston Stadium", "Boston", 0, 1),
    (6, "2026-06-14T04:00:00Z", "group_stage", "Group D", "AUS", "TUR", "BC Place Vancouver", "Vancouver", 2, 0),
    (7, "2026-06-13T22:00:00Z", "group_stage", "Group C", "BRA", "MAR", "New York/New Jersey Stadium", "New Jersey", 1, 1),
    (8, "2026-06-13T19:00:00Z", "group_stage", "Group B", "QAT", "SUI", "San Francisco Bay Area Stadium", "San Francisco Bay Area", 1, 1),
    (9, "2026-06-14T23:00:00Z", "group_stage", "Group E", "CIV", "ECU", "Philadelphia Stadium", "Philadelphia", 1, 0),
    (10, "2026-06-14T17:00:00Z", "group_stage", "Group E", "GER", "CUW", "Houston Stadium", "Houston", 7, 1),
    (11, "2026-06-14T20:00:00Z", "group_stage", "Group F", "NED", "JPN", "Dallas Stadium", "Dallas", 2, 2),
    (12, "2026-06-15T02:00:00Z", "group_stage", "Group F", "SWE", "TUN", "Monterrey Stadium", "Monterrey", 5, 1),
    (13, "2026-06-15T22:00:00Z", "group_stage", "Group H", "KSA", "URU", "Miami Stadium", "Miami", 1, 1),
    (14, "2026-06-15T16:00:00Z", "group_stage", "Group H", "ESP", "CPV", "Atlanta Stadium", "Atlanta", 0, 0),
    (15, "2026-06-16T01:00:00Z", "group_stage", "Group G", "IRN", "NZL", "Los Angeles Stadium", "Los Angeles", 2, 2),
    (16, "2026-06-15T19:00:00Z", "group_stage", "Group G", "BEL", "EGY", "Seattle Stadium", "Seattle", 1, 1),
    (17, "2026-06-16T19:00:00Z", "group_stage", "Group I", "FRA", "SEN", "New York/New Jersey Stadium", "New Jersey", 3, 1),
    (18, "2026-06-16T22:00:00Z", "group_stage", "Group I", "IRQ", "NOR", "Boston Stadium", "Boston", 1, 4),
    (19, "2026-06-17T01:00:00Z", "group_stage", "Group J", "ARG", "ALG", "Kansas City Stadium", "Kansas City", 3, 0),
    (20, "2026-06-17T04:00:00Z", "group_stage", "Group J", "AUT", "JOR", "San Francisco Bay Area Stadium", "San Francisco Bay Area", 3, 1),
    (21, "2026-06-17T23:00:00Z", "group_stage", "Group L", "GHA", "PAN", "Toronto Stadium", "Toronto", 1, 0),
    (22, "2026-06-17T20:00:00Z", "group_stage", "Group L", "ENG", "CRO", "Dallas Stadium", "Dallas", 4, 2),
    (23, "2026-06-17T17:00:00Z", "group_stage", "Group K", "POR", "COD", "Houston Stadium", "Houston", 1, 1),
    (24, "2026-06-18T02:00:00Z", "group_stage", "Group K", "UZB", "COL", "Mexico City Stadium", "Mexico City", 1, 3),
    (25, "2026-06-18T16:00:00Z", "group_stage", "Group A", "CZE", "RSA", "Atlanta Stadium", "Atlanta", 1, 1),
    (26, "2026-06-18T19:00:00Z", "group_stage", "Group B", "SUI", "BIH", "Los Angeles Stadium", "Los Angeles", 4, 1),
    (27, "2026-06-18T22:00:00Z", "group_stage", "Group B", "CAN", "QAT", "BC Place Vancouver", "Vancouver", 6, 0),
    (28, "2026-06-19T01:00:00Z", "group_stage", "Group A", "MEX", "KOR", "Guadalajara Stadium", "Guadalajara", 1, 0),
    (29, "2026-06-20T00:30:00Z", "group_stage", "Group C", "BRA", "HAI", "Philadelphia Stadium", "Philadelphia", 3, 0),
    (30, "2026-06-19T22:00:00Z", "group_stage", "Group C", "SCO", "MAR", "Boston Stadium", "Boston", 0, 1),
    (31, "2026-06-20T03:00:00Z", "group_stage", "Group D", "TUR", "PAR", "San Francisco Bay Area Stadium", "San Francisco Bay Area", 0, 1),
    (32, "2026-06-19T19:00:00Z", "group_stage", "Group D", "USA", "AUS", "Seattle Stadium", "Seattle", 2, 0),
    (33, "2026-06-20T20:00:00Z", "group_stage", "Group E", "GER", "CIV", "Toronto Stadium", "Toronto", 2, 1),
    (34, "2026-06-21T00:00:00Z", "group_stage", "Group E", "ECU", "CUW", "Kansas City Stadium", "Kansas City", 0, 0),
    (35, "2026-06-20T17:00:00Z", "group_stage", "Group F", "NED", "SWE", "Houston Stadium", "Houston", 5, 1),
    (36, "2026-06-21T04:00:00Z", "group_stage", "Group F", "TUN", "JPN", "Monterrey Stadium", "Monterrey", None, None),
    (37, "2026-06-21T22:00:00Z", "group_stage", "Group H", "URU", "CPV", "Miami Stadium", "Miami", None, None),
    (38, "2026-06-21T16:00:00Z", "group_stage", "Group H", "ESP", "KSA", "Atlanta Stadium", "Atlanta", None, None),
    (39, "2026-06-21T19:00:00Z", "group_stage", "Group G", "BEL", "IRN", "Los Angeles Stadium", "Los Angeles", None, None),
    (40, "2026-06-22T01:00:00Z", "group_stage", "Group G", "NZL", "EGY", "BC Place Vancouver", "Vancouver", None, None),
    (41, "2026-06-23T00:00:00Z", "group_stage", "Group I", "NOR", "SEN", "New York/New Jersey Stadium", "New Jersey", None, None),
    (42, "2026-06-22T21:00:00Z", "group_stage", "Group I", "FRA", "IRQ", "Philadelphia Stadium", "Philadelphia", None, None),
    (43, "2026-06-22T17:00:00Z", "group_stage", "Group J", "ARG", "AUT", "Dallas Stadium", "Dallas", None, None),
    (44, "2026-06-23T03:00:00Z", "group_stage", "Group J", "JOR", "ALG", "San Francisco Bay Area Stadium", "San Francisco Bay Area", None, None),
    (45, "2026-06-23T20:00:00Z", "group_stage", "Group L", "ENG", "GHA", "Boston Stadium", "Boston", None, None),
    (46, "2026-06-23T23:00:00Z", "group_stage", "Group L", "PAN", "CRO", "Toronto Stadium", "Toronto", None, None),
    (47, "2026-06-23T17:00:00Z", "group_stage", "Group K", "POR", "UZB", "Houston Stadium", "Houston", None, None),
    (48, "2026-06-24T02:00:00Z", "group_stage", "Group K", "COL", "COD", "Guadalajara Stadium", "Guadalajara", None, None),
    (49, "2026-06-24T22:00:00Z", "group_stage", "Group C", "SCO", "BRA", "Miami Stadium", "Miami", None, None),
    (50, "2026-06-24T22:00:00Z", "group_stage", "Group C", "MAR", "HAI", "Atlanta Stadium", "Atlanta", None, None),
    (51, "2026-06-24T19:00:00Z", "group_stage", "Group B", "SUI", "CAN", "BC Place Vancouver", "Vancouver", None, None),
    (52, "2026-06-24T19:00:00Z", "group_stage", "Group B", "BIH", "QAT", "Seattle Stadium", "Seattle", None, None),
    (53, "2026-06-25T01:00:00Z", "group_stage", "Group A", "CZE", "MEX", "Mexico City Stadium", "Mexico City", None, None),
    (54, "2026-06-25T01:00:00Z", "group_stage", "Group A", "RSA", "KOR", "Monterrey Stadium", "Monterrey", None, None),
    (55, "2026-06-25T20:00:00Z", "group_stage", "Group E", "CUW", "CIV", "Philadelphia Stadium", "Philadelphia", None, None),
    (56, "2026-06-25T20:00:00Z", "group_stage", "Group E", "ECU", "GER", "New York/New Jersey Stadium", "New Jersey", None, None),
    (57, "2026-06-25T23:00:00Z", "group_stage", "Group F", "JPN", "SWE", "Dallas Stadium", "Dallas", None, None),
    (58, "2026-06-25T23:00:00Z", "group_stage", "Group F", "TUN", "NED", "Kansas City Stadium", "Kansas City", None, None),
    (59, "2026-06-26T02:00:00Z", "group_stage", "Group D", "TUR", "USA", "Los Angeles Stadium", "Los Angeles", None, None),
    (60, "2026-06-26T02:00:00Z", "group_stage", "Group D", "PAR", "AUS", "San Francisco Bay Area Stadium", "San Francisco Bay Area", None, None),
    (61, "2026-06-26T19:00:00Z", "group_stage", "Group I", "NOR", "FRA", "Boston Stadium", "Boston", None, None),
    (62, "2026-06-26T19:00:00Z", "group_stage", "Group I", "SEN", "IRQ", "Toronto Stadium", "Toronto", None, None),
    (63, "2026-06-27T03:00:00Z", "group_stage", "Group G", "EGY", "IRN", "Seattle Stadium", "Seattle", None, None),
    (64, "2026-06-27T03:00:00Z", "group_stage", "Group G", "NZL", "BEL", "BC Place Vancouver", "Vancouver", None, None),
    (65, "2026-06-27T00:00:00Z", "group_stage", "Group H", "CPV", "KSA", "Houston Stadium", "Houston", None, None),
    (66, "2026-06-27T00:00:00Z", "group_stage", "Group H", "URU", "ESP", "Guadalajara Stadium", "Guadalajara", None, None),
    (67, "2026-06-27T21:00:00Z", "group_stage", "Group L", "PAN", "ENG", "New York/New Jersey Stadium", "New Jersey", None, None),
    (68, "2026-06-27T21:00:00Z", "group_stage", "Group L", "CRO", "GHA", "Philadelphia Stadium", "Philadelphia", None, None),
    (69, "2026-06-28T02:00:00Z", "group_stage", "Group J", "ALG", "AUT", "Kansas City Stadium", "Kansas City", None, None),
    (70, "2026-06-28T02:00:00Z", "group_stage", "Group J", "JOR", "ARG", "Dallas Stadium", "Dallas", None, None),
    (71, "2026-06-27T23:30:00Z", "group_stage", "Group K", "COL", "POR", "Miami Stadium", "Miami", None, None),
    (72, "2026-06-27T23:30:00Z", "group_stage", "Group K", "COD", "UZB", "Atlanta Stadium", "Atlanta", None, None),
    (73, "2026-06-28T19:00:00Z", "round_of_32", None, "2A", "2B", "Los Angeles Stadium", "Los Angeles", None, None),
    (74, "2026-06-29T20:30:00Z", "round_of_32", None, "1E", "3ABCDF", "Boston Stadium", "Boston", None, None),
    (75, "2026-06-30T01:00:00Z", "round_of_32", None, "1F", "2C", "Monterrey Stadium", "Monterrey", None, None),
    (76, "2026-06-29T17:00:00Z", "round_of_32", None, "1C", "2F", "Houston Stadium", "Houston", None, None),
    (77, "2026-06-30T21:00:00Z", "round_of_32", None, "1I", "3CDFGH", "New York/New Jersey Stadium", "New Jersey", None, None),
    (78, "2026-06-30T17:00:00Z", "round_of_32", None, "2E", "2I", "Dallas Stadium", "Dallas", None, None),
    (79, "2026-07-01T01:00:00Z", "round_of_32", None, "1A", "3CEFHI", "Mexico City Stadium", "Mexico City", None, None),
    (80, "2026-07-01T16:00:00Z", "round_of_32", None, "1L", "3EHIJK", "Atlanta Stadium", "Atlanta", None, None),
    (81, "2026-07-02T00:00:00Z", "round_of_32", None, "1D", "3BEFIJ", "San Francisco Bay Area Stadium", "San Francisco Bay Area", None, None),
    (82, "2026-07-01T20:00:00Z", "round_of_32", None, "1G", "3AEHIJ", "Seattle Stadium", "Seattle", None, None),
    (83, "2026-07-02T23:00:00Z", "round_of_32", None, "2K", "2L", "Toronto Stadium", "Toronto", None, None),
    (84, "2026-07-02T19:00:00Z", "round_of_32", None, "1H", "2J", "Los Angeles Stadium", "Los Angeles", None, None),
    (85, "2026-07-03T03:00:00Z", "round_of_32", None, "1B", "3EFGIJ", "BC Place Vancouver", "Vancouver", None, None),
    (86, "2026-07-03T22:00:00Z", "round_of_32", None, "1J", "2H", "Miami Stadium", "Miami", None, None),
    (87, "2026-07-04T01:30:00Z", "round_of_32", None, "1K", "3DEIJL", "Kansas City Stadium", "Kansas City", None, None),
    (88, "2026-07-03T18:00:00Z", "round_of_32", None, "2D", "2G", "Dallas Stadium", "Dallas", None, None),
    (89, "2026-07-04T21:00:00Z", "round_of_16", None, "W74", "W77", "Philadelphia Stadium", "Philadelphia", None, None),
    (90, "2026-07-04T17:00:00Z", "round_of_16", None, "W73", "W75", "Houston Stadium", "Houston", None, None),
    (91, "2026-07-05T20:00:00Z", "round_of_16", None, "W76", "W78", "New York/New Jersey Stadium", "New Jersey", None, None),
    (92, "2026-07-06T00:00:00Z", "round_of_16", None, "W79", "W80", "Mexico City Stadium", "Mexico City", None, None),
    (93, "2026-07-06T19:00:00Z", "round_of_16", None, "W83", "W84", "Dallas Stadium", "Dallas", None, None),
    (94, "2026-07-07T00:00:00Z", "round_of_16", None, "W81", "W82", "Seattle Stadium", "Seattle", None, None),
    (95, "2026-07-07T16:00:00Z", "round_of_16", None, "W86", "W88", "Atlanta Stadium", "Atlanta", None, None),
    (96, "2026-07-07T20:00:00Z", "round_of_16", None, "W85", "W87", "BC Place Vancouver", "Vancouver", None, None),
    (97, "2026-07-09T20:00:00Z", "quarterfinal", None, "W89", "W90", "Boston Stadium", "Boston", None, None),
    (98, "2026-07-10T19:00:00Z", "quarterfinal", None, "W93", "W94", "Los Angeles Stadium", "Los Angeles", None, None),
    (99, "2026-07-11T21:00:00Z", "quarterfinal", None, "W91", "W92", "Miami Stadium", "Miami", None, None),
    (100, "2026-07-12T01:00:00Z", "quarterfinal", None, "W95", "W96", "Kansas City Stadium", "Kansas City", None, None),
    (101, "2026-07-14T19:00:00Z", "semifinal", None, "W97", "W98", "Dallas Stadium", "Dallas", None, None),
    (102, "2026-07-15T19:00:00Z", "semifinal", None, "W99", "W100", "Atlanta Stadium", "Atlanta", None, None),
    (103, "2026-07-18T21:00:00Z", "third_place", None, "RU101", "RU102", "Miami Stadium", "Miami", None, None),
    (104, "2026-07-19T19:00:00Z", "final", None, "W101", "W102", "New York/New Jersey Stadium", "New Jersey", None, None),
)


TEAM_NAMES: dict[str, str] = {
    "ALG": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CIV": "Côte d'Ivoire",
    "COD": "Congo DR",
    "COL": "Colombia",
    "CPV": "Cabo Verde",
    "CRO": "Croatia",
    "CUW": "Curaçao",
    "CZE": "Czechia",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ENG": "England",
    "ESP": "Spain",
    "FRA": "France",
    "GER": "Germany",
    "GHA": "Ghana",
    "HAI": "Haiti",
    "IRN": "IR Iran",
    "IRQ": "Iraq",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KOR": "Korea Republic",
    "KSA": "Saudi Arabia",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "NED": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PAN": "Panama",
    "PAR": "Paraguay",
    "POR": "Portugal",
    "QAT": "Qatar",
    "RSA": "South Africa",
    "SCO": "Scotland",
    "SEN": "Senegal",
    "SUI": "Switzerland",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "TUR": "Türkiye",
    "URU": "Uruguay",
    "USA": "USA",
    "UZB": "Uzbekistan",
}

TEAM_COUNTRIES: dict[str, str] = {
    **{code: code for code in TEAM_NAMES},
    "ENG": "ENG",
    "SCO": "SCO",
}

VENUE_COUNTRIES_BY_CITY: dict[str, str] = {
    "Atlanta": "USA",
    "Boston": "USA",
    "Dallas": "USA",
    "Houston": "USA",
    "Kansas City": "USA",
    "Los Angeles": "USA",
    "Miami": "USA",
    "New Jersey": "USA",
    "Philadelphia": "USA",
    "San Francisco Bay Area": "USA",
    "Seattle": "USA",
    "Mexico City": "MEX",
    "Guadalajara": "MEX",
    "Monterrey": "MEX",
    "Toronto": "CAN",
    "Vancouver": "CAN",
}


def world_cup_2026_fixtures() -> tuple[WorldCupFixture, ...]:
    """Return the normalized 104-match FIFA World Cup 2026 schedule."""
    return tuple(_fixture_from_row(row) for row in _FIXTURE_ROWS)


def world_cup_2026_team_codes() -> tuple[str, ...]:
    """Return participating team codes in stable alphabetical order."""
    return tuple(sorted(TEAM_NAMES))


def _fixture_from_row(row: tuple[object, ...]) -> WorldCupFixture:
    (
        match_number,
        match_date,
        round_name,
        group_name,
        home_slot,
        away_slot,
        stadium,
        city,
        played_home_goals,
        played_away_goals,
    ) = row
    city_name = str(city)
    return WorldCupFixture(
        match_number=int(match_number),
        match_date=datetime.fromisoformat(str(match_date).replace("Z", "+00:00")),
        round_name=str(round_name),
        group_name=str(group_name) if group_name is not None else None,
        home_slot=str(home_slot),
        away_slot=str(away_slot),
        stadium=str(stadium),
        city=city_name,
        country=VENUE_COUNTRIES_BY_CITY[city_name],
        played_home_goals=played_home_goals if played_home_goals is None else int(played_home_goals),
        played_away_goals=played_away_goals if played_away_goals is None else int(played_away_goals),
    )
