"""Dashboard constants and lightweight value objects."""

from __future__ import annotations

from dataclasses import dataclass

ROUND_ORDER = {
    "round_of_32": 4,
    "round_of_16": 5,
    "quarterfinal": 6,
    "semifinal": 7,
    "third_place": 8,
    "final": 9,
}

ROUND_LABELS = {
    "group_stage": "Fase de grupos",
    "round_of_32": "16 avos de final",
    "round_of_16": "Oitavas de final",
    "quarterfinal": "Quartas de final",
    "semifinal": "Semifinais",
    "third_place": "Disputa de 3o lugar",
    "final": "Final",
}

ROUND_DISPLAY_ORDER = {
    "group_stage": 1,
    "round_of_32": 2,
    "round_of_16": 3,
    "quarterfinal": 4,
    "semifinal": 5,
    "third_place": 6,
    "final": 7,
}

KNOCKOUT_MATCH_NUMBERS = tuple(range(73, 105))

PREDICTION_SOURCE_SIMULATION = "simulation"
PREDICTION_SOURCE_OUTCOME_MODEL = "outcome_model"
PREDICTION_SOURCE_LABELS = {
    PREDICTION_SOURCE_SIMULATION: "Simulacao",
    PREDICTION_SOURCE_OUTCOME_MODEL: "Modelo V/E/D",
}

BRACKET_ROUNDS = (
    ("round_of_32", "16 avos", tuple(range(73, 89))),
    ("round_of_16", "Oitavas", tuple(range(89, 97))),
    ("quarterfinal", "Quartas", tuple(range(97, 101))),
    ("semifinal", "Semifinais", (101, 102)),
    ("final", "Final", (104,)),
    ("third_place", "3o lugar", (103,)),
)

TEAM_NAMES_PT_BR = {
    "ALG": "Argelia",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgica",
    "BIH": "Bosnia e Herzegovina",
    "BRA": "Brasil",
    "CAN": "Canada",
    "CIV": "Costa do Marfim",
    "COD": "RD Congo",
    "COL": "Colombia",
    "CPV": "Cabo Verde",
    "CRO": "Croacia",
    "CUW": "Curacao",
    "CZE": "Republica Tcheca",
    "ECU": "Equador",
    "EGY": "Egito",
    "ENG": "Inglaterra",
    "ESP": "Espanha",
    "FRA": "Franca",
    "GER": "Alemanha",
    "GHA": "Gana",
    "HAI": "Haiti",
    "IRN": "Ira",
    "IRQ": "Iraque",
    "JOR": "Jordania",
    "JPN": "Japao",
    "KOR": "Coreia do Sul",
    "KSA": "Arabia Saudita",
    "MAR": "Marrocos",
    "MEX": "Mexico",
    "NED": "Paises Baixos",
    "NOR": "Noruega",
    "NZL": "Nova Zelandia",
    "PAN": "Panama",
    "PAR": "Paraguai",
    "POR": "Portugal",
    "QAT": "Catar",
    "RSA": "Africa do Sul",
    "SCO": "Escocia",
    "SEN": "Senegal",
    "SUI": "Suica",
    "SWE": "Suecia",
    "TUN": "Tunisia",
    "TUR": "Turquia",
    "URU": "Uruguai",
    "USA": "Estados Unidos",
    "UZB": "Uzbequistao",
}

TEAM_FLAG_EMOJI = {
    "ALG": "🇩🇿",
    "ARG": "🇦🇷",
    "AUS": "🇦🇺",
    "AUT": "🇦🇹",
    "BEL": "🇧🇪",
    "BIH": "🇧🇦",
    "BRA": "🇧🇷",
    "CAN": "🇨🇦",
    "CIV": "🇨🇮",
    "COD": "🇨🇩",
    "COL": "🇨🇴",
    "CPV": "🇨🇻",
    "CRO": "🇭🇷",
    "CUW": "🇨🇼",
    "CZE": "🇨🇿",
    "ECU": "🇪🇨",
    "EGY": "🇪🇬",
    "ENG": "🏴",
    "ESP": "🇪🇸",
    "FRA": "🇫🇷",
    "GER": "🇩🇪",
    "GHA": "🇬🇭",
    "HAI": "🇭🇹",
    "IRN": "🇮🇷",
    "IRQ": "🇮🇶",
    "JOR": "🇯🇴",
    "JPN": "🇯🇵",
    "KOR": "🇰🇷",
    "KSA": "🇸🇦",
    "MAR": "🇲🇦",
    "MEX": "🇲🇽",
    "NED": "🇳🇱",
    "NOR": "🇳🇴",
    "NZL": "🇳🇿",
    "PAN": "🇵🇦",
    "PAR": "🇵🇾",
    "POR": "🇵🇹",
    "QAT": "🇶🇦",
    "RSA": "🇿🇦",
    "SCO": "🏴",
    "SEN": "🇸🇳",
    "SUI": "🇨🇭",
    "SWE": "🇸🇪",
    "TUN": "🇹🇳",
    "TUR": "🇹🇷",
    "URU": "🇺🇾",
    "USA": "🇺🇸",
    "UZB": "🇺🇿",
}


@dataclass(frozen=True, slots=True)
class RoundOption:
    """Dashboard option for one displayable tournament round."""

    key: str
    label: str
    match_numbers: tuple[int, ...]
    first_column_header: str
    dynamic_matchups: bool = False
