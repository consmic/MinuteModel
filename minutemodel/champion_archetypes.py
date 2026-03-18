from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

SCORE_COLUMNS: List[str] = [
    "early_game_score",
    "scaling_score",
    "engage_score",
    "peel_or_disengage_score",
    "poke_score",
    "waveclear_score",
    "objective_burn_score",
    "pick_catch_score",
    "teamfight_score",
    "siege_score",
]


def _profile(**kwargs: float) -> Dict[str, float]:
    base = {name: 0.50 for name in SCORE_COLUMNS}
    for key, value in kwargs.items():
        if key not in base:
            continue
        base[key] = float(np.clip(value, 0.0, 1.0))
    return base


TEMPLATES: Dict[str, Dict[str, float]] = {
    "neutral": _profile(),
    "early_fighter": _profile(
        early_game_score=0.86,
        scaling_score=0.36,
        engage_score=0.62,
        peel_or_disengage_score=0.34,
        poke_score=0.26,
        waveclear_score=0.44,
        objective_burn_score=0.66,
        pick_catch_score=0.60,
        teamfight_score=0.58,
        siege_score=0.35,
    ),
    "scaling_carry": _profile(
        early_game_score=0.30,
        scaling_score=0.90,
        engage_score=0.28,
        peel_or_disengage_score=0.42,
        poke_score=0.46,
        waveclear_score=0.66,
        objective_burn_score=0.82,
        pick_catch_score=0.42,
        teamfight_score=0.78,
        siege_score=0.72,
    ),
    "engage_tank": _profile(
        early_game_score=0.62,
        scaling_score=0.58,
        engage_score=0.90,
        peel_or_disengage_score=0.66,
        poke_score=0.18,
        waveclear_score=0.36,
        objective_burn_score=0.30,
        pick_catch_score=0.72,
        teamfight_score=0.86,
        siege_score=0.20,
    ),
    "enchanter": _profile(
        early_game_score=0.42,
        scaling_score=0.62,
        engage_score=0.22,
        peel_or_disengage_score=0.92,
        poke_score=0.56,
        waveclear_score=0.34,
        objective_burn_score=0.18,
        pick_catch_score=0.38,
        teamfight_score=0.64,
        siege_score=0.28,
    ),
    "poke_mage": _profile(
        early_game_score=0.50,
        scaling_score=0.70,
        engage_score=0.24,
        peel_or_disengage_score=0.50,
        poke_score=0.92,
        waveclear_score=0.82,
        objective_burn_score=0.42,
        pick_catch_score=0.54,
        teamfight_score=0.66,
        siege_score=0.68,
    ),
    "skirmisher": _profile(
        early_game_score=0.74,
        scaling_score=0.60,
        engage_score=0.56,
        peel_or_disengage_score=0.26,
        poke_score=0.32,
        waveclear_score=0.46,
        objective_burn_score=0.70,
        pick_catch_score=0.78,
        teamfight_score=0.62,
        siege_score=0.34,
    ),
    "siege_marksman": _profile(
        early_game_score=0.48,
        scaling_score=0.82,
        engage_score=0.18,
        peel_or_disengage_score=0.38,
        poke_score=0.70,
        waveclear_score=0.60,
        objective_burn_score=0.86,
        pick_catch_score=0.38,
        teamfight_score=0.70,
        siege_score=0.92,
    ),
    "pick_support": _profile(
        early_game_score=0.66,
        scaling_score=0.46,
        engage_score=0.64,
        peel_or_disengage_score=0.50,
        poke_score=0.34,
        waveclear_score=0.24,
        objective_burn_score=0.16,
        pick_catch_score=0.90,
        teamfight_score=0.68,
        siege_score=0.20,
    ),
    "splitpush": _profile(
        early_game_score=0.54,
        scaling_score=0.72,
        engage_score=0.30,
        peel_or_disengage_score=0.26,
        poke_score=0.22,
        waveclear_score=0.68,
        objective_burn_score=0.84,
        pick_catch_score=0.52,
        teamfight_score=0.40,
        siege_score=0.76,
    ),
    "teamfight_mage": _profile(
        early_game_score=0.44,
        scaling_score=0.78,
        engage_score=0.34,
        peel_or_disengage_score=0.46,
        poke_score=0.56,
        waveclear_score=0.86,
        objective_burn_score=0.44,
        pick_catch_score=0.56,
        teamfight_score=0.88,
        siege_score=0.60,
    ),
}


CHAMPION_TEMPLATE_OVERRIDES: Dict[str, str] = {
    "Lee Sin": "early_fighter",
    "Elise": "early_fighter",
    "Renekton": "early_fighter",
    "Pantheon": "early_fighter",
    "Nidalee": "early_fighter",
    "Jarvan IV": "early_fighter",
    "Vi": "early_fighter",
    "Xin Zhao": "early_fighter",
    "Volibear": "early_fighter",
    "Wukong": "early_fighter",
    "Nocturne": "early_fighter",
    "Rek'Sai": "early_fighter",
    "Rumble": "early_fighter",
    "Gnar": "early_fighter",
    "Sett": "early_fighter",
    "Taliyah": "poke_mage",
    "Zoe": "poke_mage",
    "Jayce": "poke_mage",
    "Varus": "siege_marksman",
    "Caitlyn": "siege_marksman",
    "Ezreal": "siege_marksman",
    "Sivir": "siege_marksman",
    "Corki": "siege_marksman",
    "Tristana": "siege_marksman",
    "Zeri": "siege_marksman",
    "Jinx": "scaling_carry",
    "Aphelios": "scaling_carry",
    "Kai'Sa": "scaling_carry",
    "Kog'Maw": "scaling_carry",
    "Smolder": "scaling_carry",
    "Vayne": "scaling_carry",
    "Xayah": "scaling_carry",
    "Senna": "scaling_carry",
    "Syndra": "teamfight_mage",
    "Orianna": "teamfight_mage",
    "Azir": "teamfight_mage",
    "Viktor": "teamfight_mage",
    "Cassiopeia": "teamfight_mage",
    "Anivia": "teamfight_mage",
    "Seraphine": "teamfight_mage",
    "Hwei": "teamfight_mage",
    "Ryze": "teamfight_mage",
    "LeBlanc": "skirmisher",
    "Akali": "skirmisher",
    "Sylas": "skirmisher",
    "Yone": "skirmisher",
    "Yasuo": "skirmisher",
    "Irelia": "skirmisher",
    "Katarina": "skirmisher",
    "Qiyana": "skirmisher",
    "Viego": "skirmisher",
    "Bel'Veth": "skirmisher",
    "Rengar": "skirmisher",
    "Kha'Zix": "skirmisher",
    "Nautilus": "engage_tank",
    "Rell": "engage_tank",
    "Alistar": "engage_tank",
    "Leona": "engage_tank",
    "Sejuani": "engage_tank",
    "Maokai": "engage_tank",
    "Sion": "engage_tank",
    "Ornn": "engage_tank",
    "Poppy": "engage_tank",
    "Rakan": "engage_tank",
    "Braum": "engage_tank",
    "Tahm Kench": "engage_tank",
    "Thresh": "pick_support",
    "Blitzcrank": "pick_support",
    "Pyke": "pick_support",
    "Bard": "pick_support",
    "Renata Glasc": "pick_support",
    "Milio": "enchanter",
    "Lulu": "enchanter",
    "Nami": "enchanter",
    "Janna": "enchanter",
    "Soraka": "enchanter",
    "Yuumi": "enchanter",
    "Karma": "enchanter",
    "Sona": "enchanter",
    "Gwen": "splitpush",
    "Camille": "splitpush",
    "Fiora": "splitpush",
    "Jax": "splitpush",
    "Tryndamere": "splitpush",
    "Yorick": "splitpush",
    "Trundle": "splitpush",
    "K'Sante": "engage_tank",
    "Aurora": "teamfight_mage",
}


def normalize_champion_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def champion_profile(champion_name: Any) -> Dict[str, float]:
    champion = normalize_champion_name(champion_name)
    if not champion:
        return TEMPLATES["neutral"]
    template_name = CHAMPION_TEMPLATE_OVERRIDES.get(champion, "neutral")
    return TEMPLATES.get(template_name, TEMPLATES["neutral"])


def aggregate_team_scores(champions: Iterable[Any]) -> Dict[str, float]:
    vectors: List[Dict[str, float]] = []
    for champ in champions:
        normalized = normalize_champion_name(champ)
        if normalized:
            vectors.append(champion_profile(normalized))

    if not vectors:
        return dict(TEMPLATES["neutral"])

    out: Dict[str, float] = {}
    for score in SCORE_COLUMNS:
        out[score] = float(np.mean([vec[score] for vec in vectors]))
    return out


def classify_archetype(team_scores: Dict[str, float]) -> str:
    early = float(team_scores.get("early_game_score", 0.5))
    scaling = float(team_scores.get("scaling_score", 0.5))
    engage = float(team_scores.get("engage_score", 0.5))
    teamfight = float(team_scores.get("teamfight_score", 0.5))
    poke = float(team_scores.get("poke_score", 0.5))

    if early - scaling >= 0.12:
        return "early"
    if scaling - early >= 0.12:
        return "scaling"
    if (engage + teamfight) >= 1.35 and poke < 0.55:
        return "teamfight"
    return "balanced"


def is_skirmish_comp(team_scores: Dict[str, float]) -> bool:
    engage = float(team_scores.get("engage_score", 0.5))
    pick = float(team_scores.get("pick_catch_score", 0.5))
    teamfight = float(team_scores.get("teamfight_score", 0.5))
    return (engage + pick + teamfight) >= 1.85
