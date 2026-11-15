"""
Agente de Avaliação Nutricional Infantil (Fase 4 — GAIA).

Recebe o perfil de uma criança e retorna:
  - z-scores WAZ, HAZ e WHZ (método LMS da OMS 2006)
  - classificação do estado nutricional (SAM / MAM / Normal / Sobrepeso)
  - metas de ingestão (DRI por faixa etária e sexo)
  - fatores de ajuste para desnutrição terapêutica

Uso:
    from agents.assessment_agent import AssessmentAgent
    agent = AssessmentAgent()
    result = agent.assess({
        "age_months": 18,
        "sex": "M",            # "M" ou "F"
        "weight_kg": 8.5,
        "height_cm": 78.0,
        "region": "Nordeste",  # opcional
    })
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

# ── caminhos ──────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMS_PATH = os.path.join(_BASE, "data", "reference", "who_lms_tables.json")
DRI_PATH = os.path.join(_BASE, "data", "reference", "dri.json")


# ── tipos de retorno ──────────────────────────────────────────────────────────
@dataclass
class ZScores:
    WAZ: Optional[float]  # peso-para-idade
    HAZ: Optional[float]  # comprimento/altura-para-idade
    WHZ: Optional[float]  # peso-para-comprimento/altura


@dataclass
class NutritionalStatus:
    classification: str        # classificação primária para ajuste de DRI
    waz_class: str             # classificação pelo WAZ
    haz_class: str             # classificação pelo HAZ
    whz_class: str             # classificação pelo WHZ
    z_scores: ZScores
    conditions: list[str] = field(default_factory=list)  # condições ativas (modelo ENANI/OMS)


@dataclass
class AssessmentResult:
    child_profile: dict
    age_group: str
    z_scores: ZScores
    status: NutritionalStatus
    dri: dict             # metas nutricionais para a faixa/sexo
    adjustment: dict      # fatores de ajuste energético e proteico
    age_group_label: str  # ex: "12-23 meses"


# ── utilitários LMS ───────────────────────────────────────────────────────────
def _lms_zscore(X: float, L: float, M: float, S: float) -> float:
    """Fórmula LMS da OMS. Truncada no intervalo [-5, +5]."""
    if abs(L) < 1e-6:
        z = math.log(X / M) / S
    else:
        z = ((X / M) ** L - 1.0) / (L * S)
    return max(-5.0, min(5.0, round(z, 2)))


def _interpolate_lms(table: dict, key: float) -> Optional[tuple]:
    """
    Interpola linearmente entre dois pontos inteiros da tabela LMS.
    Retorna (L, M, S) ou None se fora do intervalo.
    """
    keys_int = sorted(int(k) for k in table)
    if not keys_int:
        return None

    lo = int(math.floor(key))
    hi = int(math.ceil(key))

    if lo < keys_int[0] or hi > keys_int[-1]:
        return None

    if lo == hi:
        return tuple(table[str(lo)])

    lms_lo = table.get(str(lo))
    lms_hi = table.get(str(hi))
    if lms_lo is None or lms_hi is None:
        return None

    frac = key - lo
    L = lms_lo[0] + frac * (lms_hi[0] - lms_lo[0])
    M = lms_lo[1] + frac * (lms_hi[1] - lms_lo[1])
    S = lms_lo[2] + frac * (lms_hi[2] - lms_lo[2])
    return (L, M, S)


# ── classificações ────────────────────────────────────────────────────────────
def _classify_waz(z: Optional[float]) -> str:
    if z is None:
        return "unavailable"
    if z < -3:
        return "severe_underweight"
    if z < -2:
        return "underweight"
    if z <= 2:
        return "adequate"
    return "high_weight"


def _classify_haz(z: Optional[float]) -> str:
    if z is None:
        return "unavailable"
    if z < -3:
        return "severe_stunting"
    if z < -2:
        return "stunting"
    if z <= 3:
        return "adequate_height"
    return "tall_stature"


def _classify_whz(z: Optional[float]) -> str:
    if z is None:
        return "unavailable"
    if z < -3:
        return "severe_wasting"   # SAM
    if z < -2:
        return "wasting"          # MAM
    if z <= 1:
        return "normal_weight"
    if z <= 2:
        return "overweight_risk"
    if z <= 3:
        return "overweight"
    return "obese"


def _get_active_conditions(
    whz: Optional[float],
    waz: Optional[float],
    haz: Optional[float],
) -> list[str]:
    """Retorna todas as condições ativas simultaneamente (modelo ENANI/OMS independente).

    Cada indicador é avaliado de forma independente — uma criança pode ter
    emaciação (WHZ) + baixa estatura (HAZ) + baixo peso (WAZ) ao mesmo tempo.
    """
    conds: list[str] = []
    if whz is not None:
        if whz < -3:
            conds.append("SAM")
        elif whz < -2:
            conds.append("MAM")
        elif whz > 3:
            conds.append("Obese")
        elif whz > 2:
            conds.append("Overweight")
        elif whz > 1:
            conds.append("Overweight_Risk")
    if haz is not None:
        if haz < -3:
            conds.append("Stunting_Severe")
        elif haz < -2:
            conds.append("Stunting")
    if waz is not None:
        if waz < -3:
            conds.append("Underweight_Severe")
        elif waz < -2:
            conds.append("Underweight")
    if not conds:
        conds.append("Normal")
    return conds


def _overall_classification(
    whz: Optional[float],
    waz: Optional[float],
    haz: Optional[float] = None,
) -> str:
    """Classificação primária para seleção do fator de ajuste de DRI.

    Prioridade: desnutrição aguda (WHZ) > crônica/stunting (HAZ/WAZ) > Normal > sobrepeso.
    A lista completa de condições ativas é armazenada em NutritionalStatus.conditions.
    """
    conds = _get_active_conditions(whz, waz, haz)
    if "SAM" in conds:
        return "SAM"
    if "MAM" in conds:
        return "MAM"
    if "Stunting_Severe" in conds or "Stunting" in conds or "Underweight_Severe" in conds or "Underweight" in conds:
        return "Stunting"
    if "Obese" in conds:
        return "Obese"
    if "Overweight" in conds:
        return "Overweight"
    if "Overweight_Risk" in conds:
        return "Overweight_Risk"
    return "Normal"


# ── fatores de ajuste ─────────────────────────────────────────────────────────
_ADJUSTMENT_TABLE = {
    # (classification, age_group) -> {energy_factor, protein_factor, note}
    "SAM": {
        "default":  {"energy_factor": 1.50, "protein_factor": 1.80,
                     "note": "SAM rehabilitation: +50% energy, +80% protein"},
    },
    "MAM": {
        "default":  {"energy_factor": 1.20, "protein_factor": 1.30,
                     "note": "MAM rehabilitation: +20% energy, +30% protein"},
    },
    "Stunting": {
        "default":  {"energy_factor": 1.20, "protein_factor": 1.20,
                     "note": "Stunting (HAZ < -2): +20% energia e proteína para catch-up (Golden 2009)"},
    },
    "Normal": {
        "default":  {"energy_factor": 1.00, "protein_factor": 1.00,
                     "note": "Adequate nutritional status"},
    },
    "Overweight_Risk": {
        "default":  {"energy_factor": 0.90, "protein_factor": 1.00,
                     "note": "Overweight risk: mild energy reduction"},
    },
    "Overweight": {
        "default":  {"energy_factor": 0.85, "protein_factor": 1.00,
                     "note": "Overweight: moderate energy reduction"},
    },
    "Obese": {
        "default":  {"energy_factor": 0.80, "protein_factor": 1.00,
                     "note": "Obesity: energy reduction under supervision"},
    },
}


def _get_adjustment(classification: str) -> dict:
    entry = _ADJUSTMENT_TABLE.get(classification, _ADJUSTMENT_TABLE["Normal"])
    return entry.get("default", entry[list(entry.keys())[0]])


# ── faixa etária ──────────────────────────────────────────────────────────────
def _age_group(months: int) -> tuple[str, str]:
    """Retorna (code, label)."""
    if months < 9:
        return "6_8m", "6-8 months"
    if months < 12:
        return "9_11m", "9-11 months"
    if months < 24:
        return "12_23m", "12-23 months"
    if months < 36:
        return "24_35m", "24-35 months"
    return "36_59m", "36-59 months"


# ── agente principal ──────────────────────────────────────────────────────────
class AssessmentAgent:
    """Avalia o estado nutricional de uma criança de 0-60 meses."""

    def __init__(self):
        with open(LMS_PATH, encoding="utf-8") as f:
            self._lms = json.load(f)
        with open(DRI_PATH, encoding="utf-8") as f:
            self._dri = json.load(f)

    # ── z-scores ──────────────────────────────────────────────────────────────
    def _compute_waz(self, weight: float, age_months: float, sex: str) -> Optional[float]:
        sex_key = "boys" if sex == "M" else "girls"
        table = self._lms["WAZ"][sex_key]
        lms = _interpolate_lms(table, min(age_months, 60))
        if lms is None:
            return None
        return _lms_zscore(weight, *lms)

    def _compute_haz(self, height: float, age_months: float, sex: str) -> Optional[float]:
        sex_key = "boys" if sex == "M" else "girls"
        table = self._lms["HAZ"][sex_key]
        lms = _interpolate_lms(table, min(age_months, 60))
        if lms is None:
            return None
        return _lms_zscore(height, *lms)

    def _compute_whz(self, weight: float, height: float,
                     age_months: float, sex: str) -> Optional[float]:
        sex_key = "boys" if sex == "M" else "girls"
        # < 24 meses: tabela de comprimento; >= 24 meses: tabela de altura
        sub = "length" if age_months < 24 else "height"
        table = self._lms["WHZ"][sex_key][sub]
        # arredonda para cm inteiro mais próximo para busca na tabela
        h_rounded = round(height)
        lms = _interpolate_lms(table, height)
        if lms is None:
            # tenta com limite da tabela
            keys = sorted(int(k) for k in table)
            h_clamped = max(keys[0], min(keys[-1], h_rounded))
            lms = _interpolate_lms(table, float(h_clamped))
        if lms is None:
            return None
        return _lms_zscore(weight, *lms)

    # ── DRI ajustada ──────────────────────────────────────────────────────────
    def _get_dri(self, age_group: str, sex: str) -> dict:
        group = self._dri.get(age_group, {})
        return group.get(sex, group.get("M", {}))

    def _apply_adjustment(self, dri: dict, adj: dict) -> dict:
        ef = adj["energy_factor"]
        pf = adj["protein_factor"]
        adjusted = dict(dri)
        if "energy_kcal" in adjusted and adjusted["energy_kcal"]:
            adjusted["energy_kcal"] = round(adjusted["energy_kcal"] * ef)
        if "protein_g" in adjusted and adjusted["protein_g"]:
            adjusted["protein_g"] = round(adjusted["protein_g"] * pf, 1)
        return adjusted

    # ── método público ────────────────────────────────────────────────────────
    def assess(self, child: dict) -> AssessmentResult:
        """
        Parâmetros esperados em `child`:
            age_months (int|float) — idade em meses
            sex        (str)       — "M" ou "F"
            weight_kg  (float)     — peso em kg
            height_cm  (float)     — comprimento/altura em cm  (opcional)
            region     (str)       — região do Brasil           (opcional)
        """
        age  = float(child["age_months"])
        sex  = child.get("sex", "M").upper()
        wt   = float(child["weight_kg"])
        ht   = child.get("height_cm")
        ht   = float(ht) if ht is not None else None

        ag_code, ag_label = _age_group(int(round(age)))

        # z-scores
        waz = self._compute_waz(wt, age, sex)
        haz = self._compute_haz(ht, age, sex) if ht else None
        whz = self._compute_whz(wt, ht, age, sex) if ht else None
        zs  = ZScores(WAZ=waz, HAZ=haz, WHZ=whz)

        # classificação
        overall = _overall_classification(whz, waz, haz)
        status = NutritionalStatus(
            classification=overall,
            waz_class=_classify_waz(waz),
            haz_class=_classify_haz(haz),
            whz_class=_classify_whz(whz),
            z_scores=zs,
            conditions=_get_active_conditions(whz, waz, haz),
        )

        # DRI base e ajustada
        dri_base = self._get_dri(ag_code, sex)
        adj      = _get_adjustment(overall)
        dri_adj  = self._apply_adjustment(dri_base, adj)

        return AssessmentResult(
            child_profile=child,
            age_group=ag_code,
            z_scores=zs,
            status=status,
            dri=dri_adj,
            adjustment=adj,
            age_group_label=ag_label,
        )

    def assess_dict(self, child: dict) -> dict:
        """Mesmo que assess(), mas retorna dicionário serializável em JSON."""
        r = self.assess(child)
        return {
            "child_profile": r.child_profile,
            "age_group": r.age_group,
            "age_group_label": r.age_group_label,
            "z_scores": {
                "WAZ": r.z_scores.WAZ,
                "HAZ": r.z_scores.HAZ,
                "WHZ": r.z_scores.WHZ,
            },
            "nutritional_status": {
                "classification": r.status.classification,
                "conditions": r.status.conditions,
                "waz_class": r.status.waz_class,
                "haz_class": r.status.haz_class,
                "whz_class": r.status.whz_class,
            },
            "dri_adjusted": r.dri,
            "adjustment": r.adjustment,
        }
