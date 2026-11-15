"""
Avaliação do GAIA com 500 perfis sintéticos estratificados.

Estratificação:
  5 regiões × 2 sexos × 5 faixas etárias × 10 perfis = 500 perfis

Faixas etárias: 6-8m, 9-11m, 12-23m, 24-35m, 36-59m
Regiões: Norte, Nordeste, Centro-Oeste, Sudeste, Sul
Sexos: M, F

Antropometria: medianas WHO 2006 MGRS ± variação aleatória (normal)
  90% de perfis com estado nutricional normal (WHZ >= -2)
  10% com MAM (WHZ -3 a -2)

Saída: tmp_who/batch_500_results.csv
"""

from __future__ import annotations

import os
import sys
import warnings
import random

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.gaia import GAIA

SEED       = 42
OUT_CSV    = "tmp_who/batch_500_results.csv"
FOODS_CSV  = "data/processed/unified_foods.csv"

_NAS_COL = {
    "6-8m":   "NAS_6_8m",
    "9-11m":  "NAS_9_11m",
    "12-23m": "NAS_12_23m",
    "24-35m": "NAS_24_35m",
    "36-59m": "NAS_36_59m",
}

def _build_nas_lookup() -> dict[str, dict[str, float]]:
    """food_id -> {stratum -> NAS}"""
    foods = pd.read_csv(FOODS_CSV)
    lookup: dict[str, dict[str, float]] = {}
    for _, row in foods.iterrows():
        fid = str(row["food_id"])
        lookup[fid] = {s: float(row.get(col, 0) or 0) for s, col in _NAS_COL.items()}
    return lookup
REGIONS    = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
SEXES      = ["M", "F"]
N_PER_STRATUM = 10

# Mediana de peso (kg) e altura (cm) WHO 2006 MGRS por mês de idade
# Fonte: WHO Child Growth Standards, LMS tables (coluna M para meninos e meninas)
_WHO_MEDIAN: dict[str, dict[int, tuple[float, float]]] = {
    "M": {
        0:  (3.35, 49.9),  1:  (4.47, 54.7),  2:  (5.57, 58.4),
        3:  (6.38, 61.4),  4:  (7.00, 63.9),  5:  (7.51, 65.9),
        6:  (7.93, 67.6),  7:  (8.30, 69.2),  8:  (8.62, 70.6),
        9:  (8.90, 72.0),  10: (9.16, 73.3),  11: (9.41, 74.5),
        12: (9.65, 75.7),  15: (10.3, 79.1),  18: (10.9, 82.3),
        21: (11.5, 85.1),  24: (12.2, 87.1),  30: (13.3, 92.1),
        36: (14.3, 96.1),  42: (15.3, 99.9),  48: (16.3, 103.3),
        54: (17.3, 106.7), 59: (18.3, 109.7),
    },
    "F": {
        0:  (3.23, 49.1),  1:  (4.19, 53.7),  2:  (5.13, 57.1),
        3:  (5.85, 59.8),  4:  (6.42, 62.1),  5:  (6.90, 64.0),
        6:  (7.30, 65.7),  7:  (7.64, 67.3),  8:  (7.95, 68.7),
        9:  (8.23, 70.1),  10: (8.48, 71.5),  11: (8.72, 72.8),
        12: (8.95, 74.0),  15: (9.56, 77.5),  18: (10.2, 80.7),
        21: (10.9, 83.7),  24: (11.5, 85.7),  30: (12.7, 90.7),
        36: (13.9, 95.1),  42: (15.0, 99.0),  48: (16.0, 102.7),
        54: (17.0, 106.2), 59: (17.9, 109.4),
    },
}

# Faixas etárias: (label, min_months, max_months)
AGE_BANDS: list[tuple[str, int, int]] = [
    ("0-5m",   0,  5),
    ("6-8m",   6,  8),
    ("9-11m",  9, 11),
    ("12-23m", 12, 23),
    ("24-59m", 24, 59),
]


def _interpolate_median(sex: str, age_m: int) -> tuple[float, float]:
    """Interpola medianas WHO para qualquer idade em meses."""
    table = _WHO_MEDIAN[sex]
    keys  = sorted(table.keys())
    if age_m <= keys[0]:
        return table[keys[0]]
    if age_m >= keys[-1]:
        return table[keys[-1]]
    lo = max(k for k in keys if k <= age_m)
    hi = min(k for k in keys if k >= age_m)
    if lo == hi:
        return table[lo]
    t  = (age_m - lo) / (hi - lo)
    w0, h0 = table[lo]
    w1, h1 = table[hi]
    return w0 + t * (w1 - w0), h0 + t * (h1 - h0)


def _generate_profile(
    age_months: float,
    sex: str,
    region: str,
    rng: np.random.Generator,
    mam: bool = False,
) -> dict:
    """Gera um perfil com antropometria realista ao redor da mediana WHO."""
    w_med, h_med = _interpolate_median(sex, int(age_months))

    # Variação: ±8% no peso, ±3cm na altura (normal clipped)
    w_sd = w_med * 0.08
    h_sd = 3.0
    w = float(np.clip(rng.normal(w_med, w_sd), w_med * 0.7, w_med * 1.3))
    h = float(np.clip(rng.normal(h_med, h_sd), h_med - 8, h_med + 8))

    if mam:
        # MAM: reduz peso para WHZ ≈ -2.5 (aprox. -15% no peso)
        w *= 0.85

    return {
        "age_months": round(age_months, 1),
        "sex":        sex,
        "weight_kg":  round(w, 2),
        "height_cm":  round(h, 1),
        "region":     region,
    }


def build_profiles(seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    profiles = []

    for label, lo, hi in AGE_BANDS:
        for sex in SEXES:
            for region in REGIONS:
                for i in range(N_PER_STRATUM):
                    age_m = float(rng.integers(lo, hi + 1))
                    mam   = (i == N_PER_STRATUM - 1)   # último perfil de cada estrato é MAM
                    p     = _generate_profile(age_m, sex, region, rng, mam=mam)
                    p["stratum"] = label
                    profiles.append(p)

    return profiles


def pct(achieved: float, target: float) -> float | None:
    return round(achieved / target * 100, 1) if target and target > 0 else None


def _nas_wavg(plan, stratum: str, nas_lookup: dict) -> float | None:
    total_g, total_nas_g = 0.0, 0.0
    for meal in plan.meals:
        for item in meal.items:
            g   = item.portion_g
            nas = nas_lookup.get(item.food_id, {}).get(stratum, 0.0)
            total_nas_g += nas * g
            total_g     += g
    return round(total_nas_g / total_g, 4) if total_g > 0 else None


def run():
    nas_lookup = _build_nas_lookup()
    profiles   = build_profiles()
    print(f"Total de perfis gerados: {len(profiles)}")
    strata_counts = {}
    for p in profiles:
        strata_counts[p["stratum"]] = strata_counts.get(p["stratum"], 0) + 1
    for k, v in sorted(strata_counts.items()):
        print(f"  {k}: {v} perfis")

    gaia = GAIA(use_gemini=False)
    rows = []

    for i, profile in enumerate(profiles, 1):
        try:
            result = gaia.run(profile, verbose=False)
            plan   = result.plan
            ach    = plan.daily_achieved
            tgt    = plan.daily_targets

            rows.append({
                "profile_id":  i,
                "stratum":     profile["stratum"],
                "age_months":  profile["age_months"],
                "sex":         profile["sex"],
                "region":      profile["region"],
                "weight_kg":   profile["weight_kg"],
                "height_cm":   profile["height_cm"],
                "lp_status":   plan.lp_status,
                "n_foods":     sum(len(m.items) for m in plan.meals),
                "energy_pct":  pct(ach.get("energy_kcal", 0), tgt.get("energy_kcal")),
                "protein_pct": pct(ach.get("protein_g",   0), tgt.get("protein_g")),
                "iron_pct":    pct(ach.get("iron_mg",     0), tgt.get("iron_mg")),
                "calcium_pct": pct(ach.get("calcium_mg",  0), tgt.get("calcium_mg")),
                "zinc_pct":    pct(ach.get("zinc_mg",     0), tgt.get("zinc_mg")),
                "vitA_pct":    pct(ach.get("vitA_ug",     0), tgt.get("vitA_ug")),
                "vitC_pct":    pct(ach.get("vitC_mg",     0), tgt.get("vitC_mg")),
                "nas_wavg":    _nas_wavg(plan, profile["stratum"], nas_lookup),
            })

            tag = f"[{plan.lp_status}]" if plan.lp_status != "optimal" else ""
            print(f"  [{i:3d}/500] {profile['age_months']}m {profile['sex']} "
                  f"{profile['region']:<12} -> {plan.lp_status} {tag}")

        except Exception as e:
            print(f"  [{i:3d}/500] ERRO: {e}")
            rows.append({"profile_id": i, "lp_status": "error",
                         "stratum": profile.get("stratum"), **profile})

    df = pd.DataFrame(rows)
    os.makedirs("tmp_who", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResultados salvos em: {OUT_CSV}")

    # ── resumo ────────────────────────────────────────────────────────────────
    ok = df[df["lp_status"] != "error"]
    print(f"\n{'='*55}")
    print(f"{'RESUMO':^55}")
    print(f"{'='*55}")
    print(f"  Total rodados:  {len(df)}")
    print(f"  Erros:          {(df['lp_status'] == 'error').sum()}")
    print(f"\n  Status LP:")
    for s, cnt in ok["lp_status"].value_counts().items():
        print(f"    {s:<12}: {cnt:>3}  ({cnt/len(ok)*100:.1f}%)")

    nutrients = ["energy_pct","protein_pct","iron_pct","calcium_pct",
                 "zinc_pct","vitA_pct","vitC_pct"]
    print(f"\n  {'Nutriente':<14}  {'Mediana %DRI':>13}  {'P10':>7}  {'P90':>7}")
    print(f"  {'-'*47}")
    for nut in nutrients:
        col = ok[nut].dropna()
        if col.empty:
            continue
        print(f"  {nut:<14}  {col.median():>13.1f}  "
              f"{col.quantile(0.10):>7.1f}  {col.quantile(0.90):>7.1f}")

    print(f"\n  Por faixa etária (mediana % DRI energia):")
    for label, _, _ in AGE_BANDS:
        sub = ok[ok["stratum"] == label]["energy_pct"].dropna()
        print(f"    {label:<8}: {sub.median():.1f}% (n={len(sub)})")


if __name__ == "__main__":
    run()
