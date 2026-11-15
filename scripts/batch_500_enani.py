"""
Avaliação do GAIA com 500 perfis reais amostrados do ENANI-2019.

Estratificação:
  5 regiões × 2 sexos × 5 faixas etárias × 10 perfis = 500 perfis

Fonte dos perfis: data/raw/enani_profiles.csv
  - Idade: s00d_idade_peso (dias) / 30.4375
  - Peso:  s02_peso_medida1 (kg)
  - Altura: s07_altura_medida1 (cm)
  - Sexo:  b02_sexo (Masculino/Feminino)
  - Região: a00_regiao

Saída: tmp_who/batch_500_enani_results.csv
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.gaia import GAIA

SEED      = 42
ENANI_CSV = "data/raw/enani_profiles.csv"
FOODS_CSV = "data/processed/unified_foods.csv"
OUT_CSV   = "tmp_who/batch_500_enani_results.csv"

REGIONS   = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
SEXES     = ["M", "F"]
AGE_BANDS = ["6-8m", "9-11m", "12-23m", "24-35m", "36-59m"]
AGE_BINS  = [5.5, 8.5, 11.5, 23.5, 35.5, 60]
N_PER_STRATUM = 10

_NAS_COL = {
    "6-8m":   "NAS_6_8m",
    "9-11m":  "NAS_9_11m",
    "12-23m": "NAS_12_23m",
    "24-35m": "NAS_24_35m",
    "36-59m": "NAS_36_59m",
}


def load_enani() -> pd.DataFrame:
    print(f"Carregando ENANI de {ENANI_CSV} ...")
    df = pd.read_csv(ENANI_CSV, low_memory=False)

    out = pd.DataFrame({
        "age_months": pd.to_numeric(df["s00d_idade_peso"],    errors="coerce") / 30.4375,
        "weight_kg":  pd.to_numeric(df["s02_peso_medida1"],   errors="coerce"),
        "height_cm":  pd.to_numeric(df["s07_altura_medida1"], errors="coerce"),
        "sex":        df["b02_sexo"].map({"Masculino": "M", "Feminino": "F"}),
        "region":     df["a00_regiao"],
    })

    out = out.dropna()
    out = out[
        out["age_months"].between(6, 60) &
        (out["weight_kg"] > 0) &
        (out["height_cm"] > 0)
    ].copy()

    out["stratum"] = pd.cut(
        out["age_months"], bins=AGE_BINS, labels=AGE_BANDS
    ).astype(str)

    print(f"  Perfis válidos: {len(out)}")
    return out.reset_index(drop=True)


def sample_stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Amostra n perfis por combinação faixa × sexo × região."""
    rows = []
    rng = np.random.default_rng(seed)

    for band in AGE_BANDS:
        for sex in SEXES:
            for region in REGIONS:
                sub = df[
                    (df["stratum"] == band) &
                    (df["sex"] == sex) &
                    (df["region"] == region)
                ]
                if len(sub) < n:
                    print(f"  AVISO: {band} {sex} {region} tem só {len(sub)} perfis (necessário {n})")
                    sample = sub
                else:
                    sample = sub.sample(n=n, random_state=int(rng.integers(0, 99999)))
                rows.append(sample)

    result = pd.concat(rows, ignore_index=True)
    print(f"  Amostra total: {len(result)} perfis")
    return result


def _build_nas_lookup() -> dict[str, dict[str, float]]:
    foods = pd.read_csv(FOODS_CSV)
    return {
        str(row["food_id"]): {
            s: float(row.get(col, 0) or 0)
            for s, col in _NAS_COL.items()
        }
        for _, row in foods.iterrows()
    }


def _nas_wavg(plan, stratum: str, nas_lookup: dict) -> float | None:
    total_g, total_nas_g = 0.0, 0.0
    for meal in plan.meals:
        for item in meal.items:
            g   = item.portion_g
            nas = nas_lookup.get(item.food_id, {}).get(stratum, 0.0)
            total_nas_g += nas * g
            total_g     += g
    return round(total_nas_g / total_g, 4) if total_g > 0 else None


def pct(achieved: float, target: float) -> float | None:
    return round(achieved / target * 100, 1) if target and target > 0 else None


def run():
    enani      = load_enani()
    sample     = sample_stratified(enani, N_PER_STRATUM, SEED)
    nas_lookup = _build_nas_lookup()

    gaia = GAIA(use_gemini=False)
    rows = []
    total = len(sample)

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        profile = {
            "age_months": round(float(row["age_months"]), 1),
            "sex":        str(row["sex"]),
            "weight_kg":  round(float(row["weight_kg"]), 2),
            "height_cm":  round(float(row["height_cm"]), 1),
            "region":     str(row["region"]),
        }
        stratum = str(row["stratum"])

        try:
            result = gaia.run(profile, verbose=False)
            plan   = result.plan
            ach    = plan.daily_achieved
            tgt    = plan.daily_targets

            rows.append({
                "profile_id":  i,
                "stratum":     stratum,
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
                "nas_wavg":    _nas_wavg(plan, stratum, nas_lookup),
            })

            tag = f"[{plan.lp_status}]" if plan.lp_status != "optimal" else ""
            print(f"  [{i:3d}/{total}] {profile['age_months']:>4.1f}m {profile['sex']} "
                  f"{profile['region']:<12} -> {plan.lp_status} {tag}")

        except Exception as e:
            print(f"  [{i:3d}/{total}] ERRO: {e}")
            rows.append({"profile_id": i, "lp_status": "error",
                         "stratum": stratum, **profile})

    df = pd.DataFrame(rows)
    os.makedirs("tmp_who", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResultados salvos em: {OUT_CSV}")

    # ── resumo ────────────────────────────────────────────────────────────────
    ok = df[df["lp_status"] != "error"]
    nutrients = ["energy_pct","protein_pct","iron_pct","calcium_pct",
                 "zinc_pct","vitA_pct","vitC_pct"]

    print(f"\n{'='*55}")
    print(f"{'RESUMO':^55}")
    print(f"{'='*55}")
    print(f"  Total rodados:  {len(df)}")
    print(f"  Erros:          {(df['lp_status'] == 'error').sum()}")
    print(f"\n  Status LP:")
    for s, cnt in ok["lp_status"].value_counts().items():
        print(f"    {s:<12}: {cnt:>3}  ({cnt/len(ok)*100:.1f}%)")

    print(f"\n  {'Nutriente':<14}  {'Mediana %DRI':>13}  {'P10':>7}  {'P90':>7}  {'<100% n':>8}")
    print(f"  {'-'*54}")
    for nut in nutrients:
        col = ok[nut].dropna()
        if col.empty:
            continue
        below = int((col < 100).sum())
        print(f"  {nut:<14}  {col.median():>13.1f}  "
              f"{col.quantile(0.10):>7.1f}  {col.quantile(0.90):>7.1f}  {below:>8}")

    print(f"\n  NAS por faixa etária:")
    for band in AGE_BANDS:
        nas = ok[ok["stratum"]==band]["nas_wavg"].dropna()
        print(f"    {band}: mean={nas.mean():.4f}  median={nas.median():.4f}")


if __name__ == "__main__":
    run()
