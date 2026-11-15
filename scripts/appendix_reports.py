"""
Gera os 6 reports do Apêndice com use_gemini=True.

Perfis selecionados (mesma amostragem SEED=42 do system_eval.py):
  01 —  5.4m F Sul          → AME (0-5m)
  06 —  6.5m F Sul          → 6-8m
  15 — 11.2m M Centro-Oeste → 9-11m
  24 — 18.6m F Centro-Oeste → 12-23m
  33 — 30.9m F Sul          → 24-35m
  45 — 57.2m M Nordeste     → 36-59m

Saída: evaluation/appendix/
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
OUT_DIR   = "evaluation/appendix"

AGE_BANDS = ["6-8m", "9-11m", "12-23m", "24-35m", "36-59m"]
AGE_BINS  = [5.5, 8.5, 11.5, 23.5, 35.5, 60]

REGION_ABBR = {
    "Norte": "N", "Nordeste": "NE", "Centro-Oeste": "CO",
    "Sudeste": "SE", "Sul": "S",
}

# Índices (0-based) dos 6 perfis dentro do all_profiles do system_eval
SELECTED = [0, 5, 14, 23, 32, 44]


def load_enani() -> pd.DataFrame:
    df = pd.read_csv(ENANI_CSV, low_memory=False)
    out = pd.DataFrame({
        "age_months": pd.to_numeric(df["s00d_idade_peso"],    errors="coerce") / 30.4375,
        "weight_kg":  pd.to_numeric(df["s02_peso_medida1"],   errors="coerce"),
        "height_cm":  pd.to_numeric(df["s07_altura_medida1"], errors="coerce"),
        "sex":        df["b02_sexo"].map({"Masculino": "M", "Feminino": "F"}),
        "region":     df["a00_regiao"],
    })
    return out.dropna().query("weight_kg > 0 and height_cm > 0").reset_index(drop=True)


def sample_all(df: pd.DataFrame, rng) -> pd.DataFrame:
    # 0-5m
    sub_ame = df[df["age_months"].between(0, 5.99)].copy()
    sub_ame["stratum"] = "0-5m"
    ame = sub_ame.sample(n=5, random_state=int(rng.integers(0, 99999)))

    # 6-59m — 9 por faixa
    sub_plan = df[df["age_months"].between(6, 60)].copy()
    sub_plan["stratum"] = pd.cut(
        sub_plan["age_months"], bins=AGE_BINS, labels=AGE_BANDS
    ).astype(str)
    rows = []
    for band in AGE_BANDS:
        pool = sub_plan[sub_plan["stratum"] == band]
        rows.append(pool.sample(n=9, random_state=int(rng.integers(0, 99999))))
    plan = pd.concat(rows, ignore_index=True)

    return pd.concat([ame, plan], ignore_index=True)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng   = np.random.default_rng(SEED)
    enani = load_enani()

    all_p    = sample_all(enani, rng)
    selected = all_p.iloc[SELECTED].reset_index(drop=True)

    print(f"Gerando {len(selected)} reports com Gemini...\n")
    gaia = GAIA(use_gemini=True)

    for i, (_, row) in enumerate(selected.iterrows(), 1):
        profile = {
            "age_months": round(float(row["age_months"]), 1),
            "sex":        str(row["sex"]),
            "weight_kg":  round(float(row["weight_kg"]), 2),
            "height_cm":  round(float(row["height_cm"]), 1),
            "region":     str(row["region"]),
        }
        stratum     = str(row["stratum"])
        region_abbr = REGION_ABBR.get(profile["region"], profile["region"][:2])
        fname       = f"ap{i:02d}_{profile['age_months']:.0f}m_{profile['sex']}_{region_abbr}.txt"
        fpath       = os.path.join(OUT_DIR, fname)

        print(f"  [{i}/6] {profile['age_months']:.1f}m {profile['sex']} "
              f"{profile['region']:<14}  faixa={stratum}  ...")

        try:
            result = gaia.run(profile, verbose=False)
            report_text = (
                result.report.text
                if hasattr(result.report, "text")
                else str(result.report)
            )

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(report_text)

            print(f"    -> salvo: {fname}")

        except Exception as e:
            print(f"    ERRO: {e}")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"ERRO:\n{e}\n")

    print(f"\nConcluído. Reports em: {OUT_DIR}/")


if __name__ == "__main__":
    run()
