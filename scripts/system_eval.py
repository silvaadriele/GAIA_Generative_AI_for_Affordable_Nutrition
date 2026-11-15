"""
Avaliação sistêmica do GAIA (Seção 4.6) — 50 reports para inspeção manual.

  5  perfis de 0-5 meses  → espera recomendação de AME (sem plano)
  45 perfis de 6-59 meses → espera plano alimentar completo (9 por faixa etária)

Saídas: evaluation/system_eval/
  NN_Xm_SEX_REG.txt  — report completo gerado pelo GAIA
  summary.csv        — lp_status, n_foods, tipo por perfil
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
OUT_DIR   = "evaluation/system_eval"

AGE_BANDS = ["6-8m", "9-11m", "12-23m", "24-35m", "36-59m"]
AGE_BINS  = [5.5, 8.5, 11.5, 23.5, 35.5, 60]

REGION_ABBR = {
    "Norte": "N", "Nordeste": "NE", "Centro-Oeste": "CO",
    "Sudeste": "SE", "Sul": "S",
}

N_AME      = 5   # perfis 0-5m
N_PER_BAND = 9   # perfis por faixa (9 × 5 = 45)


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


def sample_ame(df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    sub = df[df["age_months"].between(0, 5.99)].copy()
    sub["stratum"] = "0-5m"
    if len(sub) < n:
        print(f"  AVISO: apenas {len(sub)} perfis 0-5m no ENANI — usando todos")
        return sub
    return sub.sample(n=n, random_state=int(rng.integers(0, 99999)))


def sample_plan(df: pd.DataFrame, n_per_band: int, rng) -> pd.DataFrame:
    sub = df[df["age_months"].between(6, 60)].copy()
    sub["stratum"] = pd.cut(sub["age_months"], bins=AGE_BINS, labels=AGE_BANDS).astype(str)
    rows = []
    for band in AGE_BANDS:
        pool = sub[sub["stratum"] == band]
        if len(pool) < n_per_band:
            print(f"  AVISO: {band} tem só {len(pool)} perfis")
            rows.append(pool)
        else:
            rows.append(pool.sample(n=n_per_band, random_state=int(rng.integers(0, 99999))))
    return pd.concat(rows, ignore_index=True)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    enani = load_enani()
    print(f"ENANI carregado: {len(enani)} perfis com medidas completas")

    ame   = sample_ame(enani, N_AME, rng)
    plan  = sample_plan(enani, N_PER_BAND, rng)
    all_p = pd.concat([ame, plan], ignore_index=True)
    print(f"Perfis selecionados: {len(all_p)}  (AME: {len(ame)} | Plano: {len(plan)})\n")

    gaia = GAIA(use_gemini=False)
    summary = []

    for i, (_, row) in enumerate(all_p.iterrows(), 1):
        profile = {
            "age_months": round(float(row["age_months"]), 1),
            "sex":        str(row["sex"]),
            "weight_kg":  round(float(row["weight_kg"]), 2),
            "height_cm":  round(float(row["height_cm"]), 1),
            "region":     str(row["region"]),
        }
        stratum     = str(row["stratum"])
        region_abbr = REGION_ABBR.get(profile["region"], profile["region"][:2])
        fname       = f"{i:02d}_{profile['age_months']:.0f}m_{profile['sex']}_{region_abbr}.txt"
        fpath       = os.path.join(OUT_DIR, fname)

        try:
            result = gaia.run(profile, verbose=False)

            report_text = (
                result.report.text
                if hasattr(result.report, "text")
                else str(result.report)
            )

            header = (
                f"{'='*68}\n"
                f"  Perfil {i:02d}  |  {profile['age_months']:.1f} meses  |  "
                f"Sexo: {profile['sex']}  |  Região: {profile['region']}\n"
                f"  Peso: {profile['weight_kg']} kg  |  Altura: {profile['height_cm']} cm"
                f"  |  Faixa: {stratum}\n"
                f"{'='*68}\n\n"
            )

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(header + report_text)

            if profile["age_months"] < 6:
                row_s = {"file": fname, "stratum": stratum, "type": "AME",
                         "lp_status": "n/a", "n_foods": 0, "daily_cost_brl": None,
                         "approved": None}
            else:
                plan_obj = result.plan
                validation_obj = result.validation
                approved = validation_obj.overall_pass if validation_obj else None
                row_s = {"file": fname, "stratum": stratum, "type": "PLAN",
                         "lp_status": plan_obj.lp_status,
                         "n_foods": sum(len(m.items) for m in plan_obj.meals),
                         "daily_cost_brl": getattr(plan_obj, "daily_cost", None),
                         "approved": approved}

            row_s.update({"age_months": profile["age_months"],
                          "sex": profile["sex"], "region": profile["region"]})
            summary.append(row_s)

            status = row_s["lp_status"]
            approved_str = " APPROVED" if row_s.get("approved") else (" REJECTED" if row_s.get("approved") is False else "")
            cost_str = f"  R${row_s['daily_cost_brl']:.2f}" if row_s.get("daily_cost_brl") is not None else ""
            print(f"  [{i:02d}/50] {profile['age_months']:>4.1f}m {profile['sex']} "
                  f"{profile['region']:<14}  faixa={stratum:<7}  -> {status}{approved_str}{cost_str}")

        except Exception as e:
            print(f"  [{i:02d}/50] ERRO: {e}")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"ERRO ao gerar report:\n{e}\n")
            summary.append({"file": fname, "stratum": stratum, "type": "ERROR",
                             "lp_status": "error", "n_foods": 0,
                             "age_months": profile["age_months"],
                             "sex": profile["sex"], "region": profile["region"]})

    # ── resumo ────────────────────────────────────────────────────────────────
    df_s = pd.DataFrame(summary)
    df_s.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    n_ame      = (df_s["type"] == "AME").sum()
    n_ok       = ((df_s["type"] == "PLAN") & (df_s["lp_status"] == "optimal")).sum()
    n_approved = ((df_s["type"] == "PLAN") & (df_s["approved"] == True)).sum()
    n_err      = (df_s["type"] == "ERROR").sum()

    cost_plans = df_s[(df_s["type"] == "PLAN") & df_s["daily_cost_brl"].notna()]["daily_cost_brl"]

    print(f"\n{'='*55}")
    print(f"  Reports em: {OUT_DIR}/")
    print(f"  AME (0-5m)  : {n_ame}/5")
    print(f"  Planos OK   : {n_ok}/45  (LP optimal)")
    print(f"  Aprovados   : {n_approved}/45  (validação)")
    print(f"  Erros       : {n_err}")
    if len(cost_plans) > 0:
        print(f"\n  -- Custo diario estimado (R$) --")
        print(f"  n={len(cost_plans)}  media={cost_plans.mean():.2f}  dp={cost_plans.std():.2f}")
        print(f"  min={cost_plans.min():.2f}  med={cost_plans.median():.2f}  max={cost_plans.max():.2f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()
