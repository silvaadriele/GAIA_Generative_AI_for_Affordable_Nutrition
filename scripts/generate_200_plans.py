"""
Gera 200 planos a partir de perfis ENANI (6-59 meses, seed=99).

Saídas:
  tmp/plans_200_foods.json   — {food_name: {price, label, plans_count}}
  tmp/plans_200_summary.csv  — um registro por plano (lp_status, custo, etc.)

Propósito: mapear todos os food_names únicos para expandir food_prices.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.food_prices import get_price, get_price_label, _DEFAULT_PRICE_PER_100G
from orchestrator.gaia import GAIA

SEED    = 99
N_PLANS = 200
OUT_DIR = "tmp"
ENANI   = "data/raw/enani_profiles.csv"

REGION_ABBR = {
    "Norte": "N", "Nordeste": "NE", "Centro-Oeste": "CO",
    "Sudeste": "SE", "Sul": "S",
}


def load_enani() -> pd.DataFrame:
    df = pd.read_csv(ENANI, low_memory=False)
    out = pd.DataFrame({
        "age_months": pd.to_numeric(df["s00d_idade_peso"],    errors="coerce") / 30.4375,
        "weight_kg":  pd.to_numeric(df["s02_peso_medida1"],   errors="coerce"),
        "height_cm":  pd.to_numeric(df["s07_altura_medida1"], errors="coerce"),
        "sex":        df["b02_sexo"].map({"Masculino": "M", "Feminino": "F"}),
        "region":     df["a00_regiao"],
    })
    return out.dropna().query(
        "age_months >= 6 and age_months < 60 and weight_kg > 0 and height_cm > 0"
    ).reset_index(drop=True)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    enani = load_enani()
    print(f"Pool disponivel: {len(enani)} perfis 6-59m")

    sample = enani.sample(n=N_PLANS, random_state=int(rng.integers(0, 99999)))
    print(f"Amostra: {N_PLANS} perfis (seed={SEED})\n")

    gaia = GAIA(use_gemini=False)

    food_plans_count: Counter = Counter()   # food_name -> quantos planos incluiram
    food_price_cache: dict = {}             # food_name -> (price, label)
    summary_rows = []

    n_ok = 0
    n_err = 0

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        profile = {
            "age_months": round(float(row["age_months"]), 1),
            "sex":        str(row["sex"]),
            "weight_kg":  round(float(row["weight_kg"]), 2),
            "height_cm":  round(float(row["height_cm"]), 1),
            "region":     str(row["region"]),
        }

        try:
            result = gaia.run(profile, verbose=False)
            plan = result.plan
            if plan is None:
                continue   # < 6m, nao deve ocorrer dado o filtro

            foods_in_plan = []
            for meal in plan.meals:
                for item in meal.items:
                    name = item.food_name
                    food_plans_count[name] += 1
                    foods_in_plan.append(name)
                    if name not in food_price_cache:
                        food_price_cache[name] = (
                            get_price(name),
                            get_price_label(name),
                        )

            n_ok += 1
            summary_rows.append({
                "idx":         i,
                "age_months":  profile["age_months"],
                "sex":         profile["sex"],
                "region":      profile["region"],
                "lp_status":   plan.lp_status,
                "approved":    result.validation.overall_pass if result.validation else None,
                "daily_cost":  getattr(plan, "daily_cost", None),
                "n_foods":     len(foods_in_plan),
            })

            if i % 20 == 0 or i == N_PLANS:
                print(f"  [{i:03d}/{N_PLANS}]  foods unicos ate agora: {len(food_plans_count)}")

        except Exception as e:
            n_err += 1
            print(f"  [{i:03d}/{N_PLANS}] ERRO: {e}")

    # ── salvar summary ────────────────────────────────────────────────────────
    df_s = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "plans_200_summary.csv")
    df_s.to_csv(summary_path, index=False)

    # ── salvar mapa de alimentos ──────────────────────────────────────────────
    foods_out = {}
    for name, count in food_plans_count.most_common():
        price, label = food_price_cache[name]
        foods_out[name] = {
            "price_per_100g": price,
            "price_label":    label,
            "plans_count":    count,
            "uses_default":   abs(price - _DEFAULT_PRICE_PER_100G) < 0.01,
        }

    foods_path = os.path.join(OUT_DIR, "plans_200_foods.json")
    with open(foods_path, "w", encoding="utf-8") as f:
        json.dump(foods_out, f, ensure_ascii=False, indent=2)

    # ── alimentos sem preco especifico ────────────────────────────────────────
    default_foods = [
        (name, d["plans_count"])
        for name, d in foods_out.items()
        if d["uses_default"]
    ]
    default_foods.sort(key=lambda x: -x[1])  # mais frequentes primeiro

    costs = df_s["daily_cost"].dropna()
    n_approved = df_s["approved"].sum()

    print(f"\n{'='*58}")
    print(f"  Planos gerados : {n_ok}/{N_PLANS}")
    print(f"  Aprovados      : {n_approved:.0f}/{n_ok}")
    print(f"  Erros          : {n_err}")
    print(f"  Foods unicos   : {len(food_plans_count)}")
    print(f"  Custo medio    : R${costs.mean():.2f} (dp={costs.std():.2f})")
    print(f"  Custo min/max  : R${costs.min():.2f} / R${costs.max():.2f}")
    print(f"\n  Foods com preco PADRAO (R$2.00) — {len(default_foods)} itens:")
    for name, cnt in default_foods:
        print(f"    {cnt:3d} planos  {name[:70]}")
    print(f"\n  Arquivos salvos em {OUT_DIR}/")
    print(f"{'='*58}")


if __name__ == "__main__":
    run()
