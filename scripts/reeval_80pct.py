"""
Re-avalia os 100 perfis do batch original com o fallback de 80%
e salva um CSV comparativo em tmp_who/batch_results_80pct.csv.
"""

import os
import sys
import warnings
import json

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.gaia import GAIA

BATCH_CSV  = "tmp_who/batch_results.csv"
OUT_CSV    = "tmp_who/batch_results_80pct.csv"
OUT_COMP   = "tmp_who/comparison_70_vs_80.csv"

def pct(achieved, target):
    return round(achieved / target * 100, 1) if target and target > 0 else None

def run():
    profiles_df = pd.read_csv(BATCH_CSV)
    print(f"Perfis carregados: {len(profiles_df)}")

    gaia = GAIA(use_gemini=False)

    rows = []
    for _, row in profiles_df.iterrows():
        profile = {
            "age_months": float(row["age_months"]),
            "sex":        str(row["sex"]),
            "weight_kg":  float(row["weight_kg"]),
            "height_cm":  float(row["height_cm"]),
            "region":     str(row["region"]),
        }
        pid = int(row["profile_id"])
        try:
            result = gaia.run(profile, verbose=False)
            plan   = result.plan
            ach    = plan.daily_achieved
            tgt    = plan.daily_targets

            rows.append({
                "profile_id":  pid,
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
                "nas_wavg":    row.get("nas_wavg", None),
            })
            status_tag = f"[{plan.lp_status}]" if plan.lp_status != "optimal" else ""
            print(f"  [{pid:3d}] {profile['age_months']}m {profile['sex']} {profile['region']} "
                  f"-> {plan.lp_status} {status_tag}")
        except Exception as e:
            print(f"  [{pid:3d}] ERRO: {e}")
            rows.append({"profile_id": pid, "lp_status": "error"})

    new_df = pd.DataFrame(rows)
    new_df.to_csv(OUT_CSV, index=False)
    print(f"\nResultados salvos em: {OUT_CSV}")

    # ── comparativo ──────────────────────────────────────────────────────────
    old_df = profiles_df.copy()
    old_df = old_df.rename(columns={c: f"{c}_70" for c in old_df.columns if c not in ("profile_id",)})
    new_df2 = new_df.rename(columns={c: f"{c}_80" for c in new_df.columns if c not in ("profile_id",)})

    comp = old_df.merge(new_df2, on="profile_id")
    comp.to_csv(OUT_COMP, index=False)
    print(f"Comparativo salvo em:  {OUT_COMP}")

    # ── resumo ───────────────────────────────────────────────────────────────
    nutrients = ["energy_pct", "protein_pct", "iron_pct", "calcium_pct",
                 "zinc_pct", "vitA_pct", "vitC_pct"]
    print("\n─── Resumo comparativo (mediana % DRI) ───")
    print(f"{'Nutriente':<14}  {'70% (orig)':>12}  {'80% (novo)':>12}  {'Δ':>8}")
    for nut in nutrients:
        c70 = profiles_df[nut].median()
        c80 = new_df[nut].dropna().median()
        print(f"  {nut:<14}  {c70:>12.1f}  {c80:>12.1f}  {c80-c70:>+8.1f}")

    print("\n─── Status LP ───")
    for label, df_ in [("70% (orig)", profiles_df), ("80% (novo)", new_df)]:
        counts = df_["lp_status"].value_counts().to_dict()
        print(f"  {label}: {counts}")


if __name__ == "__main__":
    run()
