"""
4.1 Assessment Agent Validation (TCC Section 4.1)

Validates the NUMERICAL CORRECTNESS of the Assessment Agent's z-score computation
by comparing agent output against ENANI-2019 pre-computed reference z-scores.

ENANI-2019 protocol:
  - First measurement only: s02_peso_medida1, s07_altura_medida1
  - WHO 2006 MGRS LMS formula
  - Reference z-scores stored at 1 decimal place precision (vd_zwaz, vd_zhaz, vd_anthro_zwfl)

Design:
  Ground truth: ENANI pre-computed reference z-scores (same WHO 2006 standard)
  Filter: rows where any reference z-score is 0.0 or NaN (missing or ambiguous)
  Threshold: MAE <= 0.10 (1dp rounding floor ≈ 0.025; threshold gives generous tolerance)

Usage:
    python scripts/validate_assessment_agent.py
    python scripts/validate_assessment_agent.py --n 500 --seed 0
"""
import sys
import os
import argparse

import pandas as pd
import numpy as np

GAIA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GAIA_ROOT)

from agents.assessment_agent import AssessmentAgent

ENANI_PATH    = os.path.join(GAIA_ROOT, "data", "raw", "enani_profiles.csv")
MAE_THRESHOLD = 0.10


def load_profiles(path: str) -> pd.DataFrame:
    print(f"Loading ENANI profiles from {path} ...")
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    print(f"  Total rows: {len(df)}")

    # Exact age in days → fractional months for LMS interpolation
    age_days = pd.to_numeric(df["s00d_idade_peso"], errors="coerce")
    df["age_months"] = age_days / 30.4375

    df["sex"] = df["b02_sexo"].map({"Masculino": "M", "Feminino": "F"})

    # ENANI uses the first measurement for z-score computation
    df["weight_kg"] = pd.to_numeric(df["s02_peso_medida1"], errors="coerce")
    df["height_cm"] = pd.to_numeric(df["s07_altura_medida1"], errors="coerce")

    # ENANI reference z-scores (1dp precision)
    df["ref_waz"] = pd.to_numeric(df["vd_zwaz"],        errors="coerce")
    df["ref_haz"] = pd.to_numeric(df["vd_zhaz"],        errors="coerce")
    df["ref_whz"] = pd.to_numeric(df["vd_anthro_zwfl"], errors="coerce")

    required = ["age_months", "sex", "weight_kg", "height_cm",
                "ref_waz", "ref_haz", "ref_whz"]
    valid = df.dropna(subset=required).copy()

    # Filter rows where ENANI stores 0.0 for missing/invalid z-scores
    valid = valid[
        (valid["ref_waz"] != 0.0) &
        (valid["ref_haz"] != 0.0) &
        (valid["ref_whz"] != 0.0)
    ]

    valid = valid[
        valid["age_months"].between(0, 60) &
        (valid["weight_kg"] > 0) &
        (valid["height_cm"] > 0)
    ]

    print(f"  Usable profiles (non-zero ENANI reference): {len(valid)}")
    return valid


def run_validation(n: int, seed: int) -> bool:
    profiles = load_profiles(ENANI_PATH)
    sample   = profiles.sample(n=min(n, len(profiles)), random_state=seed)
    print(f"\nSampled {len(sample)} profiles (N={n}, seed={seed}).\n")

    agent   = AssessmentAgent()
    records = []

    for _, row in sample.iterrows():
        child = {
            "age_months": float(row["age_months"]),
            "sex":        row["sex"],
            "weight_kg":  float(row["weight_kg"]),
            "height_cm":  float(row["height_cm"]),
        }
        result  = agent.assess(child)
        z_agent = result.z_scores

        records.append({
            "age_months": child["age_months"],
            "sex":        child["sex"],
            "weight_kg":  child["weight_kg"],
            "height_cm":  child["height_cm"],
            "agent_waz":  z_agent.WAZ,
            "agent_haz":  z_agent.HAZ,
            "agent_whz":  z_agent.WHZ,
            "ref_waz":    float(row["ref_waz"]),
            "ref_haz":    float(row["ref_haz"]),
            "ref_whz":    float(row["ref_whz"]),
        })

    df = pd.DataFrame(records)

    print("--- 4.1 Assessment Agent Validation (vs ENANI-2019) ---")
    print(f"{'Indicator':<10} {'MAE':>8} {'Max AE':>10} {'N':>6} {'Pass?':>8}")
    print("-" * 48)

    all_pass = True
    for ind in ["waz", "haz", "whz"]:
        sub = df.dropna(subset=[f"agent_{ind}", f"ref_{ind}"])
        if sub.empty:
            print(f"{ind.upper():<10} {'n/a':>8}")
            continue
        ae   = (sub[f"agent_{ind}"] - sub[f"ref_{ind}"]).abs()
        mae  = ae.mean()
        mxae = ae.max()
        ok   = mae <= MAE_THRESHOLD
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"{ind.upper():<10} {mae:>8.4f} {mxae:>10.4f} {len(sub):>6} {status:>8}")

    print("=" * 48)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}  (threshold = {MAE_THRESHOLD})")

    for ind in ["waz", "haz", "whz"]:
        sub = df.dropna(subset=[f"agent_{ind}", f"ref_{ind}"]).copy()
        if sub.empty:
            continue
        sub["ae"] = (sub[f"agent_{ind}"] - sub[f"ref_{ind}"]).abs()
        if sub["ae"].mean() > MAE_THRESHOLD:
            print(f"\nTop-5 worst {ind.upper()} errors:")
            for _, r in sub.nlargest(5, "ae").iterrows():
                print(f"  age={r['age_months']:>5.1f}m sex={r['sex']} "
                      f"w={r['weight_kg']:.2f}kg h={r['height_cm']:.1f}cm  "
                      f"agent={r[f'agent_{ind}']:.4f}  ref={r[f'ref_{ind}']:.4f}  "
                      f"ae={r['ae']:.4f}")

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Assessment Agent z-scores vs ENANI-2019")
    parser.add_argument("--n",    type=int, default=200, help="Number of profiles (default: 200)")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed (default: 42)")
    args = parser.parse_args()

    ok = run_validation(n=args.n, seed=args.seed)
    sys.exit(0 if ok else 1)
