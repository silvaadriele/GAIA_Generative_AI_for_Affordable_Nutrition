"""
Gera os 2 reports do Apêndice do artigo com use_gemini=True.

Perfis selecionados manualmente:
  app1 — 2.9m M Sudeste  (0-5m, AME, SAM severo)
  app2 — 11.2m M CO      (9-11m, plano alimentar, estado normal)

Saída: evaluation/appendix_artigo/
"""

from __future__ import annotations

import os, sys, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.gaia import GAIA

OUT_DIR = "evaluation/appendix_artigo"

PROFILES = [
    {
        "label":  "app1_2m_M_SE",
        "profile": {
            "age_months": 2.9,
            "sex":        "M",
            "weight_kg":  3.76,
            "height_cm":  54.0,
            "region":     "Sudeste",
        },
    },
    {
        "label":  "app2_11m_M_CO",
        "profile": {
            "age_months": 11.2,
            "sex":        "M",
            "weight_kg":  9.99,
            "height_cm":  78.0,
            "region":     "Centro-Oeste",
        },
    },
    {
        "label":  "app3_44m_M_SE",
        "profile": {
            "age_months": 43.6,
            "sex":        "M",
            "weight_kg":  14.6,
            "height_cm":  99.8,
            "region":     "Sudeste",
        },
    },
]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    gaia = GAIA(use_gemini=True)

    print(f"Gerando {len(PROFILES)} reports com Gemini...\n")

    for i, entry in enumerate(PROFILES, 1):
        label   = entry["label"]
        profile = entry["profile"]
        fpath   = os.path.join(OUT_DIR, f"{label}.txt")

        print(f"  [{i}/{len(PROFILES)}] {profile['age_months']}m {profile['sex']} "
              f"{profile['region']} ...")

        try:
            result = gaia.run(profile, verbose=False)
            report_text = (
                result.report.text
                if hasattr(result.report, "text")
                else str(result.report)
            )
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"    -> {fpath}")
        except Exception as e:
            print(f"    ERRO: {e}")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"ERRO:\n{e}\n")

    print(f"\nConcluído. Reports em: {OUT_DIR}/")


if __name__ == "__main__":
    run()
