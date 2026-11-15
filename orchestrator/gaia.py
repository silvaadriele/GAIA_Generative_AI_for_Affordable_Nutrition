"""
Orquestrador GAIA (Fase 9) — ponto de entrada do sistema multi-agente.

Pipeline:
  1. AssessmentAgent    → z-scores OMS + DRI ajustadas
  2. FoodRetrieverAgent → ranking de preferência (LightGBM)
  3. MealPlannerAgent   → plano de refeições via LP
  4. NutritionalValidator → validação nutricional
  5. ReportAgent        → relatório texto + JSON (+ Gemini opcional)

Uso rápido:
    python orchestrator/gaia.py \
        --age 18 --sex M --weight 6.5 --height 75.0 --region Nordeste

Uso como módulo:
    from orchestrator.gaia import GAIA
    result = GAIA().run({"age_months": 18, "sex": "M",
                         "weight_kg": 6.5, "height_cm": 75.0,
                         "region": "Nordeste"})
    print(result.report.text)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Optional

warnings.filterwarnings("ignore")

import sys
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from agents.assessment_agent    import AssessmentAgent, AssessmentResult
from agents.food_retriever_agent import FoodRetrieverAgent, FoodCatalog
from agents.meal_planner_agent  import MealPlannerAgent, MealPlan
from agents.report_agent        import ReportAgent, Report
from modules.validator          import NutritionalValidator, ValidationReport

OUTPUT_DIR = os.path.join(_BASE, "evaluation", "results")


# ── resultado completo ────────────────────────────────────────────────────────
@dataclass
class GaiaResult:
    assessment:  AssessmentResult
    catalog:     Optional[FoodCatalog]
    plan:        Optional[MealPlan]
    validation:  Optional[ValidationReport]
    report:      Report
    elapsed_s:   float

    def save(self, output_dir: str = OUTPUT_DIR, prefix: str = "gaia") -> dict:
        """Salva relatório .txt e dados .json no diretório de saída."""
        os.makedirs(output_dir, exist_ok=True)
        txt_path  = os.path.join(output_dir, f"{prefix}_report.txt")
        json_path = os.path.join(output_dir, f"{prefix}_report.json")
        self.report.save(txt_path)
        self.report.save_json(json_path)
        return {"txt": txt_path, "json": json_path}


# ── orquestrador ──────────────────────────────────────────────────────────────
class GAIA:
    """Sistema multi-agente para planejamento nutricional infantil."""

    def __init__(self, top_k: int = 50, use_gemini: bool = True):
        print("[GAIA] Carregando agentes...")
        t0 = time.time()
        self._assessment = AssessmentAgent()
        self._retriever  = FoodRetrieverAgent(top_k=top_k)
        self._planner    = MealPlannerAgent(top_k=top_k)
        self._validator  = NutritionalValidator()
        self._reporter   = ReportAgent()
        self._use_gemini = use_gemini
        print(f"[GAIA] Agentes prontos em {time.time()-t0:.1f}s")

    def _breastfeeding_report(self, assessment: AssessmentResult) -> Report:
        """Retorna recomendação de aleitamento materno exclusivo para crianças < 6 meses."""
        cp  = assessment.child_profile
        z   = assessment.z_scores
        st  = assessment.status
        age = cp.get("age_months")

        text = "\n".join([
            "=" * 68,
            "  GAIA — PEDIATRIC NUTRITIONAL RECOMMENDATION",
            f"  Date: {date.today().isoformat()}",
            "=" * 68,
            "",
            "── CHILD PROFILE ────────────────────────────────────────────────",
            f"  Age:    {age} months  |  Sex: {'Male' if cp.get('sex') == 'M' else 'Female'}",
            f"  Weight: {cp.get('weight_kg')} kg  |  Height: {cp.get('height_cm', 'N/A')} cm",
            f"  Region: {cp.get('region', 'N/A')}",
            "",
            "── ANTHROPOMETRIC ASSESSMENT (WHO z-scores) ────────────────────",
            f"  WAZ (weight-for-age):    {z.WAZ if z.WAZ is not None else 'N/A':>6}  → {st.waz_class}",
            f"  HAZ (height-for-age):    {z.HAZ if z.HAZ is not None else 'N/A':>6}  → {st.haz_class}",
            f"  WHZ (weight-for-height): {z.WHZ if z.WHZ is not None else 'N/A':>6}  → {st.whz_class}",
            f"  NUTRITIONAL CLASSIFICATION: {st.classification}",
            "",
            "── RECOMMENDATION ───────────────────────────────────────────────",
            "  This child is under 6 months of age.",
            "",
            "  According to World Health Organization guidelines (WHO, 2023),",
            "  children aged 0 to 6 months should receive EXCLUSIVE",
            "  BREASTFEEDING. No solid, semi-solid, or liquid food —",
            "  except medicines and supplements when clinically indicated —",
            "  should be introduced before 6 months of age.",
            "",
            "  Reference:",
            "  WHO (2023). Infant and young child feeding.",
            "  https://www.who.int/publications/i/item/9789240082830",
            "",
            "  GAIA generates meal plans only for children >= 6 months.",
            "=" * 68,
        ])

        return Report(
            text=text,
            data={
                "age_months":          age,
                "recommendation":      "exclusive_breastfeeding",
                "guideline":           "WHO (2023). Infant and young child feeding.",
                "nutritional_status":  st.classification,
                "z_scores": {
                    "WAZ": z.WAZ,
                    "HAZ": z.HAZ,
                    "WHZ": z.WHZ,
                },
            },
            source="template",
        )

    def run(self, child_profile: dict, verbose: bool = True) -> GaiaResult:
        """Executa o pipeline completo para um perfil de criança."""
        t0 = time.time()

        if verbose:
            age  = child_profile.get("age_months")
            sex  = child_profile.get("sex")
            wt   = child_profile.get("weight_kg")
            ht   = child_profile.get("height_cm", "N/A")
            reg  = child_profile.get("region", "N/A")
            print(f"\n[1/5] Avaliando criança: {age}m {sex} {wt}kg {ht}cm {reg}")

        assessment = self._assessment.assess(child_profile)
        if verbose:
            z  = assessment.z_scores
            st = assessment.status
            print(f"      WAZ={z.WAZ} HAZ={z.HAZ} WHZ={z.WHZ} -> {st.classification}")

        if child_profile.get("age_months", 0) < 6:
            if verbose:
                print("[GAIA] Criança < 6 meses — recomendando aleitamento materno exclusivo (WHO 2023).")
            report  = self._breastfeeding_report(assessment)
            elapsed = round(time.time() - t0, 2)
            return GaiaResult(
                assessment=assessment,
                catalog=None,
                plan=None,
                validation=None,
                report=report,
                elapsed_s=elapsed,
            )

        if verbose:
            print("[2/5] Recuperando alimentos preferidos...")
        catalog = self._retriever.retrieve(assessment)
        if verbose:
            print(f"      {len(catalog.top_k)} candidatos top-K  |  "
                  f"{len(catalog.by_category)} categorias")

        if verbose:
            print("[3/5] Planejando refeições (LP)...")
        plan = self._planner.plan(assessment, catalog)
        if verbose:
            print(f"      LP: {plan.lp_status}  |  "
                  f"{plan.total_g_per_day}g/dia  |  "
                  f"{plan.daily_achieved.get('energy_kcal',0):.0f} kcal atingidos")

        if verbose:
            print("[4/5] Validando plano nutricional...")
        validation = self._validator.validate(plan)
        n_ok = sum(1 for c in validation.criteria if c.passed)
        n_total = len(validation.criteria)
        if verbose:
            status = "APROVADO" if validation.overall_pass else "REPROVADO"
            print(f"      {status}: {n_ok}/{n_total} criterios OK")

        if verbose:
            print("[5/5] Gerando relatório...")
        report = self._reporter.generate(
            assessment, plan, validation, use_gemini=self._use_gemini
        )
        if verbose:
            print(f"      Fonte: {report.source}")

        elapsed = round(time.time() - t0, 2)
        if verbose:
            print(f"\n[GAIA] Pipeline concluido em {elapsed}s")

        return GaiaResult(
            assessment=assessment,
            catalog=catalog,
            plan=plan,
            validation=validation,
            report=report,
            elapsed_s=elapsed,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="GAIA — Planejamento Nutricional Pediátrico"
    )
    p.add_argument("--age",     type=float, required=True,  help="Idade em meses")
    p.add_argument("--sex",     type=str,   required=True,  help="Sexo: M ou F")
    p.add_argument("--weight",  type=float, required=True,  help="Peso em kg")
    p.add_argument("--height",  type=float, default=None,   help="Altura em cm")
    p.add_argument("--region",  type=str,   default=None,   help="Região do Brasil")
    p.add_argument("--out",     type=str,   default=OUTPUT_DIR, help="Diretório de saída")
    p.add_argument("--no-gemini", action="store_true", help="Desativar Gemini")
    p.add_argument("--top-k",  type=int,   default=50,     help="Top-K alimentos")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    child = {
        "age_months": args.age,
        "sex":        args.sex.upper(),
        "weight_kg":  args.weight,
    }
    if args.height:
        child["height_cm"] = args.height
    if args.region:
        child["region"] = args.region

    gaia   = GAIA(top_k=args.top_k, use_gemini=not args.no_gemini)
    result = gaia.run(child)
    paths  = result.save(output_dir=args.out)

    print(f"\nRelatório salvo em:")
    print(f"  {paths['txt']}")
    print(f"  {paths['json']}")
