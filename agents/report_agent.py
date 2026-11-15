"""
Agente de Geração de Relatório (Fase 8 — GAIA).

Gera relatório nutricional completo em dois modos:
  1. Template (sempre disponível) — relatório estruturado em texto + JSON
  2. Gemini (requer GEMINI_API_KEY no .env) — narrativa em linguagem natural

Uso:
    from agents.report_agent import ReportAgent
    agent  = ReportAgent()
    report = agent.generate(assessment_result, meal_plan, validation_report)
    print(report.text)             # relatório textual
    report.save("output/report.txt")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

from agents.assessment_agent import AssessmentResult
from agents.meal_planner_agent import MealPlan
from modules.validator import ValidationReport

# ── Gemini opcional ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai
    _GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    _GEMINI_OK  = bool(_GEMINI_KEY)
    if _GEMINI_OK:
        genai.configure(api_key=_GEMINI_KEY)
except ImportError:
    _GEMINI_OK = False


# ── tipos ─────────────────────────────────────────────────────────────────────
@dataclass
class Report:
    text:     str          # relatório em texto plano
    data:     dict         # dados estruturados (serializável em JSON)
    source:   str          # "template" | "gemini"

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


# ── gerador de relatório template ─────────────────────────────────────────────
def _template_report(
    assessment: AssessmentResult,
    plan: MealPlan,
    validation: ValidationReport,
) -> str:
    z  = assessment.z_scores
    st = assessment.status
    cp = assessment.child_profile
    dri = plan.daily_targets

    lines = [
        "=" * 68,
        "  GAIA — PEDIATRIC MEAL PLAN",
        f"  Date: {date.today().isoformat()}",
        "=" * 68,
        "",
        "── CHILD PROFILE ────────────────────────────────────────────────",
        f"  Age: {cp.get('age_months')} months  |  Sex: {'Male' if cp.get('sex')=='M' else 'Female'}",
        f"  Weight: {cp.get('weight_kg')} kg  |  Height: {cp.get('height_cm', 'N/A')} cm",
        f"  Region: {cp.get('region', 'N/A')}",
        f"  Age band: {assessment.age_group_label}",
        "",
        "── ANTHROPOMETRIC ASSESSMENT (WHO z-scores) ────────────────────",
        f"  WAZ (weight-for-age):      {z.WAZ if z.WAZ is not None else 'N/A':>6}  → {st.waz_class}",
        f"  HAZ (height-for-age):      {z.HAZ if z.HAZ is not None else 'N/A':>6}  → {st.haz_class}",
        f"  WHZ (weight-for-height):   {z.WHZ if z.WHZ is not None else 'N/A':>6}  → {st.whz_class}",
        f"",
        f"  NUTRITIONAL CLASSIFICATION: {st.classification}",
        f"  Active conditions: {' | '.join(st.conditions)}",
        f"  DRI Adjustment: {assessment.adjustment.get('note', '')}",
        "",
        "── DAILY NUTRITIONAL TARGETS (ADJUSTED DRI) ─────────────────────",
        f"  Energy:    {dri.get('energy_kcal', 'N/A'):>6} kcal",
        f"  Protein:   {dri.get('protein_g', 'N/A'):>6} g",
        f"  Iron:      {dri.get('iron_mg', 'N/A'):>6} mg",
        f"  Zinc:      {dri.get('zinc_mg', 'N/A'):>6} mg",
        f"  Calcium:   {dri.get('calcium_mg', 'N/A'):>6} mg",
        f"  Vitamin A: {dri.get('vitA_ug', 'N/A'):>6} µg",
        f"  Vitamin C: {dri.get('vitC_mg', 'N/A'):>6} mg",
        "",
        "── MEAL PLAN ─────────────────────────────────────────────────────",
        f"  LP Status: {plan.lp_status}  |  Total: {plan.total_g_per_day} g/day",
        "",
    ]

    for meal in plan.meals:
        lines.append(f"  [{meal.meal_name}]  target={meal.energy_target:.0f} kcal | "
                     f"achieved={meal.total_energy:.0f} kcal")
        if meal.items:
            _DISPLAY_ALIASES = {
                "Creme de arroz, pó": "Creme de arroz, mingau",
            }
            for item in meal.items:
                display_name = _DISPLAY_ALIASES.get(item.food_name, item.food_name)
                lines.append(f"    • {display_name[:40]:<40} "
                             f"{item.portion_g:>5.0f} g  "
                             f"({item.energy_kcal:.0f} kcal)")
        else:
            lines.append("    (no items)")
        lines.append("")

    ach = plan.daily_achieved
    lines += [
        "── NUTRITIONAL COVERAGE ─────────────────────────────────────────",
        f"  Energy:    {ach.get('energy_kcal',0):>7.1f} kcal  "
        f"({ach.get('energy_kcal',0)/max(dri.get('energy_kcal',1),1)*100:.1f}%)",
        f"  Protein:   {ach.get('protein_g',0):>7.1f} g     "
        f"({ach.get('protein_g',0)/max(dri.get('protein_g',1),1)*100:.1f}%)",
        f"  Iron:      {ach.get('iron_mg',0):>7.2f} mg    "
        f"({ach.get('iron_mg',0)/max(dri.get('iron_mg',1),1)*100:.1f}%)",
        f"  Zinc:      {ach.get('zinc_mg',0):>7.2f} mg    "
        f"({ach.get('zinc_mg',0)/max(dri.get('zinc_mg',1),1)*100:.1f}%)",
        f"  Calcium:   {ach.get('calcium_mg',0):>7.1f} mg    "
        f"({ach.get('calcium_mg',0)/max(dri.get('calcium_mg',1),1)*100:.1f}%)",
        f"  Vitamin A: {ach.get('vitA_ug',0):>7.1f} µg    "
        f"({ach.get('vitA_ug',0)/max(dri.get('vitA_ug',1),1)*100:.1f}%)",
        f"  Vitamin C: {ach.get('vitC_mg',0):>7.1f} mg    "
        f"({ach.get('vitC_mg',0)/max(dri.get('vitC_mg',1),1)*100:.1f}%)",
        "",
        "── ESTIMATED DAILY COST (Carrefour BR, 2026-11-08) ─────────────",
        f"  Daily cost: R$ {plan.daily_cost:>5.2f}",
        f"  (Reference: Carrefour Online; raw ingredients converted to cooked",
        f"   portions via yield factors. Prices valid for Nov 2026.)",
        "",
        "── VALIDATION ───────────────────────────────────────────────────",
        f"  Result: {'APPROVED' if validation.overall_pass else 'REJECTED'}  "
        f"({sum(1 for c in validation.criteria if c.passed)}/{len(validation.criteria)} criteria OK)",
        f"  Food categories per day: {validation.n_categories}",
    ]

    if validation.warnings:
        lines.append("  Warnings:")
        for w in validation.warnings:
            lines.append(f"    ! {w}")

    if cp.get("age_months", 99) < 24:
        lines += [
            "",
            "── BREASTFEEDING ─────────────────────────────────────────────────",
            "  This child is under 2 years old. According to WHO guidelines,",
            "  breastfeeding should be maintained as a COMPLEMENT to solid",
            "  foods up to 2 years of age or beyond. The plan above covers",
            "  complementary foods only, however, breast milk remains an essential",
            "  part of the diet at this age.",
            "  Ref: WHO (2023). https://www.who.int/publications/i/item/9789240082830",
        ]

    lines += [
        "",
        "─" * 68,
        "  GAIA — Multi-Agent System for Pediatric Nutritional Meal Planning",
        "  TCC — UNISINOS 2026",
        "=" * 68,
    ]

    return "\n".join(lines)


# ── prompt Gemini ─────────────────────────────────────────────────────────────
_GEMINI_PROMPT = """
You are a specialist pediatric nutritionist. Based on the data below,
write a paragraph of guidance for the child's parents/caregivers (maximum 200 words),
explaining: (1) the child's nutritional status, (2) the proposed meal plan, and
(3) two important practical tips.

Use accessible, empathetic language without unnecessary technical jargon.

DATA:
{summary_json}
"""


# ── agente ────────────────────────────────────────────────────────────────────
class ReportAgent:

    def generate(
        self,
        assessment: AssessmentResult,
        plan: MealPlan,
        validation: ValidationReport,
        use_gemini: bool = True,
    ) -> Report:
        """Gera o relatório completo."""
        text_template = _template_report(assessment, plan, validation)

        data = {
            "assessment": {
                "age_group":       assessment.age_group,
                "age_group_label": assessment.age_group_label,
                "z_scores": {
                    "WAZ": assessment.z_scores.WAZ,
                    "HAZ": assessment.z_scores.HAZ,
                    "WHZ": assessment.z_scores.WHZ,
                },
                "classification": assessment.status.classification,
                "adjustment":     assessment.adjustment,
            },
            "meal_plan":  plan.to_dict(),
            "validation": validation.to_dict(),
        }

        narrative = ""
        source = "template"

        if use_gemini and _GEMINI_OK:
            try:
                summary = {
                    "classification":    assessment.status.classification,
                    "age_months":        assessment.child_profile.get("age_months"),
                    "waz":               assessment.z_scores.WAZ,
                    "whz":               assessment.z_scores.WHZ,
                    "target_kcal":       plan.daily_targets.get("energy_kcal"),
                    "achieved_kcal":     plan.daily_achieved.get("energy_kcal"),
                    "n_meals":           sum(1 for m in plan.meals if m.items),
                    "validation_ok":     validation.overall_pass,
                    "adjustment":        assessment.adjustment.get("note"),
                }
                model = genai.GenerativeModel("gemini-3.6-flash")
                resp  = model.generate_content(
                    _GEMINI_PROMPT.format(summary_json=json.dumps(summary,
                                                                   ensure_ascii=False))
                )
                narrative = resp.text.strip()
                source = "gemini"
            except Exception as e:
                narrative = f"(Gemini indisponível: {e})"

        if narrative:
            text_final = text_template + "\n\n── CAREGIVER GUIDANCE (AI) ──────────────────────────────────────\n" + narrative + "\n"
        else:
            text_final = text_template

        data["narrative"] = narrative
        data["source"]    = source

        return Report(text=text_final, data=data, source=source)
