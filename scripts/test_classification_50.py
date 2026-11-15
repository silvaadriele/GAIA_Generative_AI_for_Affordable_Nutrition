"""
Testa todos os bugs e não-conformidades corrigidos no GAIA com 50 perfis ENANI.

Cobertura:
  Bug 1 -- Classificacao multi-dimensional (WHZ + HAZ + WAZ)
  Bug 2 -- Soma das refeicoes = cobertura nutricional (daily_achieved recomputado)
  Bug 3 -- Numero de refeicoes correto por faixa etaria (3/4/5 refeicoes)
  Bug 4 -- Alimentos LP com daily_g pequeno aparecem no plano (MIN_MEAL_PORTION fix)
  Guia MS 2019 -- Filtros etarios: suco <12m, desnatado <24m, pão <12m, biscoito <12m
  Guia MS 2019 -- Mínimo de hortalicas/verduras no plano
  Guia MS 2019 -- Mínimo de carnes+aves no plano
"""

from __future__ import annotations

import os, re, sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from agents.assessment_agent import AssessmentAgent, _get_active_conditions
from agents.meal_planner_agent import MealPlannerAgent, _get_meal_structure

ENANI_CSV = "data/raw/enani_profiles.csv"
N = 50
SEED = 7

_SUCO_RE    = re.compile(r"\bsuco\b|\bsumo\b", re.IGNORECASE)
_DESNAT_RE  = re.compile(r"\bdesnatado\b",      re.IGNORECASE)
_PAO_RE     = re.compile(r"\bpão\b|\bpao\b",    re.IGNORECASE)
_BISCOITO_RE = re.compile(r"\bbiscoito\b",       re.IGNORECASE)


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(ENANI_CSV, low_memory=False)
    out = pd.DataFrame({
        "age_months": pd.to_numeric(df["s00d_idade_peso"], errors="coerce") / 30.4375,
        "weight_kg":  pd.to_numeric(df["s02_peso_medida1"], errors="coerce"),
        "height_cm":  pd.to_numeric(df["s07_altura_medida1"], errors="coerce"),
        "sex":        df["b02_sexo"].map({"Masculino": "M", "Feminino": "F"}),
        "region":     df["a00_regiao"],
        "enani_waz":  pd.to_numeric(df.get("vd_zwaz",        pd.Series(dtype=float)), errors="coerce"),
        "enani_haz":  pd.to_numeric(df.get("vd_zhaz",        pd.Series(dtype=float)), errors="coerce"),
        "enani_whz":  pd.to_numeric(df.get("vd_anthro_zwfl", pd.Series(dtype=float)), errors="coerce"),
    })
    out = out.dropna(subset=["age_months", "weight_kg", "height_cm", "sex"])
    out = out[out["age_months"].between(6, 60) & (out["weight_kg"] > 0) & (out["height_cm"] > 0)]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(out), size=min(N, len(out)), replace=False)
    return out.iloc[sorted(idx)].reset_index(drop=True)


def _expected_n_meals(age_months: float) -> int:
    if age_months < 9:  return 3
    if age_months < 12: return 4
    return 5


def run():
    sample  = load_sample()
    assess  = AssessmentAgent()
    planner = MealPlannerAgent()

    rows = []
    for _, row in sample.iterrows():
        profile = {
            "age_months": round(float(row["age_months"]), 1),
            "sex":        str(row["sex"]),
            "weight_kg":  round(float(row["weight_kg"]), 2),
            "height_cm":  round(float(row["height_cm"]), 1),
            "region":     str(row.get("region", "Sudeste")),
        }
        age_m = profile["age_months"]

        try:
            # -- Bug 1: classificacao multi-dimensional ----------------------
            r = assess.assess(profile)
            from agents.assessment_agent import _classify_whz, _classify_waz
            whz = r.z_scores.WHZ
            waz = r.z_scores.WAZ
            old_clf = (
                "SAM" if whz is not None and whz < -3 else
                "MAM" if whz is not None and whz < -2 else
                "Normal" if whz is not None and whz <= 1 else
                "Overweight_Risk" if whz is not None and whz <= 2 else
                "Overweight" if whz is not None and whz <= 3 else
                "Obese" if whz is not None else
                ("SAM" if waz is not None and waz < -3 else
                 "MAM" if waz is not None and waz < -2 else "Normal")
            )
            clf_changed = old_clf != r.status.classification

            # -- Plano alimentar ---------------------------------------------
            from agents.food_retriever_agent import FoodRetrieverAgent
            cat  = FoodRetrieverAgent().retrieve(r)
            plan = planner.plan(r, cat)

            ach = plan.daily_achieved
            tgt = plan.daily_targets

            # Bug 2: soma das refeicoes = cobertura (daily_achieved recomputado)
            energy_from_meals = sum(
                item.energy_kcal for meal in plan.meals for item in meal.items
            )
            energy_reported = ach.get("energy_kcal", 0)
            bug2_ok = abs(energy_from_meals - energy_reported) < 1.0  # tolerância 1 kcal

            # Bug 3: numero de refeicoes correto
            n_meals_expected = _expected_n_meals(age_m)
            n_meals_actual   = len(plan.meals)
            bug3_ok = n_meals_actual == n_meals_expected

            # Bug 4: todos os alimentos selecionados pelo LP aparecem no plano
            from agents.meal_planner_agent import _get_meal_structure
            from modules.lp_solver import solve_daily_portions
            import warnings as ww
            food_ids, food_names, scores, nutrients, categories = planner._catalog_to_arrays(cat, r.age_group)
            with ww.catch_warnings():
                ww.simplefilter("ignore")
                sol = solve_daily_portions(food_ids, scores, nutrients, r.dri, categories, age_m, food_names)
            lp_foods = {fid for fid, g in sol.portions.items() if g >= 5}
            plan_foods = {item.food_id for meal in plan.meals for item in meal.items}
            missing_from_plan = lp_foods - plan_foods
            bug4_ok = len(missing_from_plan) == 0

            # Guia MS -- filtros etarios
            all_items = [item for meal in plan.meals for item in meal.items]
            suco_viol    = any(_SUCO_RE.search(i.food_name)    for i in all_items) if age_m < 12 else False
            desnat_viol  = any(_DESNAT_RE.search(i.food_name)  for i in all_items) if age_m < 24 else False
            pao_viol     = any(_PAO_RE.search(i.food_name)     for i in all_items) if age_m < 12 else False
            biscoito_viol = any(_BISCOITO_RE.search(i.food_name) for i in all_items) if age_m < 12 else False
            age_filters_ok = not (suco_viol or desnat_viol or pao_viol or biscoito_viol)

            # Guia MS -- hortalicas presentes
            veg_g = sum(
                i.portion_g for meal in plan.meals for i in meal.items
                if i.food_category == "Verduras, hortaliças e derivados"
            )
            veg_ok = veg_g >= 30

            # Guia MS -- carnes+aves presentes
            meat_g = sum(
                i.portion_g for meal in plan.meals for i in meal.items
                if i.food_category in {"Carnes e derivados", "Aves e derivados"}
            )
            meat_ok = meat_g >= 5  # ao menos aparece no plano

            rows.append({
                "age_m":          age_m,
                "sex":            profile["sex"],
                "stratum":        r.age_group,
                "WAZ":            r.z_scores.WAZ,
                "HAZ":            r.z_scores.HAZ,
                "WHZ":            r.z_scores.WHZ,
                "enani_waz":      round(float(row["enani_waz"]), 2) if pd.notna(row.get("enani_waz")) else None,
                "old_clf":        old_clf,
                "new_clf":        r.status.classification,
                "conditions":     " | ".join(r.status.conditions),
                "clf_changed":    clf_changed,
                "bug2_ok":        bug2_ok,
                "bug3_ok":        bug3_ok,
                "bug4_ok":        bug4_ok,
                "n_meals":        n_meals_actual,
                "n_meals_exp":    n_meals_expected,
                "missing_foods":  len(missing_from_plan),
                "age_filters_ok": age_filters_ok,
                "veg_g":          round(veg_g, 1),
                "veg_ok":         veg_ok,
                "meat_g":         round(meat_g, 1),
                "meat_ok":        meat_ok,
                "energy_pct":     round(ach.get("energy_kcal", 0) / tgt.get("energy_kcal", 1) * 100, 1),
                "iron_pct":       round(ach.get("iron_mg", 0)     / tgt.get("iron_mg", 1)     * 100, 1),
                "vitA_pct":       round(ach.get("vitA_ug", 0)     / tgt.get("vitA_ug", 1)     * 100, 1),
            })

        except Exception as e:
            rows.append({"age_m": age_m, "error": str(e)})

    df = pd.DataFrame(rows)

    # -- relatório -------------------------------------------------------------
    W = 68
    print(f"\n{'='*W}")
    print(f"{'GAIA -- SUITE DE TESTES (50 perfis ENANI)':^{W}}")
    print(f"{'='*W}")

    ok = df[~df.get("error", pd.Series(dtype=str, index=df.index)).fillna("").astype(bool)].copy()
    erros = len(df) - len(ok)
    print(f"\nPerfis avaliados : {len(df)}  |  Erros: {erros}")

    def _bar(passed, total):
        pct = passed / total * 100 if total else 0
        mark = "OK " if passed == total else "FAIL"
        return f"[{mark}]  {passed:>3}/{total}  ({pct:.0f}%)"

    print(f"\n-- Bug 1 -- Classificacao multi-dimensional --------------------------")
    clf_changed = ok["clf_changed"].sum()
    print(f"  Perfis com classificacao corrigida : {clf_changed}")
    print(f"  Distribuicao nova classificacao:")
    for clf, cnt in ok["new_clf"].value_counts().items():
        print(f"    {clf:<22} {cnt:>3}")
    multi = ok[ok["conditions"].str.contains(r"\|", na=False)]
    print(f"  Perfis com >=2 condicoes ativas     : {len(multi)}")

    print(f"\n-- Bug 2 -- Soma refeicoes = cobertura nutricional -------------------")
    n = len(ok)
    print(f"  Energia refeicoes ~ daily_achieved : {_bar(ok['bug2_ok'].sum(), n)}")
    failures = ok[~ok["bug2_ok"]]
    for _, r2 in failures.iterrows():
        print(f"    x {r2['age_m']}m {r2['sex']} -- discrepancia detectada")

    print(f"\n-- Bug 3 -- Numero de refeicoes por faixa etaria ---------------------")
    print(f"  Refeicoes corretas                 : {_bar(ok['bug3_ok'].sum(), n)}")
    wrong = ok[~ok["bug3_ok"]]
    for _, r3 in wrong.iterrows():
        print(f"    x {r3['age_m']}m -> esperado {r3['n_meals_exp']}, obtido {r3['n_meals']}")

    print(f"\n-- Bug 4 -- Todos os alimentos LP aparecem no plano ------------------")
    print(f"  Nenhum alimento LP descartado      : {_bar(ok['bug4_ok'].sum(), n)}")
    dropped = ok[~ok["bug4_ok"]]
    for _, r4 in dropped.iterrows():
        print(f"    x {r4['age_m']}m {r4['sex']} -- {r4['missing_foods']} alimento(s) LP ausentes no plano")

    print(f"\n-- Guia MS 2019 -- Filtros etarios -----------------------------------")
    print(f"  Nenhuma violacao de filtro etário  : {_bar(ok['age_filters_ok'].sum(), n)}")
    age_viol = ok[~ok["age_filters_ok"]]
    for _, rv in age_viol.iterrows():
        print(f"    x {rv['age_m']}m {rv['sex']} -- violacao de filtro")

    print(f"\n-- Guia MS 2019 -- Hortalicas no plano (>=30g/dia) -------------------")
    print(f"  Planos com hortalicas >=30g/dia     : {_bar(ok['veg_ok'].sum(), n)}")
    print(f"  Mediana hortalicas                 : {ok['veg_g'].median():.0f} g/dia")

    print(f"\n-- Guia MS 2019 -- Carnes+aves no plano -----------------------------")
    print(f"  Planos com carnes/aves >=5g/dia     : {_bar(ok['meat_ok'].sum(), n)}")
    print(f"  Mediana carnes+aves                : {ok['meat_g'].median():.0f} g/dia")

    print(f"\n-- Cobertura nutricional (mediana dos 50 perfis) --------------------")
    print(f"  Energia   : {ok['energy_pct'].median():.1f}%")
    print(f"  Ferro     : {ok['iron_pct'].median():.1f}%")
    print(f"  Vitamina A: {ok['vitA_pct'].median():.1f}%")

    # resumo final
    all_passed = (
        ok["bug2_ok"].all() and ok["bug3_ok"].all() and
        ok["bug4_ok"].all() and ok["age_filters_ok"].all() and
        ok["veg_ok"].all()  and ok["meat_ok"].all()
    )
    print(f"\n{'='*W}")
    if all_passed and erros == 0:
        print(f"{'[PASS] TODOS OS TESTES PASSARAM':^{W}}")
    else:
        print(f"{'[FAIL] ALGUNS TESTES FALHARAM -- ver detalhes acima':^{W}}")
    print(f"{'='*W}\n")

    return df


if __name__ == "__main__":
    run()
