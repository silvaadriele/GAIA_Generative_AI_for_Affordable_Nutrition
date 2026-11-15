"""
Módulo de Programação Linear para alocação diária de porções alimentares.

Formulação:
  Variáveis:  x_i = gramas do alimento i por dia  (i ∈ candidatos)
  Objetivo:   maximizar Σ score_i · x_i            (preferência ponderada)

  Restrições nutricionais (nutrientes por 100g):
    E_min ≤ Σ (kcal_i/100)·x_i ≤ E_max
    Σ (prot_i/100)·x_i ≥ prot_target
    Σ (fe_i/100)·x_i   ≥ fe_target
    Σ (zn_i/100)·x_i   ≥ zn_target
    Σ (ca_i/100)·x_i   ≥ ca_target
    Σ (vitA_i/100)·x_i ≥ vitA_target
    Σ (vitC_i/100)·x_i ≥ vitC_target

  Limites de porção:
    lb_i ≤ x_i ≤ ub_i  (por categoria alimentar e por alimento individual)

Solver: scipy.optimize.linprog (HiGHS)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog

# ── limites de porção por categoria (gramas/dia) ──────────────────────────────
_CATEGORY_BOUNDS: dict[str, tuple[float, float]] = {
    "Cereais e derivados":         (0, 200),
    "Leguminosas e derivados":     (0, 150),
    "Carnes e derivados":          (0, 120),
    "Aves e derivados":            (0, 120),
    "Peixes e frutos do mar":      (0, 100),
    "Leite e derivados":           (0, 250),   # reduzido de 500 para evitar dominância
    "Ovos e derivados":            (0, 100),
    "Frutas e derivados":          (0, 200),
    "Hortaliças e derivados":        (0, 200),
    "Verduras, hortaliças e derivados": (0, 200),
    "Óleos e gorduras":            (0, 30),
    "Açúcares e doces":            (0, 30),
    "Bebidas":                     (0, 200),
    "TBCA":                        (0, 200),
    "default":                     (0, 150),
}

# mínimo total por grupo de categoria (apenas para age >= 6 meses)
# 50g garante porções >= 15g por refeição após distribuição (passa MIN_MEAL_PORTION=10g)
_MIN_CATEGORY_TOTALS: dict[str, float] = {
    "Cereais e derivados":              50.0,
    "Frutas e derivados":               50.0,
    "Leite e derivados":                50.0,  # laticínio mínimo para cálcio e diversidade
    "Verduras, hortaliças e derivados": 80.0,  # Guia MS 2019: hortaliça variada; 80g > cap abobrinha (60g) → força segunda hortaliça
    "Leguminosas e derivados":          30.0,  # Guia MS 2019: feijão obrigatório no almoço e jantar
}

# mínimos para grupos combinados de categorias (qualquer uma satisfaz o total)
# Guia MS 2019: carne ou ave diariamente (fonte de ferro heme)
_MIN_CATEGORY_GROUPS: dict[str, tuple[list[str], float]] = {
    "carnes_aves": (["Carnes e derivados", "Aves e derivados"], 40.0),
}

# teto total por categoria (evita dominância de alimentos de baixa densidade calórica)
# Sem este teto, o LP pode alocar 4 frutas × 200g = 800g/dia
_CATEGORY_GROUP_CAPS: dict[str, float] = {
    "Frutas e derivados":               200.0,
    "Verduras, hortaliças e derivados": 150.0,
    "Leite e derivados":                400.0,
}

# cap por nome de alimento individual (sobrepõe o limite de categoria)
# biscoito polvilho: LP seleciona 85-130g/dia sem restrição; 30g é o máximo razoável
# creme de arroz, pó: farinha para mingau — porção pediátrica = 15-30g (2 colheres)
#   60-70g seco = 232-270 kcal → equivalente ao café inteiro sozinho; cap 30g/dia
_FOOD_NAME_CAPS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bbiscoito\b",    re.IGNORECASE), 30.0),
    (re.compile(r"\bcreme de arroz\b", re.IGNORECASE), 30.0),
]

# cap de grupo por nome: soma de todos os alimentos que casam o padrão ≤ max_g/dia
# pão: 1 fatia média brasileira ≈ 50g; 2 fatias/dia ≈ 100g = consumo adulto
# para crianças 12–35m com energy target ~900–1200 kcal: 60g pão = 180 kcal (15–20% energia)
# iogurte: TACO + TBCA são entradas distintas para o mesmo alimento; 200g/dia evita duplicação
# macarrão: cap 120g/dia → força LP a selecionar um segundo amido (arroz/polenta) para o jantar
# abobrinha: "cozida" + "refogada" são food_ids distintos no TACO; group cap limita a SOMA
#   60g < mínimo hortaliças (80g) → LP precisa selecionar segunda hortaliça (cenoura, abóbora, etc.)
_FOOD_NAME_GROUP_CAPS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bpão\b|\bpao\b", re.IGNORECASE), 60.0),
    (re.compile(r"\biogurte\b", re.IGNORECASE), 200.0),
    (re.compile(r"macarr", re.IGNORECASE), 120.0),
    (re.compile(r"\babobrinha\b", re.IGNORECASE), 60.0),
]

# mínimo por nome de alimento: lb_i = max(lb_categoria, min_g)
# arroz cozido: 30g mínimo garante que o LP selecione arroz como segundo amido
# quando macarrão já está no cap de 120g — promove diversidade almoço/jantar
_FOOD_NAME_MIN_BOUNDS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\barroz\b.*\bcozido\b|\bcozido\b.*\barroz\b", re.IGNORECASE), 30.0),
]

# categorias que fornecem proteína sólida (carne/leguminosa)
_PROTEIN_SOURCE_CATS = frozenset({
    "Carnes e derivados", "Aves e derivados",
    "Pescados e frutos do mar", "Leguminosas e derivados",
})

_ENERGY_TOLERANCE = 0.10   # ±10% da meta energética
_MAX_TOTAL_G      = 1800   # teto de peso total por dia (g)


def _amdr_protein_upper(age_months: float) -> float | None:
    """
    Limite superior da AMDR para proteína como fração da energia total.
    Fonte: IOM (2005) DRI for Macronutrients (iom2005dietary).
    Retorna None para < 6 meses (sem AMDR definido pelo IOM para lactentes).
    """
    if age_months < 12:    # 6–11 meses: sem AMDR definido
        return None
    elif age_months < 48:  # 1–3 anos: AMDR 5–20%
        return 0.20
    else:                  # 4–8 anos: AMDR 10–30% (conservador para 48–59m)
        return 0.30


def _ul_for_age(age_months: float) -> dict[str, float]:
    """
    Retorna UL (Tolerable Upper Intake Level) por nutriente para a faixa etária.
    Fonte: IOM (2001) Dietary Reference Intakes for Vitamin A, Zinc, Iron;
           IOM (2000) DRI for Vitamin C; IOM (2011) DRI for Calcium.

    Nota: vitA_ug NÃO é aplicado como UL no LP para 6–11 meses. O UL de 600µg
    do IOM refere-se a retinol pré-formado de suplementos; fígado de frango é a
    principal fonte de ferro para lactentes nessa faixa e fornece >3000µg/100g,
    tornando o LP infeasível se o UL for aplicado como restrição rígida.
    A supervisão clínica e a limitação de porções de fígado (≤30g/semana,
    conforme MS 2023) são mais adequadas do que um hard constraint no LP.
    """
    if age_months < 6:       # 0–5 meses
        return {"iron_mg": 40.0}
    elif age_months < 12:    # 6–11 meses
        return {"iron_mg": 40.0, "zinc_mg": 5.0}
    elif age_months < 48:    # 12–47 meses (1–3 anos)
        return {
            "vitA_ug":    600.0,
            "vitC_mg":    400.0,
            "iron_mg":     40.0,
            "zinc_mg":      7.0,
            "calcium_mg": 2500.0,
        }
    else:                    # 48–59 meses (conservador: usar UL de 1–3 anos)
        return {
            "vitA_ug":    600.0,
            "vitC_mg":    400.0,
            "iron_mg":     40.0,
            "zinc_mg":      7.0,
            "calcium_mg": 2500.0,
        }


@dataclass
class LPSolution:
    status:     str                       # "optimal" | "infeasible" | "approx"
    portions:   dict[str, float]          # food_id -> gramas/dia
    achieved:   dict[str, float]          # nutriente -> valor atingido
    targets:    dict[str, float]          # nutriente -> meta
    total_g:    float
    daily_cost: float = 0.0               # custo diário em R$ (0 se sem preços)


def solve_daily_portions(
    food_ids:     list[str],
    scores:       np.ndarray,              # preferência [0-1] por alimento
    nutrients:    dict[str, np.ndarray],   # nome -> array por alimento (por 100g)
    targets:      dict[str, float],        # nome -> meta diária
    categories:   list[str],
    age_months:   float,
    food_names:   list[str] | None = None,
    prices:       np.ndarray | None = None,  # R$/100g por alimento
    daily_budget: float | None = None,       # orçamento máximo em R$/dia (None = sem restrição)
    nas_scores:   np.ndarray | None = None,  # NAS por alimento (faixa etária já selecionada)
    alpha:        float = 0.8,               # peso da preferência no objetivo composto (0=só NAS, 1=só pref)
) -> LPSolution:
    """
    Resolve o LP e retorna as porções em gramas/dia.

    nutrients esperados: energy_kcal, protein_g, iron_mg, zinc_mg,
                         calcium_mg, vitA_ug, vitC_mg
    targets com as mesmas chaves (valores DRI ajustados).
    Se prices e daily_budget forem fornecidos, adiciona restrição de custo:
        Σ (prices_i/100 × x_i) ≤ daily_budget
    Se nas_scores for fornecido, objetivo composto:
        maximize α × pref_norm + (1-α) × NAS_norm
    ambos normalizados pelo valor máximo do vetor.
    """
    n = len(food_ids)
    if n == 0:
        return LPSolution("infeasible", {}, {}, targets, 0.0)

    # limites por categoria
    bounds = []
    for cat in categories:
        lo, hi = _CATEGORY_BOUNDS.get(cat, _CATEGORY_BOUNDS["default"])
        bounds.append((lo, hi))

    # cap individual por nome de alimento (sobrepõe limite de categoria)
    if food_names:
        for i, name in enumerate(food_names):
            for pattern, max_g in _FOOD_NAME_CAPS:
                if pattern.search(name):
                    lo, hi = bounds[i]
                    bounds[i] = (lo, min(hi, max_g))
                    break

    # mínimo individual por nome de alimento (lower bound ≥ min_g)
    if food_names:
        for i, name in enumerate(food_names):
            for pattern, min_g in _FOOD_NAME_MIN_BOUNDS:
                if pattern.search(name):
                    lo, hi = bounds[i]
                    bounds[i] = (max(lo, min_g), hi)
                    break

    # ── objetivo: maximizar α × pref_norm + (1-α) × NAS_norm ────────────────
    s = scores.astype(float)
    if nas_scores is not None and alpha < 1.0:
        nas = nas_scores.astype(float)
        s_max   = s.max()   if s.max()   > 0 else 1.0
        nas_max = nas.max() if nas.max() > 0 else 1.0
        c = -(alpha * (s / s_max) + (1.0 - alpha) * (nas / nas_max))
    else:
        c = -s

    # ── restrições de inequação Ax ≤ b ────────────────────────────────────────
    A_ub: list[np.ndarray] = []
    b_ub: list[float]       = []

    def add_ineq(coeff: np.ndarray, rhs: float):
        A_ub.append(coeff)
        b_ub.append(rhs)

    kcal = nutrients.get("energy_kcal", np.zeros(n))
    prot = nutrients.get("protein_g",   np.zeros(n))
    fe   = nutrients.get("iron_mg",     np.zeros(n))
    zn   = nutrients.get("zinc_mg",     np.zeros(n))
    ca   = nutrients.get("calcium_mg",  np.zeros(n))
    va   = nutrients.get("vitA_ug",     np.zeros(n))
    vc   = nutrients.get("vitC_mg",     np.zeros(n))

    e_target = targets.get("energy_kcal", 0)
    if e_target > 0:
        # energia <= E_max
        add_ineq(kcal / 100.0, e_target * (1 + _ENERGY_TOLERANCE))
        # energia >= E_min  → -(kcal/100)·x ≤ -E_min
        add_ineq(-kcal / 100.0, -e_target * (1 - _ENERGY_TOLERANCE))

    # macros e micronutrientes: >= target  →  -coeff·x ≤ -target
    for arr, key in [
        (prot, "protein_g"),
        (fe,   "iron_mg"),
        (zn,   "zinc_mg"),
        (ca,   "calcium_mg"),
        (va,   "vitA_ug"),
        (vc,   "vitC_mg"),
    ]:
        t = targets.get(key, 0)
        if t > 0:
            add_ineq(-arr / 100.0, -t)

    # mínimo por grupo de categoria (diversidade alimentar, apenas >= 6 meses)
    if age_months >= 6:
        for cat, min_g in _MIN_CATEGORY_TOTALS.items():
            cat_idx = [i for i, c in enumerate(categories) if c == cat]
            if cat_idx:
                coeff = np.zeros(n)
                for idx in cat_idx:
                    coeff[idx] = -1.0
                add_ineq(coeff, -min_g)

        # mínimos combinados: soma de categorias agrupadas (ex: carne OU ave)
        for _, (cats, min_g) in _MIN_CATEGORY_GROUPS.items():
            cat_idx = [i for i, c in enumerate(categories) if c in cats]
            if cat_idx:
                coeff = np.zeros(n)
                for idx in cat_idx:
                    coeff[idx] = -1.0
                add_ineq(coeff, -min_g)

        # teto total por categoria (evita dominância de frutas/hortaliças/laticínios)
        for cat, cap in _CATEGORY_GROUP_CAPS.items():
            cat_idx = [i for i, c_cat in enumerate(categories) if c_cat == cat]
            if cat_idx:
                coeff = np.zeros(n)
                for idx in cat_idx:
                    coeff[idx] = 1.0
                add_ineq(coeff, cap)

        # teto de grupo por nome: soma de alimentos que casam o padrão (ex: todos os pães)
        if food_names:
            for pattern, max_g in _FOOD_NAME_GROUP_CAPS:
                matched = [i for i, nm in enumerate(food_names) if pattern.search(nm)]
                if matched:
                    coeff = np.zeros(n)
                    for idx in matched:
                        coeff[idx] = 1.0
                    add_ineq(coeff, max_g)

        # Guia MS 2019 p.28: suco de frutas ≤ 120 mL/dia para 12–35 meses
        # (<12m já filtrado no food_retriever; ≥36m sem restrição explícita no Guia)
        if 12 <= age_months < 36 and food_names:
            _suco_pat = re.compile(r"\bsuco\b|\bsumo\b", re.IGNORECASE)
            suco_idx = [i for i, nm in enumerate(food_names) if _suco_pat.search(nm)]
            if suco_idx:
                coeff = np.zeros(n)
                for idx in suco_idx:
                    coeff[idx] = 1.0
                add_ineq(coeff, 120.0)

        # mínimo de fonte proteica sólida (carne/aves/peixe/leguminosa >= 20g)
        prot_idx = [i for i, c_cat in enumerate(categories)
                    if c_cat in _PROTEIN_SOURCE_CATS]
        if prot_idx:
            coeff = np.zeros(n)
            for idx in prot_idx:
                coeff[idx] = -1.0
            add_ineq(coeff, -20.0)

    # restrição de custo diário: Σ (price_i/100 × x_i) ≤ daily_budget
    if prices is not None and daily_budget is not None:
        add_ineq(prices / 100.0, daily_budget)

    # UL superior (DRI Tolerable Upper Intake Level): nutriente ≤ UL
    ul = _ul_for_age(age_months)
    for name, arr in [
        ("vitA_ug", va), ("vitC_mg", vc),
        ("iron_mg", fe), ("zinc_mg", zn), ("calcium_mg", ca),
    ]:
        u = ul.get(name)
        if u is not None:
            add_ineq(arr / 100.0, u)

    # AMDR proteína: prot_g * 4 kcal/g ≤ amdr_upper * E_target
    # Fonte: IOM (2005) iom2005dietary — evita superalocação proteica
    amdr_prot = _amdr_protein_upper(age_months)
    if amdr_prot is not None and e_target > 0:
        add_ineq(prot / 100.0 * 4.0, amdr_prot * e_target)

    # peso total ≤ teto
    add_ineq(np.ones(n), _MAX_TOTAL_G)

    A_mat = np.array(A_ub, dtype=float) if A_ub else np.empty((0, n))
    b_vec = np.array(b_ub, dtype=float) if b_ub else np.empty(0)

    # ── resolver ──────────────────────────────────────────────────────────────
    res = linprog(
        c,
        A_ub=A_mat,
        b_ub=b_vec,
        bounds=bounds,
        method="highs",
        options={"disp": False},
    )

    if res.status == 0:
        x = res.x
        status = "optimal"
    else:
        # LP infeasível: relaxa metas para 80% e tenta novamente
        # UL mantidos mesmo no fallback — são limites de segurança, não metas
        relaxed_targets = {k: v * 0.80 for k, v in targets.items()}
        A_ub2, b_ub2 = [], []
        e2 = relaxed_targets.get("energy_kcal", 0)
        if e2 > 0:
            A_ub2.append(kcal / 100.0);  b_ub2.append(e2 * 1.15)
            A_ub2.append(-kcal / 100.0); b_ub2.append(-e2 * 0.85)
        for arr, key in [(prot,"protein_g"),(fe,"iron_mg"),(zn,"zinc_mg"),
                         (ca,"calcium_mg"),(va,"vitA_ug"),(vc,"vitC_mg")]:
            t = relaxed_targets.get(key, 0)
            if t > 0:
                A_ub2.append(-arr / 100.0); b_ub2.append(-t)
        for name, arr in [
            ("vitA_ug", va), ("vitC_mg", vc),
            ("iron_mg", fe), ("zinc_mg", zn), ("calcium_mg", ca),
        ]:
            u = ul.get(name)
            if u is not None:
                A_ub2.append(arr / 100.0); b_ub2.append(u)
        if amdr_prot is not None and e2 > 0:
            A_ub2.append(prot / 100.0 * 4.0); b_ub2.append(amdr_prot * e2 * 1.15)
        # group caps mantidos no fallback (limites de segurança, não metas)
        if age_months >= 6:
            for cat, cap in _CATEGORY_GROUP_CAPS.items():
                cat_idx = [i for i, c_cat in enumerate(categories) if c_cat == cat]
                if cat_idx:
                    coeff = np.zeros(n)
                    for idx in cat_idx:
                        coeff[idx] = 1.0
                    A_ub2.append(coeff); b_ub2.append(cap)
            # teto de grupo por nome (fallback: mantém cap de pão)
            if food_names:
                for pattern, max_g in _FOOD_NAME_GROUP_CAPS:
                    matched = [i for i, nm in enumerate(food_names) if pattern.search(nm)]
                    if matched:
                        coeff = np.zeros(n)
                        for idx in matched:
                            coeff[idx] = 1.0
                        A_ub2.append(coeff); b_ub2.append(max_g)
            # suco ≤ 120g/dia para 12–35m (fallback: mantém cap do Guia MS 2019)
            if 12 <= age_months < 36 and food_names:
                _suco_pat = re.compile(r"\bsuco\b|\bsumo\b", re.IGNORECASE)
                suco_idx = [i for i, nm in enumerate(food_names) if _suco_pat.search(nm)]
                if suco_idx:
                    coeff = np.zeros(n)
                    for idx in suco_idx:
                        coeff[idx] = 1.0
                    A_ub2.append(coeff); b_ub2.append(120.0)
            # mínimo de hortaliças no fallback para garantir 4ª categoria
            hortalica_idx = [i for i, c_cat in enumerate(categories)
                             if c_cat == "Verduras, hortaliças e derivados"]
            if hortalica_idx:
                coeff = np.zeros(n)
                for idx in hortalica_idx:
                    coeff[idx] = -1.0
                A_ub2.append(coeff); b_ub2.append(-10.0)
            # mínimo de leite no fallback para cálcio (≥11m: faixa 9-11m em complementação)
            if age_months >= 11:
                leite_idx = [i for i, c_cat in enumerate(categories)
                             if c_cat == "Leite e derivados"]
                if leite_idx:
                    coeff = np.zeros(n)
                    for idx in leite_idx:
                        coeff[idx] = -1.0
                    A_ub2.append(coeff); b_ub2.append(-15.0)
            # carnes_aves e leguminosas: não forçados no fallback.
            # vitA UL (600μg para 12-35m) limita fígado a ~4g → contribuição de ferro
            # quase nula. Adicionar carnes_aves ≥ 40g + iron ≥ 8mg no fallback cria
            # infeasibilidade para Nordeste (onde LightGBM prioriza paçoca/amendoim
            # sobre feijão e as carnes disponíveis têm ferro baixo).
            # Fígado e feijão estão no pool via âncoras; LP usa-os naturalmente
            # para cumprir iron ≥ 80% da meta, dentro do UL de vitA.
            # sobre feijão, tornando a constraint infeasível junto com carnes_aves + iron ≥ 8mg.
            # Feijão é garantido por âncora no pool; LP escolhe naturalmente se preferência alta.
        A_ub2.append(np.ones(n)); b_ub2.append(_MAX_TOTAL_G)
        # custo no fallback: mantém restrição se fornecida (afford. é limite de segurança)
        if prices is not None and daily_budget is not None:
            A_ub2.append(prices / 100.0); b_ub2.append(daily_budget)

        A2 = np.array(A_ub2, dtype=float) if A_ub2 else np.empty((0, n))
        b2 = np.array(b_ub2, dtype=float) if b_ub2 else np.empty(0)
        res2 = linprog(c, A_ub=A2, b_ub=b2, bounds=bounds, method="highs",
                       options={"disp": False})
        x = res2.x if res2.status == 0 else np.zeros(n)
        status = "approx" if res2.status == 0 else "infeasible"

    # ── pós-processamento: arredondar para 5g ────────────────────────────────
    x_pre = x.copy()
    x = np.round(x / 5.0) * 5.0
    # alimentos que LP selecionou (>0) mas arredondaram para 0: mínimo 5g
    x[(x_pre > 0) & (x < 5)] = 5.0

    portions = {food_ids[i]: float(x[i]) for i in range(n) if x[i] > 0}

    # nutrientes alcançados
    achieved = {}
    for name, arr in [("energy_kcal", kcal), ("protein_g", prot),
                      ("iron_mg", fe), ("zinc_mg", zn), ("calcium_mg", ca),
                      ("vitA_ug", va), ("vitC_mg", vc)]:
        achieved[name] = round(float(np.dot(arr / 100.0, x)), 2)

    # custo diário calculado (R$)
    if prices is not None:
        daily_cost = round(float(np.dot(prices / 100.0, x)), 2)
    else:
        daily_cost = 0.0

    return LPSolution(
        status=status,
        portions=portions,
        achieved=achieved,
        targets=targets,
        total_g=round(float(x.sum()), 1),
        daily_cost=daily_cost,
    )
