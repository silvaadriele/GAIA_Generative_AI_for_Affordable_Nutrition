"""
Agente de Planejamento de Refeições (Fase 6 — GAIA).

Recebe FoodCatalog (Fase 5) + AssessmentResult (Fase 4) e produz
um plano de refeições diário dividido em 5 refeições.

Fluxo:
  1. LP diário → porções em gramas para cada alimento selecionado
  2. Distribuição pelas 5 refeições conforme proporções energéticas
  3. Retorna MealPlan com estrutura serializable em JSON

Uso:
    from agents.meal_planner_agent import MealPlannerAgent
    planner = MealPlannerAgent()
    plan    = planner.plan(assessment_result, food_catalog)
    # ou:
    plan    = planner.plan_from_profile(child_profile)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.assessment_agent import AssessmentAgent, AssessmentResult
from agents.food_retriever_agent import FoodRetrieverAgent, FoodCatalog, FoodCandidate
from modules.lp_solver import solve_daily_portions, LPSolution
from modules.food_prices import build_price_array, get_price

# ── estrutura de refeições por faixa etária ───────────────────────────────────
# WHO (2023): 6-8m → 2-3 refeições; 9-11m → 3-4; 12m+ → 3-4 + lanches.
# Implementação: 6-8m = 3 refeições principais; 9-11m = 4; 12m+ = 5.
# (nome, label_en, proporção de energia, máximo de alimentos distintos)
def _get_meal_structure(age_months: float) -> list:
    if age_months < 9:      # 6–8m: 3 refeições (WHO: 2–3)
        return [
            ("cafe_manha", "Breakfast",       0.30, 3),
            ("almoco",     "Lunch",           0.40, 6),
            ("jantar",     "Dinner",          0.30, 5),
        ]
    elif age_months < 12:   # 9–11m: 4 refeições (WHO: 3–4)
        return [
            ("cafe_manha",   "Breakfast",         0.25, 3),
            ("almoco",       "Lunch",             0.35, 6),
            ("lanche_tarde", "Afternoon Snack",   0.15, 3),
            ("jantar",       "Dinner",            0.25, 5),
        ]
    else:                   # 12m+: 5 refeições (WHO: 3–4 + lanches)
        return [
            ("cafe_manha",    "Breakfast",         0.25, 3),
            ("lanche_manha",  "Morning Snack",     0.10, 4),
            ("almoco",        "Lunch",             0.35, 6),
            ("lanche_tarde",  "Afternoon Snack",   0.10, 4),
            ("jantar",        "Dinner",            0.20, 5),
        ]


# mantido para compatibilidade com código externo que importe a constante
MEAL_STRUCTURE = _get_meal_structure(24)

MAX_MEALS_PER_FOOD = 2   # cada alimento aparece em no máximo 2 refeições
MIN_MEAL_PORTION  = 10.0 # porção mínima significativa por refeição (g)
MIN_DAILY_FOR_THRESHOLD = 15.0  # alimentos com total diário < este valor são isentos do mínimo

# ── cap de diversidade de categorias por refeição ─────────────────────────────
# Reflete o padrão alimentar brasileiro para crianças pequenas:
#   - 1 cereal por refeição principal; 2 permitidos no café (pão + biscoito é habitual)
#   - 1 laticínio por refeição (não misturar dois tipos de leite/iogurte)
# O pool do LP pode ter múltiplas opções por categoria; o cap só age na distribuição.
_MEAL_CAT_MAX: dict[str, dict[str, int]] = {
    "cafe_manha":   {"Cereais e derivados": 2, "Leite e derivados": 1},
    "lanche_manha": {"Cereais e derivados": 1, "Leite e derivados": 1},
    "almoco":       {"Cereais e derivados": 1, "Leite e derivados": 1, "TBCA": 3},
    "lanche_tarde": {"Cereais e derivados": 1, "Leite e derivados": 1},
    "jantar":       {"Cereais e derivados": 1},
}

# ── cap de nome por refeição ───────────────────────────────────────────────────
# Limita alimentos cujo nome coincide com um padrão a no máximo N por refeição.
# Independente de categoria: "Pão, trigo, francês" e "Pão, de soja" são ambos
# pão e não devem aparecer juntos na mesma refeição (mesmo sendo cereais distintos).
_MEAL_NAME_MAX: dict[str, list[tuple]] = {
    mid: [(re.compile(r"\bpão\b|\bpao\b", re.IGNORECASE), 1)]
    for mid in ("cafe_manha", "lanche_manha", "almoco", "lanche_tarde", "jantar")
}
# amido base (arroz, polenta, macarrão, lasanha): máx 1 por refeição principal
_STARCH_BASE_RE = re.compile(r"\bpolenta\b|\bmacarr[aã]o\b|\blasanha\b|\barroz\b", re.IGNORECASE)
_MEAL_NAME_MAX["almoco"].append((_STARCH_BASE_RE, 1))
_MEAL_NAME_MAX["jantar"].append((_STARCH_BASE_RE, 1))

# ── cap de grupo: categorias que compartilham 1 slot por refeição principal ───
# Carnes + Aves: no máximo 1 item proteico animal por almoço/jantar.
# Evita frango caipira (Carnes) + frango fígado (Aves) juntos na mesma refeição,
# já que são classificados em categorias distintas no TACO vs USDA.
_MEAL_CAT_GROUP_MAX: dict[str, list[tuple]] = {
    "almoco": [({"Carnes e derivados", "Aves e derivados"}, 1)],
    "jantar": [({"Carnes e derivados", "Aves e derivados"}, 1)],
}

# ── limite de refeições por alimento de acordo com a categoria ─────────────────
# Carnes/Aves: 1 refeição por dia (ex: fígado no almoço OU jantar, não ambos).
# Evita que frango fígado ocupe o slot de Aves em almoco E jantar, impedindo
# que frango caipira encontre espaço nas refeições principais.
_CAT_MAX_MEALS: dict[str, int] = {
    "Carnes e derivados": 1,
    "Aves e derivados":   1,
}

# ── alimentos âncora: garantem presença no pool do LP independente do score ───
# Arroz cozido é base do almoço/jantar brasileiro; nas regiões onde polenta
# pontua mais (Centro-Oeste), o N_PER_CAT=3 pode já estar preenchido por
# polenta + creme de arroz + mucilon, excluindo arroz branco do pool.
# Âncora: se nenhum arroz cozido estiver no pool, busca em catalog.ranked.
# Nota: o loop de anchors em _catalog_to_arrays NÃO aplica filtro NAS>0,
# permitindo que alimentos com NAS=0 (alto sódio na forma seca, e.g. Mucilon)
# sejam adicionados ao pool. Esses alimentos são adequados em porções normais.
_ANCHOR_FOOD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Cereais e derivados", re.compile(r"\barroz.*cozido", re.IGNORECASE)),
    # Mucilon (mistura para vitamina): maior fonte de ferro em cereais (12.64mg/100g)
    # NAS=0 no pó seco (sódio 1163mg/100g), mas porções típicas (30g) ficam dentro
    # do MRV; necessário para atingir meta de ferro em 6-11m quando fígado não é
    # suficiente por si só.
    ("Cereais e derivados", re.compile(r"mistura.*vitamina|vitamina.*cereal", re.IGNORECASE)),
    # Fígado bovino: maior densidade de ferro heme na categoria Carnes (necessário quando
    # LP pool não inclui fígado e plano fica sem ferro suficiente)
    ("Carnes e derivados", re.compile(r"f.{1,2}gado", re.IGNORECASE)),
    # Nota: fígado de frango (TACO_400) está em "Carnes e derivados" (não "Aves"),
    # e é cru (sem versão cozida no banco) — portanto coberto pelo anchor acima.
    # Feijão: garante leguminosa canônica no pool (ENANI Nordeste pode priorizar
    # paçoca/amendoim no top-k, deslocando feijão da categoria)
    ("Leguminosas e derivados", re.compile(r"feij.{1,2}o", re.IGNORECASE)),
]

# ── exclusões culturais por refeição (padrão alimentar brasileiro) ────────────
# Alimentos que NÃO pertencem a determinados horários segundo hábitos nacionais.
# Usadas em passes 1 e 2; ignoradas no pass 3 (fallback de emergência).
def _fix_mojibake(name: str) -> str:
    """Corrige nomes com dupla-codificação (Latin-1 bytes lidos como UTF-8)."""
    try:
        return name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


_MEAL_EXCL: dict[str, re.Pattern] = {
    # Café da manhã: sem arroz/macarrão/feijão/sopa e sem folhas de salada.
    # "creme de arroz, pó" é exceção explícita (papa de cereais para lactentes).
    "cafe_manha": re.compile(
        r"\bmacarr[aã]o\b"
        r"|\barroz\b(?!.*\bpó\b)"   # arroz sim se for pó (creme de arroz, pó)
        r"|\bfeij[oõ]|feij[aã]o\b"
        r"|\bsopa\b|\bcaldo\b"
        r"|\bpolenta\b"              # polenta é prato principal; reservado para almoco/jantar
        r"|\balface\b|\brúcula\b|\bagrião\b|\bchicória\b|\bescarola\b|\bagrião\b",
        re.IGNORECASE,
    ),
    # Lanches: sem pratos salgados de almoço/jantar, sem folhas de salada, sem polenta
    # (polenta é prato principal; arroz/macarrão/feijão pertencem à refeição principal)
    "lanche_manha": re.compile(
        r"\bmacarr[aã]o\b|\barroz\b|\bfeij[oõ]\b|\bfeij[aã]o\b|\bsopa\b|\bcaldo\b"
        r"|\balface\b|\brúcula\b|\bagrião\b|\bchicória\b|\bescarola\b"
        r"|\bpolenta\b",
        re.IGNORECASE,
    ),
    "lanche_tarde": re.compile(
        r"\bmacarr[aã]o\b|\barroz\b|\bfeij[oõ]\b|\bfeij[aã]o\b|\bsopa\b|\bcaldo\b"
        r"|\balface\b|\brúcula\b|\bagrião\b|\bchicória\b|\bescarola\b"
        r"|\bpolenta\b",
        re.IGNORECASE,
    ),
    # Almoço: sem biscoito, pão, iogurte, cereais vitamina nem papa de bebê
    # (biscoito/pão são lanches; iogurte é lanche; Mucilon/creme de arroz são papas de café)
    "almoco": re.compile(
        r"\bbiscoito\b|\bpão\b|\bpao\b"
        r"|\biogurte\b"
        r"|mistura para vitamina"
        r"|\bcreme de arroz\b",
        re.IGNORECASE,
    ),
    # Jantar: sem biscoito, pão, iogurte, cereais vitamina nem papa de bebê
    "jantar": re.compile(
        r"\bbiscoito\b|\bpão\b|\bpao\b"
        r"|\biogurte\b"
        r"|mistura para vitamina"
        r"|\bcreme de arroz\b",
        re.IGNORECASE,
    ),
}


# ── tipos ─────────────────────────────────────────────────────────────────────
@dataclass
class MealItem:
    food_id:       str
    food_name:     str
    food_category: str
    portion_g:     float           # gramas nesta refeição
    energy_kcal:   float           # kcal desta porção
    protein_g:     float

    def to_dict(self) -> dict:
        return {
            "food_id":       self.food_id,
            "food_name":     self.food_name,
            "food_category": self.food_category,
            "portion_g":     self.portion_g,
            "energy_kcal":   round(self.energy_kcal, 1),
            "protein_g":     round(self.protein_g, 2),
        }


@dataclass
class Meal:
    meal_id:       str
    meal_name:     str
    energy_target: float           # kcal meta para esta refeição
    items:         list[MealItem] = field(default_factory=list)

    @property
    def total_energy(self) -> float:
        return sum(i.energy_kcal for i in self.items)

    @property
    def total_protein(self) -> float:
        return sum(i.protein_g for i in self.items)

    def to_dict(self) -> dict:
        return {
            "meal_id":        self.meal_id,
            "meal_name":      self.meal_name,
            "energy_target":  round(self.energy_target, 1),
            "energy_actual":  round(self.total_energy, 1),
            "protein_actual": round(self.total_protein, 2),
            "items":          [i.to_dict() for i in self.items],
        }


@dataclass
class MealPlan:
    child_profile:     dict
    age_group:         str
    age_group_label:   str
    lp_status:         str          # "optimal" | "approx" | "infeasible"
    daily_targets:     dict         # DRI ajustados
    daily_achieved:    dict         # nutrientes alcançados pelo LP
    meals:             list[Meal]
    total_g_per_day:   float
    daily_cost:        float = 0.0  # custo estimado em R$/dia (Carrefour, 2026-11-08)

    @property
    def total_energy(self) -> float:
        return sum(m.total_energy for m in self.meals)

    @property
    def total_protein(self) -> float:
        return sum(m.total_protein for m in self.meals)

    def to_dict(self) -> dict:
        return {
            "child_profile":   self.child_profile,
            "age_group":       self.age_group,
            "age_group_label": self.age_group_label,
            "lp_status":       self.lp_status,
            "daily_targets":   self.daily_targets,
            "daily_achieved":  self.daily_achieved,
            "total_energy_kcal":  round(self.total_energy, 1),
            "total_protein_g":    round(self.total_protein, 2),
            "total_g_per_day":    self.total_g_per_day,
            "daily_cost_brl":     round(self.daily_cost, 2),
            "meals":           [m.to_dict() for m in self.meals],
        }


# ── agente ────────────────────────────────────────────────────────────────────
class MealPlannerAgent:

    def __init__(self, top_k: int = 50):
        self._assessment = AssessmentAgent()
        self._retriever  = FoodRetrieverAgent(top_k=top_k)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _catalog_to_arrays(catalog: FoodCatalog, age_group: str):
        """Extrai arrays numpy dos candidatos top-K do catálogo.

        Garante ao menos N_PER_CAT representantes de cada categoria alimentar no pool
        do LP, complementando o top-K com os melhores itens de cada categoria ainda
        sub-representada. Isso permite ao LP escolher entre opções nutricionalmente
        diversas dentro da mesma categoria (ex: arroz vs pão em "Cereais e derivados")
        e reduz infeasibilidade causada por restrições AMDR quando a categoria já tem
        um representante no top-K mas com perfil proteico desfavorável.
        """
        _N_PER_CAT = 3  # mínimo de alimentos por categoria no pool do LP

        cands = list(catalog.top_k)
        # Excluir alimentos com NAS=0 do top-k: captura tabletes de caldo (sódio >20 000 mg/100 g)
        # e alimentos sem nutrientes relevantes. Para 6-8m o MRV de sódio=370 mg faz
        # alimentos com >370 mg/100g terem NAS=0 (e.g. feijão temperado, iogurte sabor).
        # Alimentos críticos de ferro com NAS=0 por sódio (e.g. Mucilon 1163mg/100g seco)
        # são re-adicionados via âncoras abaixo, que operam sem filtro NAS.
        cands = [c for c in cands if c.NAS > 0.0]
        top_ids = {c.food_id for c in cands}

        # conta quantos alimentos de cada categoria já estão no pool
        in_pool_by_cat: dict[str, int] = {}
        for c in catalog.top_k:
            in_pool_by_cat[c.food_category] = in_pool_by_cat.get(c.food_category, 0) + 1

        for cat, cat_foods in catalog.by_category.items():
            needed = max(0, _N_PER_CAT - in_pool_by_cat.get(cat, 0))
            added = 0
            for cand in cat_foods:
                if added >= needed:
                    break
                if cand.food_id not in top_ids and cand.NAS > 0.0:
                    cands.append(cand)
                    top_ids.add(cand.food_id)
                    added += 1

        # âncoras: garante ao menos 1 representante de cada padrão no pool,
        # buscando em catalog.ranked (todos os alimentos elegíveis rankeados)
        # se o pool atual não tiver nenhum item que case com o padrão.
        # Anchors NÃO aplicam filtro NAS>0: alimentos com NAS=0 por sódio no pó
        # seco (e.g. Mucilon 1163mg/100g) são re-adicionados aqui conforme
        # documentado no comentário acima. Porções típicas ficam dentro do MRV.
        for anc_cat, anc_pattern in _ANCHOR_FOOD_PATTERNS:
            already = any(
                c.food_category == anc_cat and anc_pattern.search(c.food_name)
                for c in cands
            )
            if not already:
                for cand in catalog.ranked:
                    if (cand.food_category == anc_cat
                            and anc_pattern.search(cand.food_name)
                            and cand.food_id not in top_ids):
                        cands.append(cand)
                        top_ids.add(cand.food_id)
                        break
        n = len(cands)
        food_ids   = [c.food_id        for c in cands]
        food_names = [c.food_name      for c in cands]
        scores     = np.array([c.preference_score for c in cands])
        nas_arr    = np.array([c.NAS              for c in cands], dtype=float)
        categories = [c.food_category  for c in cands]

        def col(attr):
            return np.array([getattr(c, attr, 0.0) or 0.0 for c in cands],
                            dtype=float)

        nutrients = {
            "energy_kcal": col("energy_kcal"),
            "protein_g":   col("protein_g"),
            "iron_mg":     col("iron_mg"),
            "zinc_mg":     col("zinc_mg"),
            "calcium_mg":  col("calcium_mg"),
            "vitA_ug":     col("vitA_ug"),
            "vitC_mg":     col("vitC_mg"),
        }
        return food_ids, food_names, scores, nutrients, categories, nas_arr

    @staticmethod
    def _distribute_to_meals(
        solution: LPSolution,
        catalog: FoodCatalog,
        daily_energy_target: float,
        meal_structure: list,
    ) -> list[Meal]:
        """
        Distribui as porções diárias do LP pelas refeições da estrutura recebida.

        Estratégia: proporcional à energia de cada refeição, priorizando
        alimentos de categorias coerentes com cada momento do dia.
        """
        # mapa food_id -> FoodCandidate
        food_map: dict[str, FoodCandidate] = {
            c.food_id: c for c in catalog.ranked
        }

        # alimentos selecionados pelo LP (porção > 0)
        selected = {
            fid: g for fid, g in solution.portions.items() if g >= 5
        }
        if not selected:
            meals_out = []
            for mid, mname, prop, _ in meal_structure:
                meals_out.append(Meal(
                    meal_id=mid, meal_name=mname,
                    energy_target=daily_energy_target * prop,
                ))
            return meals_out

        # preferência de categoria por refeição
        _meal_pref: dict[str, set[str]] = {
            "cafe_manha":   {"Leite e derivados", "Cereais e derivados",
                             "Frutas e derivados", "Ovos e derivados"},
            "lanche_manha": {"Frutas e derivados", "Leite e derivados"},
            "almoco":       {"Carnes e derivados", "Aves e derivados",
                             "Leguminosas e derivados", "Cereais e derivados",
                             "Verduras, hortaliças e derivados",
                             "Pescados e frutos do mar", "TBCA"},
            "lanche_tarde": {"Frutas e derivados", "Leite e derivados",
                             "Cereais e derivados"},
            "jantar":       {"Carnes e derivados", "Aves e derivados",
                             "Leguminosas e derivados", "Cereais e derivados",
                             "Verduras, hortaliças e derivados",
                             "Pescados e frutos do mar", "TBCA"},
        }

        # ordenar alimentos selecionados por score decrescente
        sorted_foods = sorted(
            selected.items(),
            key=lambda kv: food_map[kv[0]].preference_score
            if kv[0] in food_map else 0,
            reverse=True,
        )

        meals_out = []
        assigned: dict[str, dict[str, float]] = {
            mid: {} for mid, *_ in meal_structure
        }
        used_g: dict[str, float] = {fid: 0.0 for fid in selected}
        meal_count: dict[str, int] = {fid: 0 for fid in selected}

        def _cat_slots_used(mid: str, category: str) -> int:
            """Quantos alimentos de `category` já foram atribuídos a `mid`."""
            return sum(
                1 for aid in assigned[mid]
                if food_map.get(aid) and food_map[aid].food_category == category
            )

        def _cat_group_blocked(mid: str, category: str) -> bool:
            """True se adicionar `category` a `mid` violaria algum cap de grupo."""
            for cat_set, cap in _MEAL_CAT_GROUP_MAX.get(mid, []):
                if category in cat_set:
                    used = sum(
                        1 for aid in assigned[mid]
                        if food_map.get(aid) and food_map[aid].food_category in cat_set
                    )
                    if used >= cap:
                        return True
            return False

        def _name_cap_blocked(mid: str, food_name: str) -> bool:
            """True se adicionar um alimento com `food_name` violaria algum cap de nome."""
            fixed = _fix_mojibake(food_name)
            for pattern, cap in _MEAL_NAME_MAX.get(mid, []):
                if pattern.search(fixed):
                    used = sum(
                        1 for aid in assigned[mid]
                        if food_map.get(aid) and pattern.search(_fix_mojibake(food_map[aid].food_name))
                    )
                    if used >= cap:
                        return True
            return False

        def _max_meals(fid: str) -> int:
            """Número máximo de refeições para o alimento (1 para carnes/aves)."""
            fc = food_map.get(fid)
            if fc and fc.food_category in _CAT_MAX_MEALS:
                return _CAT_MAX_MEALS[fc.food_category]
            return MAX_MEALS_PER_FOOD

        # passagem 1: alimentos com afinidade de categoria
        for mid, mname, prop, max_items in meal_structure:
            pref_cats = _meal_pref.get(mid, set())
            excl_re   = _MEAL_EXCL.get(mid)
            cat_max   = _MEAL_CAT_MAX.get(mid, {})
            # Jantar: prioriza alimentos já alocados (ex: polenta do almoço) para
            # garantir continuidade calórica e evitar jantar sem amido base.
            if mid == "jantar":
                _placed = {fid for fid in selected if meal_count.get(fid, 0) > 0}
                _sfoods = (
                    [(fid, g) for fid, g in sorted_foods if fid in _placed] +
                    [(fid, g) for fid, g in sorted_foods if fid not in _placed]
                )
            else:
                _sfoods = sorted_foods
            for fid, daily_g in _sfoods:
                if len(assigned[mid]) >= max_items:
                    break
                if meal_count.get(fid, 0) >= _max_meals(fid):
                    continue
                fc = food_map.get(fid)
                if fc and fc.food_category in pref_cats:
                    if excl_re and excl_re.search(_fix_mojibake(fc.food_name)):
                        continue
                    cap = cat_max.get(fc.food_category)
                    if cap is not None and _cat_slots_used(mid, fc.food_category) >= cap:
                        continue
                    if _cat_group_blocked(mid, fc.food_category):
                        continue
                    if _name_cap_blocked(mid, fc.food_name):
                        continue
                    portion = round(daily_g * prop / 5) * 5
                    portion = max(5.0, portion)
                    assigned[mid][fid] = portion
                    used_g[fid] = used_g.get(fid, 0) + portion
                    meal_count[fid] = meal_count.get(fid, 0) + 1

        # passagem 2: alimentos sem nenhuma refeição atribuída (sem afinidade de categoria)
        # Usa apenas itens com 0 slots: itens com >=1 slot têm sua porção normalizada depois.
        # Três sweeps:
        #   (a) todos os constraints incluindo name cap;
        #   (b) relax do cat cap e group cap (mantém excl_re e name cap);
        #   (c) último recurso — relaxa tudo.
        remaining = {
            fid: g for fid, g in selected.items()
            if meal_count.get(fid, 0) == 0
        }
        for fid, daily_g in sorted(
            remaining.items(),
            key=lambda kv: food_map[kv[0]].preference_score
            if kv[0] in food_map else 0,
            reverse=True,
        ):
            # sweep (a): respeita todos os constraints incluindo cat cap e group cap
            for mid, _, prop, max_items in meal_structure:
                if meal_count.get(fid, 0) >= _max_meals(fid):
                    break
                if len(assigned[mid]) < max_items and fid not in assigned[mid]:
                    excl_re = _MEAL_EXCL.get(mid)
                    fc = food_map.get(fid)
                    if excl_re and fc and excl_re.search(_fix_mojibake(fc.food_name)):
                        continue
                    cat_max = _MEAL_CAT_MAX.get(mid, {})
                    cap = cat_max.get(fc.food_category) if fc else None
                    if cap is not None and _cat_slots_used(mid, fc.food_category) >= cap:
                        continue
                    if fc and _cat_group_blocked(mid, fc.food_category):
                        continue
                    if fc and _name_cap_blocked(mid, fc.food_name):
                        continue
                    portion = round(daily_g * prop / 5) * 5
                    portion = max(5.0, portion)
                    assigned[mid][fid] = portion
                    meal_count[fid] = meal_count.get(fid, 0) + 1
                    break
            # sweep (b): se ainda sem refeição, relaxa cat cap e group cap (mantém excl_re e name cap)
            if meal_count.get(fid, 0) == 0:
                for mid, _, prop, max_items in meal_structure:
                    if len(assigned[mid]) < max_items and fid not in assigned[mid]:
                        excl_re = _MEAL_EXCL.get(mid)
                        fc = food_map.get(fid)
                        if excl_re and fc and excl_re.search(_fix_mojibake(fc.food_name)):
                            continue
                        if fc and _name_cap_blocked(mid, fc.food_name):
                            continue
                        portion = round(daily_g * prop / 5) * 5
                        portion = max(5.0, portion)
                        assigned[mid][fid] = portion
                        meal_count[fid] = meal_count.get(fid, 0) + 1
                        break
            # sweep (c): relaxa cat cap + group cap, MAS mantém excl_re.
            # Alimentos excluídos de almoco+jantar (iogurte, pão, biscoito) podem
            # transbordar em +1 item para cafe/lanche em vez de irem para almoco.
            if meal_count.get(fid, 0) == 0:
                fc_sw = food_map.get(fid)
                fc_sw_name = _fix_mojibake(fc_sw.food_name) if fc_sw else ""
                _excl_almoco = _MEAL_EXCL.get("almoco")
                _excl_jantar = _MEAL_EXCL.get("jantar")
                excluded_from_main = bool(
                    fc_sw and
                    _excl_almoco and _excl_almoco.search(fc_sw_name) and
                    _excl_jantar and _excl_jantar.search(fc_sw_name)
                )
                for mid, _, prop, max_items in meal_structure:
                    excl_re = _MEAL_EXCL.get(mid)
                    if excl_re and fc_sw and excl_re.search(fc_sw_name):
                        continue
                    if fc_sw and _name_cap_blocked(mid, fc_sw.food_name):
                        continue
                    overflow = 1 if excluded_from_main else 0
                    if len(assigned[mid]) < max_items + overflow and fid not in assigned[mid]:
                        portion = round(daily_g * prop / 5) * 5
                        portion = max(5.0, portion)
                        assigned[mid][fid] = portion
                        meal_count[fid] = meal_count.get(fid, 0) + 1
                        break

        # passagem 3: fallback — garantir ao menos 1 item por refeição vazia
        for mid, mname, prop, _ in meal_structure:
            if not assigned[mid]:
                for fid, daily_g in sorted_foods:
                    if fid not in assigned[mid]:
                        portion = round(daily_g * prop / 5) * 5
                        portion = max(5.0, portion)
                        assigned[mid][fid] = portion
                        break

        # normalizar porções: cada alimento deve somar daily_g entre as refeições
        # onde foi atribuído (passes 1-3 usavam prop isolada → só ~60% distribuído)
        props_map = {mid: prop for mid, _, prop, _ in meal_structure}

        # boost de normalização para jantar: se projeção de kcal < 65% da meta,
        # aumenta o peso do jantar na redistribuição (√ para suavizar o ajuste)
        _jt_mid = "jantar"
        if _jt_mid in assigned:
            _jt_nom = props_map[_jt_mid]
            _jt_target_pre = daily_energy_target * _jt_nom
            _jt_proj = sum(
                food_map[fid].energy_kcal * assigned[_jt_mid].get(fid, 0) / 100
                for fid in assigned[_jt_mid] if fid in food_map
            )
            if _jt_target_pre > 0 and _jt_proj < 0.65 * _jt_target_pre:
                _boost = (_jt_target_pre / max(_jt_proj, 1.0)) ** 0.5
                props_map[_jt_mid] = _jt_nom * _boost

        for fid, daily_g in selected.items():
            assigned_mids = [mid for mid in assigned if fid in assigned[mid]]
            if not assigned_mids:
                continue
            total_prop = sum(props_map[mid] for mid in assigned_mids)
            for mid in assigned_mids:
                raw = daily_g * props_map[mid] / total_prop
                portion = round(raw / 5) * 5
                portion = max(5.0, portion)
                assigned[mid][fid] = portion

        # ── diversidade de amido: starch_1 → almoço apenas; starch_2 → jantar apenas
        # evita que o mesmo cereal (macarrão) apareça nas duas refeições principais.
        # requer que o LP tenha selecionado ≥ 2 amidos (macarrão cap + mínimo de arroz).
        _main_meal_pair = [m for m in ("almoco", "jantar") if m in assigned]
        if len(_main_meal_pair) == 2:
            _starches = sorted(
                [
                    (fid, selected[fid])
                    for fid in selected
                    if food_map.get(fid) and _STARCH_BASE_RE.search(
                        _fix_mojibake(food_map[fid].food_name)
                    )
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            if len(_starches) >= 2:
                _st1, _st2 = _starches[0][0], _starches[1][0]
                _all_starch_ids = {fid for fid, _ in _starches}
                # starch_1 → almoço: garantir presença no almoço ANTES de remover do jantar
                if _st1 not in assigned["almoco"]:
                    assigned["almoco"][_st1] = max(
                        5.0, round(selected[_st1] * props_map.get("almoco", 0.35) / 5) * 5
                    )
                # remover do jantar: starch_1 e qualquer outro amido que não seja starch_2
                for _sfx in _all_starch_ids - {_st2}:
                    assigned["jantar"].pop(_sfx, None)
                # starch_2 → jantar: garantir presença no jantar ANTES de remover do almoço
                if _st2 not in assigned["jantar"]:
                    assigned["jantar"][_st2] = max(
                        5.0, round(selected[_st2] * props_map.get("jantar", 0.25) / 5) * 5
                    )
                # remover do almoço: starch_2 e qualquer outro amido que não seja starch_1
                for _sfx in _all_starch_ids - {_st1}:
                    assigned["almoco"].pop(_sfx, None)
                # re-normalizar cada amido concentrando toda a porção diária na refeição destino
                for _sf, _dest in [(_st1, "almoco"), (_st2, "jantar")]:
                    _sg = selected[_sf]
                    assigned[_dest][_sf] = max(5.0, round(_sg / 5) * 5)

        # ── rescue jantar: se jantar < 60% da meta calórica, recupera alimentos de alta
        # densidade calórica descartados pelo starch cap ou não distribuídos (meal_count=0).
        # Migra do almoço ou do pool não alocado para garantir continuidade alimentar noturna.
        _jantar = "jantar"
        if _jantar in assigned:
            _jt_prop = props_map.get(_jantar, 0)
            _jt_target = daily_energy_target * _jt_prop
            _jt_kcal = sum(
                food_map[fid].energy_kcal * assigned[_jantar][fid] / 100
                for fid in assigned[_jantar] if fid in food_map
            )
            if _jt_target > 0 and _jt_kcal < 0.60 * _jt_target:
                _excl_j = _MEAL_EXCL.get(_jantar)
                # candidatos: (a) alimentos do almoço não no jantar; (b) alimentos do LP sem refeição
                _from_almoco = [
                    fid for fid in assigned.get("almoco", {})
                    if fid not in assigned[_jantar]
                    and meal_count.get(fid, 0) < _max_meals(fid)
                    and not (_excl_j and food_map.get(fid)
                             and _excl_j.search(_fix_mojibake(food_map[fid].food_name)))
                ]
                _unplaced = [
                    fid for fid in selected
                    if meal_count.get(fid, 0) == 0 and selected[fid] >= 5
                    and not (_excl_j and food_map.get(fid)
                             and _excl_j.search(_fix_mojibake(food_map[fid].food_name)))
                ]
                _rescue_candidates = sorted(
                    set(_from_almoco) | set(_unplaced),
                    key=lambda fid: (food_map[fid].energy_kcal if fid in food_map else 0),
                    reverse=True,
                )
                for fid in _rescue_candidates:
                    if _jt_kcal >= 0.60 * _jt_target:
                        break
                    if fid not in food_map:
                        continue
                    if _name_cap_blocked(_jantar, food_map[fid].food_name):
                        continue
                    daily_g = selected.get(fid, 0)
                    if daily_g <= 0:
                        continue
                    _mids_now = [m for m in assigned if fid in assigned[m]]
                    _tp_new = sum(props_map[m] for m in _mids_now) + _jt_prop
                    assigned[_jantar][fid] = max(5.0, round(daily_g * _jt_prop / _tp_new / 5) * 5)
                    for _m in _mids_now:
                        assigned[_m][fid] = max(5.0, round(daily_g * props_map[_m] / _tp_new / 5) * 5)
                    meal_count[fid] = meal_count.get(fid, 0) + 1
                    _jt_kcal += food_map[fid].energy_kcal * assigned[_jantar][fid] / 100

        # montar objetos Meal
        for mid, mname, prop, _ in meal_structure:
            items = []
            for fid, g in assigned[mid].items():
                fc = food_map.get(fid)
                if fc is None or g <= 0:
                    continue
                kcal = fc.energy_kcal * g / 100.0
                prot = fc.protein_g * g / 100.0
                items.append(MealItem(
                    food_id=fid,
                    food_name=fc.food_name,
                    food_category=fc.food_category,
                    portion_g=g,
                    energy_kcal=round(kcal, 1),
                    protein_g=round(prot, 2),
                ))
            meals_out.append(Meal(
                meal_id=mid,
                meal_name=mname,
                energy_target=round(daily_energy_target * prop, 1),
                items=sorted(items, key=lambda i: -i.energy_kcal),
            ))

        return meals_out

    # ── métodos públicos ──────────────────────────────────────────────────────
    def plan(
        self,
        assessment: AssessmentResult,
        catalog: FoodCatalog,
    ) -> MealPlan:
        """Gera plano dado AssessmentResult e FoodCatalog já computados."""
        targets = assessment.dri      # DRI ajustados (com fatores SAM/MAM)
        age_m   = float(assessment.child_profile.get("age_months", 24))

        food_ids, food_names, scores, nutrients, categories, nas_arr = self._catalog_to_arrays(
            catalog, assessment.age_group
        )

        prices_arr = np.array(build_price_array(food_names), dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = solve_daily_portions(
                food_ids=food_ids,
                scores=scores,
                nutrients=nutrients,
                targets=targets,
                categories=categories,
                age_months=age_m,
                food_names=food_names,
                prices=prices_arr,
                nas_scores=nas_arr,
                alpha=0.8,
            )

        e_target      = targets.get("energy_kcal", 1000.0)
        meal_structure = _get_meal_structure(age_m)
        meals = self._distribute_to_meals(solution, catalog, e_target, meal_structure)

        # Recompute achieved from actual distributed meal portions (avoids discrepancy
        # between LP solution and rounded/distributed items shown in the meal plan).
        food_map_full = {c.food_id: c for c in catalog.ranked}
        actual: dict[str, float] = {k: 0.0 for k in (
            "energy_kcal", "protein_g", "iron_mg", "zinc_mg",
            "calcium_mg", "vitA_ug", "vitC_mg"
        )}
        for meal in meals:
            for item in meal.items:
                fc = food_map_full.get(item.food_id)
                if fc:
                    g = item.portion_g
                    actual["energy_kcal"] += fc.energy_kcal * g / 100.0
                    actual["protein_g"]   += fc.protein_g   * g / 100.0
                    actual["iron_mg"]     += fc.iron_mg     * g / 100.0
                    actual["zinc_mg"]     += fc.zinc_mg     * g / 100.0
                    actual["calcium_mg"]  += fc.calcium_mg  * g / 100.0
                    actual["vitA_ug"]     += fc.vitA_ug     * g / 100.0
                    actual["vitC_mg"]     += fc.vitC_mg     * g / 100.0

        # recuperação de energia pós-distribuição: se energia real < 90% da meta,
        # escala proporcionalmente todas as porções já distribuídas.
        # meta de escala = 95% para absorver perdas do round-to-5g e garantir ≥90%
        _e_floor = e_target * 0.90
        if actual["energy_kcal"] > 0 and actual["energy_kcal"] < _e_floor:
            _scale = (e_target * 0.95) / actual["energy_kcal"]
            for meal in meals:
                for item in meal.items:
                    item.portion_g = round(item.portion_g * _scale / 5) * 5
                    item.portion_g = max(5.0, item.portion_g)
                    fc = food_map_full.get(item.food_id)
                    if fc:
                        item.energy_kcal = round(fc.energy_kcal * item.portion_g / 100.0, 1)
                        item.protein_g   = round(fc.protein_g   * item.portion_g / 100.0, 2)
            # recalcula actual com porções ajustadas
            actual = {k: 0.0 for k in actual}
            for meal in meals:
                for item in meal.items:
                    fc = food_map_full.get(item.food_id)
                    if fc:
                        g = item.portion_g
                        actual["energy_kcal"] += fc.energy_kcal * g / 100.0
                        actual["protein_g"]   += fc.protein_g   * g / 100.0
                        actual["iron_mg"]     += fc.iron_mg     * g / 100.0
                        actual["zinc_mg"]     += fc.zinc_mg     * g / 100.0
                        actual["calcium_mg"]  += fc.calcium_mg  * g / 100.0
                        actual["vitA_ug"]     += fc.vitA_ug     * g / 100.0
                        actual["vitC_mg"]     += fc.vitC_mg     * g / 100.0

        actual_achieved = {k: round(v, 2) for k, v in actual.items()}
        actual_total_g  = round(sum(i.portion_g for m in meals for i in m.items), 1)

        # custo diário estimado: Σ preço_i/100g × porção_i
        actual_daily_cost = round(sum(
            get_price(item.food_name) / 100.0 * item.portion_g
            for meal in meals
            for item in meal.items
        ), 2)

        return MealPlan(
            child_profile=assessment.child_profile,
            age_group=assessment.age_group,
            age_group_label=assessment.age_group_label,
            lp_status=solution.status,
            daily_targets=targets,
            daily_achieved=actual_achieved,
            meals=meals,
            total_g_per_day=actual_total_g,
            daily_cost=actual_daily_cost,
        )

    def plan_from_profile(self, child_profile: dict) -> MealPlan:
        """Pipeline completo: avalia perfil → recupera alimentos → planeja."""
        assessment = self._assessment.assess(child_profile)
        catalog    = self._retriever.retrieve(assessment)
        return self.plan(assessment, catalog)

    def plan_dict(self, child_profile: dict) -> dict:
        """Retorna dicionário serializável em JSON."""
        return self.plan_from_profile(child_profile).to_dict()
