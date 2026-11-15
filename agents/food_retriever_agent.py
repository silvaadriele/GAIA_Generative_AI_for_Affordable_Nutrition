"""
Agente de Recuperação de Alimentos (Fase 5 — GAIA).

Recebe o perfil da criança (já avaliado pela Fase 4) e pontua os 619
alimentos do banco usando o modelo LightGBM treinado na Fase 3.

Retorna:
  - lista completa ranqueada por score de preferência
  - top-K candidatos (padrão: 50)
  - agrupamento por categoria alimentar (para diversidade nas refeições)
  - composição nutricional de cada alimento (para o LP da Fase 6)

Uso:
    from agents.food_retriever_agent import FoodRetrieverAgent
    from agents.assessment_agent import AssessmentAgent

    assessment = AssessmentAgent().assess(child_profile)
    retriever  = FoodRetrieverAgent()
    catalog    = retriever.retrieve(assessment)

    # ou direto pelo perfil bruto:
    catalog    = retriever.retrieve_from_profile(child_profile)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from agents.assessment_agent import AssessmentAgent, AssessmentResult

_MEAT_CATS   = {"Carnes e derivados", "Aves e derivados"}
_COOKED_RE   = re.compile(
    r"\b(cozido|cozida|assado|assada|grelhado|grelhada|frito|frita|refogado|refogada)\b",
    re.IGNORECASE,
)
# Normaliza variações de teor de gordura para comparação de bases:
# "paleta, com gordura" e "paleta, sem gordura" referem-se ao mesmo corte.
_FAT_NORM_RE = re.compile(r",\s*(?:com|sem)\s+gordura\b", re.IGNORECASE)

# ── filtros por faixa etária (aplicados no retrieve, não no __init__) ─────────
# Guia Alimentar para Crianças Brasileiras Menores de 2 Anos (MS, 2019):
#   - Suco de frutas contraindicado antes de 1 ano (p. 83)
#   - Leite desnatado inadequado antes de 2 anos (baixo teor de gordura e vitaminas lipossolúveis)
#   - Pão/biscoito industrializado não recomendado antes de 1 ano (complementação com alimentos caseiros)
_AGE_FILTERS: list[tuple[float, re.Pattern]] = [
    (12.0, re.compile(r"\bsuco\b|\bsumo\b",    re.IGNORECASE)),  # suco <12m
    (24.0, re.compile(r"\bdesnatado\b",         re.IGNORECASE)),  # leite desnatado <24m
    (12.0, re.compile(r"\bpão\b|\bpao\b",       re.IGNORECASE)),  # pão industrializado <12m
    (12.0, re.compile(r"\bbiscoito\b",          re.IGNORECASE)),  # biscoito <12m
]


def _normalize_meat_base(base: str) -> str:
    return _FAT_NORM_RE.sub("", base).strip().rstrip(",").strip()


def _raw_meats_with_cooked_version(foods: pd.DataFrame) -> pd.Series:
    """
    Retorna máscara booleana (True = remover): itens crus de carnes/aves
    que possuem versão cozida no banco.
    Itens crus sem par cozido (ex: fígado de frango) são preservados.
    """
    meat_mask = foods["food_category"].isin(_MEAT_CATS)
    raw_mask  = foods["food_name"].str.contains(r"[,\.]\s*crua?\b", case=False,
                                                na=False, regex=True)

    # conjunto de "base names" dos itens cozidos (sem o último descritor de preparo)
    # normalizado para ignorar variações de gordura (com/sem gordura)
    cooked_bases: set[str] = set()
    for _, row in foods[meat_mask].iterrows():
        if _COOKED_RE.search(row["food_name"]):
            parts = [p.strip() for p in row["food_name"].split(",")]
            if len(parts) > 1:
                base = _normalize_meat_base(", ".join(parts[:-1]).lower())
                cooked_bases.add(base)

    to_remove = pd.Series(False, index=foods.index)
    for idx, row in foods[meat_mask & raw_mask].iterrows():
        parts = [p.strip() for p in row["food_name"].split(",")]
        if len(parts) > 1:
            base = _normalize_meat_base(", ".join(parts[:-1]).lower())
            if base in cooked_bases:
                to_remove[idx] = True
    return to_remove


# ── caminhos ──────────────────────────────────────────────────────────────────
_BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(_BASE, "models", "preference_model.pkl")
ENC_PATH    = os.path.join(_BASE, "models", "label_encoders.pkl")
FOODS_PATH  = os.path.join(_BASE, "data", "processed", "unified_foods.csv")

FEATURES = [
    "age_group_enc", "region_enc", "food_category_enc",
    "energy_kcal", "protein_g", "fat_g", "carbs_g",
    "NAS", "source_enc",
]

# ── tipos ─────────────────────────────────────────────────────────────────────
@dataclass
class FoodCandidate:
    food_id:        str
    food_name:      str
    food_category:  str
    source:         str
    preference_score: float   # probabilidade do modelo [0-1]
    NAS:            float     # Nutritional Affordability Score para a faixa
    energy_kcal:    float
    protein_g:      float
    fat_g:          float
    carbs_g:        float
    iron_mg:        float
    zinc_mg:        float
    calcium_mg:     float
    vitA_ug:        float
    vitC_mg:        float

    def to_dict(self) -> dict:
        return {
            "food_id":          self.food_id,
            "food_name":        self.food_name,
            "food_category":    self.food_category,
            "source":           self.source,
            "preference_score": round(self.preference_score, 4),
            "NAS":              round(self.NAS, 4),
            "energy_kcal":      self.energy_kcal,
            "protein_g":        self.protein_g,
            "fat_g":            self.fat_g,
            "carbs_g":          self.carbs_g,
            "iron_mg":          self.iron_mg,
            "zinc_mg":          self.zinc_mg,
            "calcium_mg":       self.calcium_mg,
            "vitA_ug":          self.vitA_ug,
            "vitC_mg":          self.vitC_mg,
        }


@dataclass
class FoodCatalog:
    age_group:        str
    age_group_label:  str
    region:           str
    ranked:           list[FoodCandidate]          # todos os alimentos, ordenados
    top_k:            list[FoodCandidate]           # top-K
    by_category:      dict[str, list[FoodCandidate]]  # categoria -> top alimentos

    def summary(self) -> str:
        cats = {c: len(v) for c, v in self.by_category.items()}
        return (
            f"FoodCatalog(faixa={self.age_group_label}, região={self.region}, "
            f"total={len(self.ranked)}, top_k={len(self.top_k)}, "
            f"categorias={len(cats)})"
        )


# ── agente ────────────────────────────────────────────────────────────────────
class FoodRetrieverAgent:
    """Pontua e ranqueia alimentos por preferência para uma criança específica."""

    def __init__(self, top_k: int = 50, top_per_category: int = 10):
        self.top_k = top_k
        self.top_per_category = top_per_category
        self._model    = joblib.load(MODEL_PATH)
        self._encoders = joblib.load(ENC_PATH)
        foods = pd.read_csv(FOODS_PATH)

        # Alimentos explicitamente excluídos por inadequação para alimentação
        # complementar infantil (alta gordura saturada, alto sódio ou perfil
        # inadequado mesmo que tecnicamente processados/in natura).
        _BLACKLIST_IDS = {
            "TACO_447",  # Creme de Leite — gordura saturada elevada (22g/100g); não é alimento complementar
            "TACO_263",  # Margarina c/ óleo hidrogenado c/ sal
            "TACO_264",  # Margarina c/ óleo hidrogenado s/ sal
            "TACO_265",  # Margarina c/ óleo interesterificado c/ sal
            "TACO_266",  # Margarina c/ óleo interesterificado s/ sal
            "TACO_424",  # Mortadela — embutido processado, alto sódio
            "TACO_438",  # Presunto com capa de gordura — embutido, alto sódio
            "TACO_439",  # Presunto sem capa de gordura — embutido, alto sódio
            "TACO_443",  # Salame — embutido processado, alto sódio
            # TACO itens crus cujo par cozido existe apenas na base TBCA (nomes divergentes
            # impedem a detecção automática em _raw_meats_with_cooked_version):
            "TACO_391",  # Frango, asa, com pele, crua — versão cozida: TBCA_C0740F/C0742F
            "TACO_330",  # Carne, bovina, almôndegas, cruas — forma cozida típica; inadequado cru
            "TACO_402",  # Frango, inteiro, com pele, cru — versão inteira crua; cortes cozidos disponíveis
            "TACO_42",   # Milho, amido, cru (maizena) — ingrediente culinário, não alimento complementar infantil
            # Farinhas cruas: ingredientes culinários, não alimentos complementares
            "TACO_31",   # Farinha de arroz enriquecida
            "TACO_32",   # Farinha de centeio integral
            "TACO_33",   # Farinha de milho amarela
            "TACO_34",   # Farinha de rosca
            "TACO_35",   # Farinha de trigo
            "TACO_36",   # Farinha láctea de cereais
            "TACO_121",  # Farinha de mandioca crua
            "TACO_122",  # Farinha de mandioca torrada
            "TACO_123",  # Farinha de puba
            "TACO_581",  # Soja farinha
            "TACO_592",  # Farinha de mesocarpo de babaçu
            "TBCA_C0022A",  # Aveia farinha crua
            # Condimento processado — alto sódio, não é alimento complementar
            "TACO_159",  # Tomate, molho industrializado
        }
        _blacklist_mask = foods["food_id"].isin(_BLACKLIST_IDS)

        # Remove leguminosas cruas: valores nutricionais não refletem o consumo real
        # (pós-cozimento) e o consumo cru é inadequado para crianças pequenas.
        # Versões cozidas estão disponíveis no banco para os itens relevantes.
        _raw_legume = (
            (foods["food_category"] == "Leguminosas e derivados") &
            foods["food_name"].str.contains(r"[,\.]\s*cru", case=False, na=False, regex=True)
        )

        # Remove ovos crus: versão cozida (TACO_488) disponível; ovo cru é
        # inadequado para crianças pequenas (segurança alimentar — Salmonella).
        _raw_egg = (
            (foods["food_category"] == "Ovos e derivados") &
            foods["food_name"].str.contains(r"[,\.]\s*cru", case=False, na=False, regex=True)
        )

        # Remove carnes/aves cruas que possuem versão cozida no banco.
        # Itens sem par cozido (ex: fígado de frango) são preservados.
        _raw_meat = _raw_meats_with_cooked_version(foods)

        # Remove TODOS os pescados crus: peixe cru é inadequado para crianças
        # < 5 anos independente de existir versão cozida (Listeria, parasitas).
        _raw_fish = (
            (foods["food_category"] == "Pescados e frutos do mar") &
            foods["food_name"].str.contains(r"[,\.]\s*cru", case=False, na=False, regex=True)
        )

        # Remove tubérculos amiláceos crus (batata, inhame, mandioca, cará):
        # segurança alimentar — tuberosas cruas são inadequadas para crianças pequenas.
        # Cobre itens classificados em qualquer categoria.
        _raw_tuber = (
            foods["food_name"].str.contains(
                r"\b(?:batata|inhame|mandioca|car[aá])\b", case=False, na=False, regex=True
            ) &
            foods["food_name"].str.contains(r"[,\.]\s*cru", case=False, na=False, regex=True)
        )

        # Remove TODAS as hortaliças cruas e de salada da categoria "Verduras, hortaliças e
        # derivados": devem ser cozidas antes do consumo por crianças pequenas. Abrange
        # cenoura, tomate, alface, agrião, pepino, beterraba, brócolis e todos os demais.
        _raw_hard_veg = (
            (foods["food_category"] == "Verduras, hortaliças e derivados") &
            foods["food_name"].str.contains(r"[,\.]\s*cru|[,\.]\s*salada\b", case=False, na=False, regex=True)
        )

        # Remove alimentos fritos: fritura não é método de preparo recomendado para
        # crianças < 5 anos (acrilamida, alta densidade calórica por gordura de adição).
        # Padrão ",\s*frit[ao]" captura frita/frito após vírgula (método de preparo isolado)
        # sem afetar itens TBCA com "grelhado/frito" (precedido por "/" não por vírgula).
        # "milanesa" é incluído pois o preparo padrão é frito empanado.
        # Fonte: Guia Alimentar para Crianças Brasileiras Menores de 2 Anos (MS, 2019).
        _fried = foods["food_name"].str.contains(
            r",\s*frit[ao]\b|\bmilanesa\b", case=False, na=False, regex=True
        )

        # Remove TODOS os alimentos crus, independente de categoria:
        # cobre cereais crus (macarrão, polenta, pastel cru), tubérculos, etc.
        # O padrão "[,\.]\s*cru" captura "cru", "crua", "cruas" após vírgula (TACO)
        # ou ponto (TBCA), que é o marcador de preparo em ambas as bases.
        _raw_any = foods["food_name"].str.contains(r"[,\.]\s*cru", case=False, na=False, regex=True)

        _eligible = ~_raw_legume & ~_raw_egg & ~_raw_meat & ~_raw_fish & ~_raw_tuber & ~_raw_hard_veg & ~_fried & ~_blacklist_mask & ~_raw_any

        # Exclui alimentos ultraprocessados (NOVA Grupo 4).
        # Classificação baseada em Monteiro et al. (2019) e Guia Alimentar MS (2014).
        if "nova_group" in foods.columns:
            _ultra = foods["nova_group"] == 4
            _eligible = _eligible & ~_ultra

        self._foods = foods[_eligible].reset_index(drop=True)

        self._assessment_agent = AssessmentAgent()
        self._precompute_food_features()

    # ── pré-computação ────────────────────────────────────────────────────────
    def _precompute_food_features(self):
        """Pré-codifica as features fixas dos alimentos (independem da criança)."""
        foods = self._foods
        enc   = self._encoders

        def safe_encode(encoder_key, series):
            le    = enc[encoder_key]
            known = set(le.classes_)
            mapped = series.fillna("unknown").astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            return le.transform(mapped)

        self._food_cat_enc = safe_encode("food_category",
                                         foods["food_category"])
        self._food_src_enc = safe_encode("source", foods["source"])
        self._food_numeric = foods[
            ["energy_kcal", "protein_g", "fat_g", "carbs_g"]
        ].fillna(0).values.astype(np.float32)

        self._food_ids   = foods["food_id"].tolist()
        self._food_names = foods["food_name"].fillna("").tolist()
        self._food_cats  = foods["food_category"].fillna("").tolist()
        self._food_srcs  = foods["source"].fillna("").tolist()
        self._n_foods    = len(foods)

        # nutrientes extras para o LP
        extra_cols = ["iron_mg", "zinc_mg", "calcium_mg", "vitA_ug", "vitC_mg"]
        self._food_extras = {}
        for col in extra_cols:
            self._food_extras[col] = (
                foods[col].fillna(0).values.astype(np.float32)
                if col in foods.columns
                else np.zeros(self._n_foods, dtype=np.float32)
            )

    # ── scoring ───────────────────────────────────────────────────────────────
    def _score_foods(self, assessment: AssessmentResult) -> np.ndarray:
        """Retorna array de probabilidades [0-1] para cada alimento."""
        enc = self._encoders
        ag  = assessment.age_group
        # 24_35m e 36_59m não existem no encoder do modelo — mapear para 24_59m
        ag_model = "24_59m" if ag in ("24_35m", "36_59m") else ag
        reg = assessment.child_profile.get("region", "unknown") or "unknown"

        def safe_enc_scalar(key, val):
            le    = enc[key]
            known = set(le.classes_)
            v     = val if val in known else le.classes_[0]
            return int(le.transform([v])[0])

        age_enc = safe_enc_scalar("age_group", ag_model)
        reg_enc = safe_enc_scalar("region", str(reg))

        nas_col = f"NAS_{ag}"
        nas_vals = (
            self._foods[nas_col].fillna(0).values.astype(np.float32)
            if nas_col in self._foods.columns
            else np.zeros(self._n_foods, dtype=np.float32)
        )

        # matriz (n_foods × 9)
        X = np.column_stack([
            np.full(self._n_foods, age_enc, dtype=np.float32),
            np.full(self._n_foods, reg_enc, dtype=np.float32),
            self._food_cat_enc.astype(np.float32),
            self._food_numeric,          # energy, protein, fat, carbs
            nas_vals,
            self._food_src_enc.astype(np.float32),
        ])

        return self._model.predict_proba(X)[:, 1]

    # ── montagem do catálogo ──────────────────────────────────────────────────
    def _build_catalog(self, scores: np.ndarray,
                       assessment: AssessmentResult) -> FoodCatalog:
        ag      = assessment.age_group
        age_m   = float(assessment.child_profile.get("age_months", 24))
        nas_col = f"NAS_{ag}"
        nas_arr = (
            self._foods[nas_col].fillna(0).values
            if nas_col in self._foods.columns
            else np.zeros(self._n_foods)
        )

        candidates = []
        for i in range(self._n_foods):
            name = self._food_names[i]
            # filtros por faixa etária (Guia MS 2019)
            skip = False
            for max_age, pattern in _AGE_FILTERS:
                if age_m < max_age and pattern.search(name):
                    skip = True
                    break
            if skip:
                continue
            num = self._food_numeric[i]
            cand = FoodCandidate(
                food_id=self._food_ids[i],
                food_name=self._food_names[i],
                food_category=self._food_cats[i],
                source=self._food_srcs[i],
                preference_score=float(scores[i]),
                NAS=float(nas_arr[i]),
                energy_kcal=float(num[0]),
                protein_g=float(num[1]),
                fat_g=float(num[2]),
                carbs_g=float(num[3]),
                iron_mg=float(self._food_extras["iron_mg"][i]),
                zinc_mg=float(self._food_extras["zinc_mg"][i]),
                calcium_mg=float(self._food_extras["calcium_mg"][i]),
                vitA_ug=float(self._food_extras["vitA_ug"][i]),
                vitC_mg=float(self._food_extras["vitC_mg"][i]),
            )
            candidates.append(cand)

        ranked = sorted(candidates, key=lambda c: c.preference_score, reverse=True)
        top_k  = ranked[: self.top_k]

        # agrupamento por categoria
        by_cat: dict[str, list[FoodCandidate]] = {}
        for cand in ranked:
            cat = cand.food_category or "Outros"
            by_cat.setdefault(cat, [])
            if len(by_cat[cat]) < self.top_per_category:
                by_cat[cat].append(cand)

        region = assessment.child_profile.get("region", "N/A") or "N/A"
        return FoodCatalog(
            age_group=ag,
            age_group_label=assessment.age_group_label,
            region=str(region),
            ranked=ranked,
            top_k=top_k,
            by_category=by_cat,
        )

    # ── métodos públicos ──────────────────────────────────────────────────────
    def retrieve(self, assessment: AssessmentResult) -> FoodCatalog:
        """Pontua e ranqueia alimentos dado um AssessmentResult."""
        scores = self._score_foods(assessment)
        return self._build_catalog(scores, assessment)

    def retrieve_from_profile(self, child_profile: dict) -> FoodCatalog:
        """Avalia o perfil e ranqueia alimentos em uma única chamada."""
        assessment = self._assessment_agent.assess(child_profile)
        return self.retrieve(assessment)

    def retrieve_dict(self, child_profile: dict, include_full: bool = False) -> dict:
        """Retorna dicionário serializável em JSON."""
        cat = self.retrieve_from_profile(child_profile)
        result = {
            "age_group":       cat.age_group,
            "age_group_label": cat.age_group_label,
            "region":          cat.region,
            "total_foods":     len(cat.ranked),
            "top_k":           [c.to_dict() for c in cat.top_k],
            "by_category":     {
                k: [c.to_dict() for c in v]
                for k, v in cat.by_category.items()
            },
        }
        if include_full:
            result["ranked"] = [c.to_dict() for c in cat.ranked]
        return result
