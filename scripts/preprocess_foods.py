import os
import sys
import pandas as pd
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.food_prices import get_price

TACO_PATH  = "data/raw/TACO.xlsx"
TBCA_PATH  = "data/raw/TBCA_BR.csv"
DRI_PATH   = "data/reference/dri.json"
OUT_PATH   = "data/processed/unified_foods.csv"

# nutrientes usados no NAS (folato ausente no TACO)
NAS_NUTRIENTS = ["iron_mg", "zinc_mg", "calcium_mg", "vitA_ug", "vitC_mg", "protein_g"]

REMOVE_CATEGORIES = {
    "bebidas (alco",        # bebidas alcoólicas e não alcoólicas
    "produtos a",           # produtos açucarados
    "outros alimentos",     # outros alimentos industrializados
    "miscel",               # miscellâneas
    "alimentos preparados", # alimentos preparados
}

AGE_GROUPS = ["6_8m", "9_11m", "12_23m", "24_35m", "36_59m"]


# ── 1. TACO ──────────────────────────────────────────────────────────────────
def load_taco():
    raw = pd.read_excel(TACO_PATH, sheet_name="CMVCol taco3", header=None)

    # mapear categorias: percorrer linhas e atribuir categoria ao alimento
    col_map = {
        "food_id":    0,
        "food_name":  1,
        "energy_kcal": 3,
        "protein_g":  5,
        "fat_g":      6,
        "carbs_g":    8,
        "calcium_mg": 11,
        "iron_mg":    16,
        "zinc_mg":    20,
        "vitA_ug":    23,
        "vitC_mg":    28,
        "sodium_mg":  17,
    }

    records = []
    current_category = None

    for _, row in raw.iterrows():
        val0 = row[0]
        val1 = row[1]
        val3 = row[3]

        # linha de categoria: col0 tem texto, col1 e col3 são NaN
        if pd.notna(val0) and pd.isna(val1) and pd.isna(val3):
            current_category = str(val0).strip()
            continue

        # linha de dados: col0 é numérico
        try:
            food_id = int(float(val0))
        except (ValueError, TypeError):
            continue

        rec = {"food_id": f"TACO_{food_id}", "food_category": current_category, "source": "TACO"}
        for field, col in col_map.items():
            if field == "food_id":
                continue
            v = row[col]
            if field == "food_name":
                rec[field] = str(v).strip() if pd.notna(v) else np.nan
                continue
            # "Tr" (traço) → 0.0
            if isinstance(v, str) and v.strip().lower() in ("tr", "nd", "*"):
                v = 0.0
            try:
                rec[field] = float(v)
            except (ValueError, TypeError):
                rec[field] = np.nan
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"TACO carregado: {len(df)} alimentos")
    return df


# ── 2. TBCA ──────────────────────────────────────────────────────────────────
def load_tbca():
    raw = pd.read_csv(TBCA_PATH, encoding="latin-1")

    # Fix column name encoding: TBCA CSV is UTF-8 but read as Latin-1,
    # so accented chars in column headers are double-encoded mojibake.
    # Example: "Proteína" (UTF-8 \xc3\xad) → "ProteÃ­na" (Latin-1 U+00C3 U+00AD).
    # Re-encoding column names: Latin-1 bytes → UTF-8 decode restores them.
    raw.columns = [
        col.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
        for col in raw.columns
    ]

    # normalizar separador decimal (vírgula → ponto)
    for col in raw.columns:
        if raw[col].dtype == object:
            raw[col] = raw[col].astype(str).str.replace(",", ".", regex=False)
            try:
                raw[col] = pd.to_numeric(raw[col], errors="ignore")
            except Exception:
                pass

    tbca_map = {
        "food_name":   "ALIMENTO",
        "energy_kcal": "Energia Kcal",
        "protein_g":   "Proteína",
        "fat_g":       "Lipídios",
        "carbs_g":     "Carboidrato total",
        "calcium_mg":  "Cálcio",
        "iron_mg":     "Ferro",
        "zinc_mg":     "Zinco",
        "vitA_ug":     "Vitamina A (RAE)",
        "vitC_mg":     "Vitamina C",
        "sodium_mg":   "Sódio",
    }

    records = []
    for i, row in raw.iterrows():
        rec = {
            "food_id":       f"TBCA_{row.get('ID', i)}",
            "food_category": "TBCA",
            "source":        "TBCA",
        }
        for field, col in tbca_map.items():
            if col not in raw.columns:
                rec[field] = np.nan
                continue
            v = row[col]
            if field == "food_name":
                rec[field] = str(v).strip() if pd.notna(v) and str(v).strip().lower() != "nan" else np.nan
                continue
            if isinstance(v, str) and v.strip().lower() in ("tr", "nd", "*", "nan"):
                v = 0.0
            try:
                rec[field] = float(v)
            except (ValueError, TypeError):
                rec[field] = np.nan
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"TBCA carregado: {len(df)} alimentos")
    return df


# ── 3. Filtrar processados ────────────────────────────────────────────────────
def filter_eligible(df):
    def is_eligible(cat):
        if pd.isna(cat):
            return True
        cat_lower = str(cat).lower()
        for kw in REMOVE_CATEGORIES:
            if cat_lower.startswith(kw):
                return False
        return True

    df["is_eligible"] = df["food_category"].apply(is_eligible)
    before = len(df)
    df = df[df["is_eligible"]].reset_index(drop=True)
    print(f"Filtro processados: {before} -> {len(df)} alimentos elegiveis")
    return df


# ── 4. NAS por faixa etária ───────────────────────────────────────────────────
def compute_nas(df, dri):
    """
    NAS = Nutritional Affordability Score = densidade nutricional / preço.

    Metodologia: adaptação pediátrica do NRF9.3/$ (Drewnowski & Fulgoni, JAND 2008).
    - NAS_pos: média dos ratios nutriente/DRI para 6 micronutrientes prioritários
      para a faixa etária 6-59 meses (ferro, zinco, cálcio, vitamina A, vitamina C,
      proteína), usando DRIs pediátricos por faixa (MS/WHO) em vez de DVs adultos FDA.
    - LIM_sodium: min(1, sódio_por_100g / MRV_sódio_diário) — penaliza alimentos
      com alto teor de sódio (ex.: caldos concentrados, embutidos), seguindo o
      subscore LIM do NRF9.3.
    - Denominador: preço de varejo por 100g (Carrefour Online, nov/2026).
    - Resultado: NAS = max(0, NAS_pos - LIM_sodium) / preço — quanto maior, mais
      custo-efetivo e nutricionalmente adequado para a faixa etária.
    """
    # pré-computar preços (independe da faixa etária)
    prices = df["food_name"].apply(lambda n: max(get_price(str(n)), 0.01)).values

    for age_group in AGE_GROUPS:
        dri_vals = dri.get(age_group, {})
        # média M/F para DRI dos nutrientes positivos
        dri_avg = {}
        for nutrient in NAS_NUTRIENTS:
            vals = [dri_vals.get(sex, {}).get(nutrient, np.nan) for sex in ["M", "F"]]
            vals = [v for v in vals if not np.isnan(v)]
            dri_avg[nutrient] = np.mean(vals) if vals else np.nan

        # MRV do sódio (mesmo para M/F)
        sodium_mrv = dri_vals.get("M", {}).get("sodium_mrv_mg", np.nan)

        scores = []
        for idx, (_, row) in enumerate(df.iterrows()):
            # NAS_pos — nutrientes a encorajar
            ratios = []
            for n in NAS_NUTRIENTS:
                dri_n = dri_avg.get(n, np.nan)
                food_n = row.get(n, np.nan)
                if np.isnan(dri_n) or dri_n == 0 or np.isnan(food_n):
                    ratios.append(0.0)
                else:
                    ratios.append(min(1.0, food_n / dri_n))
            nas_pos = np.mean(ratios)

            # LIM_sodium — nutriente a limitar
            lim_sodium = 0.0
            if not np.isnan(sodium_mrv) and sodium_mrv > 0:
                food_sodium = row.get("sodium_mg", np.nan)
                if not np.isnan(food_sodium) and food_sodium > 0:
                    lim_sodium = min(1.0, food_sodium / sodium_mrv)

            nas_raw = max(0.0, nas_pos - lim_sodium)
            scores.append(nas_raw / prices[idx])

        df[f"NAS_{age_group}"] = scores
        print(f"  NAS_{age_group}: média={np.mean(scores):.4f}")

    return df


# ── 5. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Carregando dados ===")
    taco = load_taco()
    tbca = load_tbca()

    print("\n=== Unificando ===")
    foods = pd.concat([taco, tbca], ignore_index=True)
    print(f"Total antes do filtro: {len(foods)}")

    print("\n=== Filtrando processados ===")
    foods = filter_eligible(foods)

    print("\n=== Calculando NAS ===")
    with open(DRI_PATH, encoding="utf-8") as f:
        dri = json.load(f)
    foods = compute_nas(foods, dri)

    print(f"\n=== Salvando em {OUT_PATH} ===")
    foods.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Concluído: {len(foods)} alimentos salvos")
    print("\nColunas:", foods.columns.tolist())
