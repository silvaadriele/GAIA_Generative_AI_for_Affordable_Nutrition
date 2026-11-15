#!/usr/bin/env python3
"""
Fase 3: Treina o modelo LightGBM de preferencia alimentar infantil.

Entradas:
  data/raw/enani_recall_24h_clean.csv  - 477k registros de consumo
  data/raw/enani_profiles.csv          - 14.558 perfis de criancas
  data/processed/unified_foods.csv     - 619 alimentos elegiveis com NAS

Saidas:
  models/preference_model.pkl          - modelo LightGBM treinado
  models/label_encoders.pkl            - encoders para categorias
  models/feature_importance.csv        - importancia das features
  evaluation/results/phase3_metrics.json - metricas de avaliacao
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

TACO_MAPPING_PATH = "data/reference/recall_taco_mapping.json"

# ── caminhos ──────────────────────────────────────────────────────────────────
RECALL_PATH   = "data/raw/enani_recall_24h_clean.csv"
PROFILES_PATH = "data/raw/enani_profiles.csv"
FOODS_PATH    = "data/processed/unified_foods.csv"
MODEL_OUT     = "models/preference_model.pkl"
ENC_OUT       = "models/label_encoders.pkl"
FEAT_OUT      = "models/feature_importance.csv"
METRICS_OUT   = "evaluation/results/phase3_metrics.json"

NEG_RATIO   = 5     # amostras negativas por positiva por crianca
RANDOM_SEED = 42
TOP_K       = 20    # k para Precision@K

# ordem das features deve ser identica em treino e inferencia
FEATURES = [
    "age_group_enc", "region_enc", "food_category_enc",
    "energy_kcal", "protein_g", "fat_g", "carbs_g",
    "NAS", "source_enc",
]


# ── utilitarios ───────────────────────────────────────────────────────────────
def age_to_group(months_str):
    """Converte '22 meses' -> 'age_group' label."""
    try:
        m = int(str(months_str).split()[0])
    except (ValueError, AttributeError):
        return None
    if m < 6:  return "0_5m"
    if m < 9:  return "6_8m"
    if m < 12: return "9_11m"
    if m < 24: return "12_23m"
    if m < 60: return "24_59m"
    return None


def encode_features(df, encoders=None):
    """Label-encode colunas categoricas; ajusta no primeiro uso."""
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    cat_cols = ["age_group", "region", "food_category", "source"]
    for col in cat_cols:
        enc_key = col
        vals = df[col].fillna("unknown").astype(str)
        if fit_mode:
            le = LabelEncoder()
            df[f"{enc_key}_enc"] = le.fit_transform(vals)
            encoders[enc_key] = le
        else:
            le = encoders[enc_key]
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else le.classes_[0])
            df[f"{enc_key}_enc"] = le.transform(vals)

    return df, encoders


# ── etapa 1: carregar dados ───────────────────────────────────────────────────
def load_data():
    print("=== 1. Carregando dados ===")

    # perfis
    profiles = pd.read_csv(
        PROFILES_PATH,
        usecols=["id_anon", "b05a_idade_em_meses", "a00_regiao"],
        low_memory=False,
    )
    profiles["age_group"] = profiles["b05a_idade_em_meses"].apply(age_to_group)
    profiles = profiles[profiles["age_group"].notna()].copy()
    profiles["region"] = profiles["a00_regiao"].fillna("Desconhecido")
    profiles = profiles[["id_anon", "age_group", "region"]]
    print(f"  Perfis validos: {len(profiles)}")
    print(f"  Distribuicao por faixa: {profiles['age_group'].value_counts().to_dict()}")

    # recordatorio — combina match direto TBCA + match fuzzy TACO
    df_recall = pd.read_csv(
        RECALL_PATH,
        usecols=["id_anon", "cod_tbca", "rr24_alimento_descricao"],
        low_memory=False,
    )
    df_recall["tbca_food_id"] = df_recall["cod_tbca"].apply(
        lambda x: f"TBCA_{x}" if pd.notna(x) else None
    )
    if os.path.exists(TACO_MAPPING_PATH):
        with open(TACO_MAPPING_PATH, encoding="utf-8") as f:
            taco_mapping = json.load(f)
        df_recall["taco_food_id"] = df_recall["rr24_alimento_descricao"].map(
            lambda d: taco_mapping.get(d, {}).get("taco_food_id") if pd.notna(d) else None
        )
        n_taco = df_recall["taco_food_id"].notna().sum()
        print(f"  Match TACO (fuzzy): {n_taco} registros")
    else:
        df_recall["taco_food_id"] = None
        print("  AVISO: recall_taco_mapping.json nao encontrado")

    df_tbca = df_recall[df_recall["tbca_food_id"].notna()][["id_anon", "tbca_food_id"]].rename(
        columns={"tbca_food_id": "food_id"}
    )
    df_taco = df_recall[df_recall["taco_food_id"].notna()][["id_anon", "taco_food_id"]].rename(
        columns={"taco_food_id": "food_id"}
    )
    combined = pd.concat([df_tbca, df_taco], ignore_index=True)
    recall_consumed = (
        combined.groupby("id_anon")["food_id"]
        .apply(set)
        .reset_index()
        .rename(columns={"food_id": "consumed_foods"})
    )
    print(f"  Criancas com consumo registrado: {recall_consumed['id_anon'].nunique()}")

    # banco de alimentos
    foods = pd.read_csv(FOODS_PATH)
    print(f"  Alimentos no banco: {len(foods)}")

    # diagnostico de sobreposicao
    all_recall_ids = set(combined["food_id"])
    tbca_ids = {f for f in all_recall_ids if f.startswith("TBCA_")}
    taco_ids = {f for f in all_recall_ids if f.startswith("TACO_")}
    tbca_in_bank = tbca_ids & set(foods["food_id"])
    taco_in_bank = taco_ids & set(foods["food_id"])
    print(f"  Foods do recall no banco: TBCA={len(tbca_in_bank)} | TACO={len(taco_in_bank)}")

    return profiles, recall_consumed, foods


# ── etapa 2: construir amostras ───────────────────────────────────────────────
def build_samples(profiles, recall_consumed, foods):
    """Para cada crianca: positivos (consumidos) + negativos amostrados."""
    print("\n=== 2. Construindo amostras de treino ===")

    rng = np.random.default_rng(RANDOM_SEED)
    food_dict  = foods.set_index("food_id").to_dict("index")
    all_food_ids = list(food_dict.keys())
    all_food_set = set(all_food_ids)

    merged = profiles.merge(recall_consumed, on="id_anon", how="inner")
    print(f"  Criancas com perfil + consumo: {len(merged)}")

    rows = []
    for i, row in enumerate(merged.itertuples(index=False)):
        if i % 2000 == 0:
            print(f"  Processando {i}/{len(merged)}...", end="\r")

        age_group    = row.age_group
        region       = row.region
        consumed_ids = row.consumed_foods & all_food_set
        nas_col      = f"NAS_{age_group}"

        if not consumed_ids:
            continue

        # positivos
        for fid in consumed_ids:
            f = food_dict[fid]
            rows.append((
                row.id_anon, age_group, region, fid,
                f["food_category"], f["source"],
                f.get("energy_kcal", np.nan), f.get("protein_g", np.nan),
                f.get("fat_g", np.nan), f.get("carbs_g", np.nan),
                f.get(nas_col, np.nan), 1,
            ))

        # negativos amostrados
        non_consumed = list(all_food_set - consumed_ids)
        n_neg = min(len(non_consumed), len(consumed_ids) * NEG_RATIO)
        for fid in rng.choice(non_consumed, size=n_neg, replace=False):
            f = food_dict[fid]
            rows.append((
                row.id_anon, age_group, region, fid,
                f["food_category"], f["source"],
                f.get("energy_kcal", np.nan), f.get("protein_g", np.nan),
                f.get("fat_g", np.nan), f.get("carbs_g", np.nan),
                f.get(nas_col, np.nan), 0,
            ))

    print()
    cols = [
        "child_id", "age_group", "region", "food_id",
        "food_category", "source",
        "energy_kcal", "protein_g", "fat_g", "carbs_g", "NAS", "label",
    ]
    df = pd.DataFrame(rows, columns=cols)
    n_pos = int(df["label"].sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"  Total: {len(df)} amostras | pos={n_pos} | neg={n_neg} | ratio=1:{n_neg//max(n_pos,1)}")
    return df


# ── etapa 3: treinar ──────────────────────────────────────────────────────────
def train(samples):
    print("\n=== 3. Codificando features e treinando ===")

    samples, encoders = encode_features(samples)

    # split 80/20 estratificado por (age_group x label)
    strat_key = samples["age_group_enc"].astype(str) + "_" + samples["label"].astype(str)
    train_idx, test_idx = train_test_split(
        samples.index, test_size=0.2, random_state=RANDOM_SEED, stratify=strat_key
    )
    train_df = samples.loc[train_idx]
    test_df  = samples.loc[test_idx]
    print(f"  Treino: {len(train_df)} | Teste: {len(test_df)}")

    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df["label"]
    X_test  = test_df[FEATURES].fillna(0)
    y_test  = test_df["label"]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=NEG_RATIO,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"  Iteracoes usadas: {model.best_iteration_}")

    return model, encoders, train_df, test_df, X_test, y_test


# ── etapa 4: avaliar ──────────────────────────────────────────────────────────
def evaluate(model, encoders, test_df, X_test, y_test, recall_consumed, profiles, foods):
    print("\n=== 4. Avaliando ===")

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC geral
    auc_overall = roc_auc_score(y_test, y_pred_proba)
    print(f"  AUC-ROC geral: {auc_overall:.4f}")

    # AUC-ROC por faixa etaria
    auc_by_age = {}
    for ag in sorted(test_df["age_group"].unique()):
        mask = test_df["age_group"] == ag
        if mask.sum() < 20:
            continue
        auc_ag = roc_auc_score(y_test[mask], y_pred_proba[mask])
        auc_by_age[ag] = round(float(auc_ag), 4)
        print(f"    AUC {ag}: {auc_ag:.4f}")

    # Precision@20 — vetorizado por faixa etaria
    print(f"\n  Calculando Precision@{TOP_K} (amostra de criancas do teste)...")
    p_at_k = compute_precision_at_k(
        model, encoders, test_df, recall_consumed, profiles, foods, k=TOP_K
    )
    print(f"  Precision@{TOP_K} media: {p_at_k:.4f}")

    return auc_overall, auc_by_age, p_at_k


def compute_precision_at_k(model, encoders, test_df, recall_consumed, profiles, foods, k=20):
    """
    Para cada crianca de teste: pontua todos os 619 alimentos,
    ranqueia e verifica quantos dos top-k estao no consumo real.
    Vetorizado por crianca para eficiencia.
    """
    # pre-codificar features fixas dos alimentos (mesmo valor para toda crianca)
    food_cat_vals  = foods["food_category"].fillna("unknown").astype(str)
    food_src_vals  = foods["source"].fillna("unknown").astype(str)
    known_cat = set(encoders["food_category"].classes_)
    known_src = set(encoders["source"].classes_)
    food_cat_vals = food_cat_vals.apply(lambda x: x if x in known_cat else encoders["food_category"].classes_[0])
    food_src_vals = food_src_vals.apply(lambda x: x if x in known_src else encoders["source"].classes_[0])
    food_cat_enc = encoders["food_category"].transform(food_cat_vals)
    food_src_enc = encoders["source"].transform(food_src_vals)
    food_numeric = foods[["energy_kcal", "protein_g", "fat_g", "carbs_g"]].fillna(0).values
    all_food_ids = np.array(foods["food_id"].tolist())
    n_foods = len(all_food_ids)

    # mapa consumo real por crianca
    recall_dict = recall_consumed.set_index("id_anon")["consumed_foods"].to_dict()
    profiles_idx = profiles.set_index("id_anon")

    # pegar amostra de criancas do teste (max 500 para velocidade)
    test_child_ids = list(test_df["child_id"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    if len(test_child_ids) > 500:
        test_child_ids = rng.choice(test_child_ids, 500, replace=False).tolist()

    prec_scores = []
    for child_id in test_child_ids:
        if child_id not in profiles_idx.index:
            continue
        consumed_ids = recall_dict.get(child_id, set())
        if not consumed_ids:
            continue

        age_group = profiles_idx.loc[child_id, "age_group"]
        region    = profiles_idx.loc[child_id, "region"]
        nas_col   = f"NAS_{age_group}"

        known_age = set(encoders["age_group"].classes_)
        known_reg = set(encoders["region"].classes_)
        age_enc = encoders["age_group"].transform(
            [age_group if age_group in known_age else encoders["age_group"].classes_[0]]
        )[0]
        reg_enc = encoders["region"].transform(
            [region if region in known_reg else encoders["region"].classes_[0]]
        )[0]

        nas_vals = foods[nas_col].fillna(0).values if nas_col in foods.columns else np.zeros(n_foods)

        # matriz (n_foods x 9 features)
        X = np.column_stack([
            np.full(n_foods, age_enc),   # age_group_enc
            np.full(n_foods, reg_enc),   # region_enc
            food_cat_enc,                # food_category_enc
            food_numeric,                # energy, protein, fat, carbs
            nas_vals,                    # NAS
            food_src_enc,                # source_enc
        ]).astype(np.float32)

        preds   = model.predict_proba(X)[:, 1]
        top_idx = np.argsort(preds)[-k:]
        top_ids = set(all_food_ids[top_idx])
        hit = len(top_ids & consumed_ids)
        prec_scores.append(hit / k)

    return float(np.mean(prec_scores)) if prec_scores else 0.0


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation/results", exist_ok=True)

    profiles, recall_consumed, foods = load_data()
    samples = build_samples(profiles, recall_consumed, foods)
    model, encoders, train_df, test_df, X_test, y_test = train(samples)
    auc_overall, auc_by_age, p_at_k = evaluate(
        model, encoders, test_df, X_test, y_test, recall_consumed, profiles, foods
    )

    # feature importance
    fi = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    print("\n  Importancia das features:")
    print(fi.to_string(index=False))

    # salvar artefatos
    print("\n=== 5. Salvando artefatos ===")
    joblib.dump(model, MODEL_OUT)
    print(f"  Modelo: {MODEL_OUT}")

    joblib.dump(encoders, ENC_OUT)
    print(f"  Encoders: {ENC_OUT}")

    fi.to_csv(FEAT_OUT, index=False)
    print(f"  Feature importance: {FEAT_OUT}")

    metrics = {
        "auc_roc_overall": round(float(auc_overall), 4),
        "auc_roc_by_age_group": auc_by_age,
        "precision_at_k": p_at_k,
        "k": TOP_K,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_positives": int(samples["label"].sum()),
        "n_negatives": int((samples["label"] == 0).sum()),
        "neg_ratio": NEG_RATIO,
        "best_iteration": int(model.best_iteration_),
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas: {METRICS_OUT}")

    print("\n=== FASE 3 CONCLUIDA ===")
    print(f"  AUC-ROC: {auc_overall:.4f} | Precision@{TOP_K}: {p_at_k:.4f}")
