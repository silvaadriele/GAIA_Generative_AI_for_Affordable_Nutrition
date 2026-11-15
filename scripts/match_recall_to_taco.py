#!/usr/bin/env python3
"""
Fase 3b: Mapeamento fuzzy recall ENANI -> TACO.
Estrategia:
  1) Filtrar candidatos TACO que contem o ingrediente principal (1a palavra)
  2) Dentro do grupo: word overlap recall (quantas palavras da query estao no candidato)
  3) Tiebreaker: char n-gram TF-IDF cosine similarity
  4) Fallback global com char n-gram threshold alto (0.70)

Saida: data/reference/recall_taco_mapping.json
"""

import json
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RECALL_PATH = "data/raw/enani_recall_24h_clean.csv"
FOODS_PATH  = "data/processed/unified_foods.csv"
OUT_PATH    = "data/reference/recall_taco_mapping.json"

OVERLAP_THRESH  = 0.50   # minimo de palavras da query presentes no candidato
GLOBAL_THRESH   = 0.70   # char n-gram threshold para fallback global


def normalize(text):
    text = str(text).lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_overlap_score(query_tokens, candidate_tokens):
    """Fracao das palavras da query encontradas no candidato (recall-focused)."""
    if not query_tokens:
        return 0.0
    return len(set(query_tokens) & set(candidate_tokens)) / len(set(query_tokens))


def main():
    # ── carregar ──────────────────────────────────────────────────────────────
    recall = pd.read_csv(RECALL_PATH, usecols=["rr24_alimento_descricao"])
    unique_desc = recall["rr24_alimento_descricao"].dropna().unique().tolist()
    print(f"Descricoes unicas no recall: {len(unique_desc)}")

    foods = pd.read_csv(FOODS_PATH)
    taco = foods[foods["source"] == "TACO"][["food_id", "food_name"]].dropna().copy()
    taco["name_norm"] = taco["food_name"].apply(normalize)
    taco["name_tokens"] = taco["name_norm"].apply(str.split)
    taco_names_norm = taco["name_norm"].tolist()
    print(f"Alimentos TACO com nome: {len(taco)}")

    # ── TF-IDF global (char n-grams) para fallback ────────────────────────────
    desc_norm  = [normalize(d) for d in unique_desc]
    all_texts  = taco_names_norm + desc_norm
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    tfidf_all  = vectorizer.fit_transform(all_texts)
    taco_mat   = tfidf_all[: len(taco)]
    desc_mat   = tfidf_all[len(taco) :]
    sims_all   = cosine_similarity(desc_mat, taco_mat)  # (n_desc, n_taco)

    # ── indice: primeiro token -> indices TACO ────────────────────────────────
    first_token_index = {}
    for idx, tokens in enumerate(taco["name_tokens"]):
        for tok in tokens:
            first_token_index.setdefault(tok, []).append(idx)

    # ── matching ──────────────────────────────────────────────────────────────
    mapping = {}
    for i, (desc, desc_n) in enumerate(zip(unique_desc, desc_norm)):
        query_tokens = desc_n.split()
        if not query_tokens:
            continue

        best_idx, best_sim = -1, 0.0
        main_token = query_tokens[0]

        # Passo 1: candidatos pelo ingrediente principal
        candidates = first_token_index.get(main_token, [])
        if candidates:
            # word overlap para cada candidato
            overlaps = np.array([
                word_overlap_score(query_tokens, taco["name_tokens"].iloc[c])
                for c in candidates
            ])
            max_overlap = overlaps.max()

            if max_overlap >= OVERLAP_THRESH:
                # entre empatados no overlap, usar char n-gram como tiebreaker
                top_mask = overlaps == max_overlap
                top_cands = [candidates[j] for j, ok in enumerate(top_mask) if ok]
                char_scores = sims_all[i, top_cands]
                local_best = int(np.argmax(char_scores))
                best_idx = top_cands[local_best]
                best_sim = float(max_overlap)   # reportar overlap como score

        # Passo 2: fallback global com char n-gram (threshold alto)
        if best_idx == -1:
            g = int(np.argmax(sims_all[i]))
            if sims_all[i, g] >= GLOBAL_THRESH:
                best_idx = g
                best_sim = float(sims_all[i, g])

        if best_idx != -1:
            mapping[desc] = {
                "taco_food_id":   taco.iloc[best_idx]["food_id"],
                "taco_food_name": taco.iloc[best_idx]["food_name"],
                "similarity":     round(best_sim, 4),
            }

    print(f"\nDescricoes mapeadas: {len(mapping)}/{len(unique_desc)}")
    print(f"Alimentos TACO distintos cobertos: {len({v['taco_food_id'] for v in mapping.values()})}/{len(taco)}")

    # ── salvar ────────────────────────────────────────────────────────────────
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Mapeamento salvo: {OUT_PATH}")

    # ── amostras ──────────────────────────────────────────────────────────────
    items_sorted = sorted(mapping.items(), key=lambda x: -x[1]["similarity"])

    print("\n[overlap = 1.0 - match perfeito]")
    for desc, info in [(d, i) for d, i in items_sorted if i["similarity"] == 1.0][:15]:
        print(f"  '{desc}' -> '{info['taco_food_name']}'")

    print("\n[0.5 <= overlap < 1.0 - match parcial]")
    for desc, info in [(d, v) for d, v in items_sorted if 0.5 <= v["similarity"] < 1.0][:15]:
        print(f"  '{desc}' ({info['similarity']}) -> '{info['taco_food_name']}'")

    no_match = [d for d in unique_desc if d not in mapping]
    print(f"\n[SEM MATCH: {len(no_match)}] - amostra:")
    for d in no_match[:15]:
        print(f"  '{d}'")

    print("\n[Verificacao alimentos chave]")
    for key in ["ARROZ BRANCO", "FEIJAO COZIDO", "BANANA", "FRANGO COZIDO",
                "OVO COZIDO", "BATATA COZIDA", "LEITE MATERNO", "CARNE MOIDA",
                "ARROZ COZIDO", "FEIJAO PRETO COZIDO"]:
        info = mapping.get(key)
        if info:
            print(f"  '{key}' ({info['similarity']}) -> '{info['taco_food_name']}'")
        else:
            print(f"  '{key}' -> SEM MATCH")

    return mapping


if __name__ == "__main__":
    main()
