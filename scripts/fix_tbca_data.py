"""
Script de correção dos 102 itens TBCA em unified_foods.csv.

Problema: todos os itens TBCA têm food_category="TBCA" e nutrientes NaN.
Solução: mapear cada item TBCA para o equivalente TACO mais próximo e
         copiar food_category + nutrientes + NAS. Energia TBCA é preservada.

Executar: python scripts/fix_tbca_data.py
"""

import os
import shutil
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(BASE, "data", "processed", "unified_foods.csv")
BAK  = CSV.replace(".csv", "_backup_pre_tbca_fix.csv")

# Colunas copiadas do equivalente TACO
COPY_COLS = [
    "food_category",
    "protein_g", "fat_g", "carbs_g",
    "calcium_mg", "iron_mg", "zinc_mg", "vitA_ug", "vitC_mg",
    "NAS_6_8m", "NAS_9_11m", "NAS_12_23m", "NAS_24_35m", "NAS_36_59m",
]

# Mapeamento completo: TBCA food_id → TACO food_id equivalente (None = sem par)
TBCA_TO_TACO: dict[str, str | None] = {
    # ── FRUTAS ───────────────────────────────────────────────────────────────
    "TBCA_C0002C": "TACO_164",   # Abacaxi → Abacaxi, cru
    "TBCA_C0007C": "TACO_179",   # Banana nanica
    "TBCA_C0020C": "TACO_208",   # Laranja Pera → Laranja, baía
    "TBCA_C0045C": "TACO_225",   # Mamão Formosa
    "TBCA_C0027C": "TACO_235",   # Melancia
    "TBCA_C0035C": "TACO_257",   # Uva Rubi
    "TBCA_C0029C": "TACO_239",   # Morango
    "TBCA_C0048C": "TACO_228",   # Manga Tommy Atkins → Manga Haden
    "TBCA_C0003C": "TACO_168",   # Açaí polpa
    "TBCA_C0066C": "TACO_238",   # Mexerica Rio
    "TBCA_C0064C": "TACO_221",   # Maçã Fuji → Maçã Argentina

    # ── CEREAIS ──────────────────────────────────────────────────────────────
    "TBCA_C0209A": "TACO_3",     # Arroz polido c/ tempero → arroz tipo 1 cozido
    "TBCA_C0211A": "TACO_1",     # Arroz integral c/ tempero
    "TBCA_C0022A": "TACO_7",     # Aveia farinha crua
    "TBCA_C0024A": "TACO_7",     # Aveia flocos finos
    "TBCA_C0835A": "TACO_40",    # Macarrão integral cozido
    "TBCA_C0214A": "TACO_40",    # Macarrão c/ alho e óleo
    "TBCA_C0218A": "TACO_40",    # Macarrão c/ molho de tomate
    "TBCA_C0220A": "TACO_40",    # Macarrão c/ molho branco
    "TBCA_C0235A": "TACO_40",    # Macarrão integral c/ molho de tomate
    "TBCA_C0237A": "TACO_40",    # Macarrão integral c/ molho branco
    "TBCA_C0238A": "TACO_40",    # Macarrão c/ molho bolonhesa
    "TBCA_C0885A": "TACO_40",    # Macarrão carbonara
    "TBCA_C0909A": "TACO_40",    # Macarrão c/ molho pesto
    "TBCA_C0911A": "TACO_40",    # Macarrão integral c/ molho pesto
    "TBCA_C0962A": "TACO_40",    # Macarrão integral c/ atum e azeite

    # ── HORTALIÇAS ───────────────────────────────────────────────────────────
    "TBCA_C0063B": "TACO_77",    # Alface americana crua
    "TBCA_C0060B": "TACO_78",    # Alface crespa crua
    "TBCA_C0009B": "TACO_79",    # Alface crua → alface lisa
    "TBCA_C0913B": "TACO_109",   # Cenoura cozida
    "TBCA_C0020B": "TACO_110",   # Cenoura crua
    "TBCA_C0044B": "TACO_97",    # Beterraba cozida
    "TBCA_C0116B": "TACO_118",   # Couve-flor cozida
    "TBCA_C0144B": "TACO_116",   # Couve cozida → couve manteiga refogada
    "TBCA_C0151B": "TACO_68",    # Abóbora moranga refogada
    "TBCA_C0355B": "TACO_68",    # Abóbora moranga assada
    "TBCA_C0226B": "TACO_68",    # Abóbora moranga cozida
    "TBCA_C0228B": "TACO_68",    # Abóbora pescoço cozida (proxy: moranga)
    "TBCA_C0232B": "TACO_70",    # Abobrinha cozida
    "TBCA_C0153B": "TACO_70",    # Abobrinha refogada

    # ── LEGUMINOSAS ──────────────────────────────────────────────────────────
    "TBCA_C0091T": "TACO_561",   # Feijão carioca c/ tempero
    "TBCA_C0108T": "TACO_567",   # Feijão preto c/ tempero

    # ── TUBÉRCULOS ───────────────────────────────────────────────────────────
    "TBCA_C0303B": "TACO_88",    # Batata doce assada
    "TBCA_C0882B": "TACO_88",    # Batata doce assada c/ azeite
    "TBCA_C0299B": "TACO_88",    # Batata doce cozida
    "TBCA_C0305B": "TACO_88",    # Batata doce frita
    "TBCA_C0118B": "TACO_91",    # Batata inglesa frita
    "TBCA_C0141B": "TACO_91",    # Batata inglesa assada
    "TBCA_C0117B": "TACO_91",    # Batata inglesa cozida

    # ── LEITE E DERIVADOS ────────────────────────────────────────────────────
    "TBCA_C0020G": "TACO_448",   # Iogurte natural
    "TBCA_C0022G": "TACO_449",   # Iogurte natural desnatado

    # ── CARNES BOVINAS ───────────────────────────────────────────────────────
    "TBCA_C0354F": "TACO_326",   # Acém moída cozida
    "TBCA_C0608F": "TACO_377",   # Alcatra frita (proxy: patinho grelhado)
    "TBCA_C0322F": "TACO_358",   # Capa contra-filé grelhada → filé mignon
    "TBCA_C0660F": "TACO_326",   # Charque dessalgada frita (proxy: acém)
    "TBCA_C0661F": "TACO_326",   # Charque dessalgada refogada
    "TBCA_C0334F": "TACO_358",   # Contra-filé de costela grelhada
    "TBCA_C0296F": "TACO_358",   # Contra-filé grelhado/frito c/ óleo
    "TBCA_C0302F": "TACO_358",   # Contra-filé grelhado c/ manteiga
    "TBCA_C0870F": "TACO_326",   # Coração bovino cozido (proxy: acém)
    "TBCA_C0869F": "TACO_349",   # Costela cozida (proxy: coxão duro)
    "TBCA_C0369F": "TACO_349",   # Coxão mole cozido
    "TBCA_C0924F": "TACO_349",   # Coxão mole frito
    "TBCA_C0597F": "TACO_358",   # Filé mignon frito
    "TBCA_C0374F": "TACO_358",   # Filé mignon grelhado c/ óleo
    "TBCA_C0380F": "TACO_358",   # Filé mignon grelhado c/ manteiga
    "TBCA_C0595F": "TACO_358",   # Filé mignon grelhado/assado s/ óleo
    "TBCA_C0804F": "TACO_377",   # Fraldinha frita (proxy: patinho)
    "TBCA_C0431F": "TACO_374",   # Paleta cozida refogada
    "TBCA_C0665F": "TACO_374",   # Paleta frita
    "TBCA_C0634F": "TACO_377",   # Patinho frito
    "TBCA_C0441F": "TACO_377",   # Patinho grelhado c/ óleo
    "TBCA_C0447F": "TACO_377",   # Patinho grelhado c/ manteiga
    "TBCA_C0823F": "TACO_381",   # Picanha grelhada/assada
    "TBCA_C0466F": "TACO_381",   # Picanha s/ gordura grelhada c/ manteiga

    # ── AVES ─────────────────────────────────────────────────────────────────
    "TBCA_C0742F": "TACO_396",   # Frango asa assada
    "TBCA_C0740F": "TACO_396",   # Frango asa cozida
    "TBCA_C0744F": "TACO_396",   # Frango asa frita
    "TBCA_C0773F": "TACO_395",   # Frango coração frito
    "TBCA_C0771F": "TACO_395",   # Frango coração grelhado
    "TBCA_C0842F": "TACO_396",   # Frango coxa c/ pele assada
    "TBCA_C0843F": "TACO_396",   # Frango coxa s/ pele cozida (proxy: coxa assada)

    # ── PORCO ────────────────────────────────────────────────────────────────
    "TBCA_C0685F": "TACO_428",   # Porco bisteca frita
    "TBCA_C0689F": "TACO_432",   # Porco costela assada (proxy: lombo assado)
    "TBCA_C0692F": "TACO_428",   # Porco costela frita (proxy: bisteca)
    "TBCA_C0694F": "TACO_428",   # Porco costela refogada
    "TBCA_C0697F": "TACO_432",   # Porco lombo grelhado/assado

    # ── PESCADOS ─────────────────────────────────────────────────────────────
    "TBCA_C0091E": "TACO_290",   # Corimba cozido
    "TBCA_C0089E": "TACO_294",   # Corvina cozida
    "TBCA_C0093E": "TACO_295",   # Dourado filé cozido (proxy: dourada fresca)
    "TBCA_C0103E": "TACO_290",   # Peixe água doce filé cozido média
    "TBCA_C0113E": "TACO_290",   # Peixe água doce filé ensopado
    "TBCA_C0104E": "TACO_311",   # Peixe água doce filé grelhado (proxy: pintado)
    "TBCA_C0097E": "TACO_311",   # Pintado cozido
    "TBCA_C0099E": "TACO_294",   # Tilápia filé cozido (proxy: corvina)
    "TBCA_C0130E": "TACO_277",   # Atum assado/grelhado → atum conserva
    "TBCA_C0365E": "TACO_280",   # Bacalhau cozido dessalgado
    "TBCA_C0044E": "TACO_280",   # Bacalhau refogado
    "TBCA_C0064E": "TACO_303",   # Merluza filé frito
    "TBCA_C0228E": "TACO_303",   # Pescada branca (proxy: merluza)
    "TBCA_C0202E": "TACO_317",   # Salmão frito

    # ── SUPLEMENTO ───────────────────────────────────────────────────────────
    "TBCA_C0097N": None,         # Suplemento proteico em pó (sem equivalente)
}


def main():
    print(f"Carregando {CSV} ...")
    df = pd.read_csv(CSV)

    # backup
    shutil.copy2(CSV, BAK)
    print(f"Backup salvo em: {BAK}")

    tbca_ids = df[df["source"] == "TBCA"]["food_id"].tolist()
    print(f"\nTotal itens TBCA: {len(tbca_ids)}")
    print(f"Mapeamentos definidos: {len(TBCA_TO_TACO)}")

    not_in_map = [fid for fid in tbca_ids if fid not in TBCA_TO_TACO]
    if not_in_map:
        print(f"AVISO: sem mapeamento para {len(not_in_map)} itens: {not_in_map}")

    # pré-indexa TACO por food_id para lookup rápido
    taco_lookup = df[df["source"] == "TACO"].set_index("food_id")

    updated = 0
    skipped = 0

    for tbca_id, taco_id in TBCA_TO_TACO.items():
        idx = df.index[df["food_id"] == tbca_id]
        if len(idx) == 0:
            print(f"  AVISO: {tbca_id} não encontrado no CSV")
            continue
        row_idx = idx[0]

        # suplemento proteico: marcar inelegível e categoria genérica
        if taco_id is None:
            df.at[row_idx, "food_category"] = "Outros"
            df.at[row_idx, "is_eligible"]   = False
            print(f"  {tbca_id}: marcado inelegível (suplemento)")
            updated += 1
            continue

        if taco_id not in taco_lookup.index:
            print(f"  AVISO: {taco_id} não encontrado no CSV (proxy para {tbca_id})")
            skipped += 1
            continue

        taco_row = taco_lookup.loc[taco_id]
        energia_original = df.at[row_idx, "energy_kcal"]

        for col in COPY_COLS:
            if col in df.columns and col in taco_row.index:
                df.at[row_idx, col] = taco_row[col]

        # preservar energia TBCA (preparações diferentes dos TACO afetam kcal)
        df.at[row_idx, "energy_kcal"] = energia_original
        updated += 1

    print(f"\nItens atualizados: {updated}")
    print(f"Itens ignorados (proxy não encontrado): {skipped}")

    # verificação final
    tbca_after = df[df["source"] == "TBCA"]
    still_tbca_cat = (tbca_after["food_category"] == "TBCA").sum()
    nan_protein    = tbca_after["protein_g"].isna().sum()
    print(f"\nVerificação pós-correção:")
    print(f"  Ainda com food_category='TBCA': {still_tbca_cat}")
    print(f"  Ainda com protein_g NaN:        {nan_protein}")
    print(f"\nDistribuição de categorias TBCA após fix:")
    print(tbca_after["food_category"].value_counts().to_string())

    df.to_csv(CSV, index=False)
    print(f"\nCSV salvo: {CSV}")


if __name__ == "__main__":
    main()
