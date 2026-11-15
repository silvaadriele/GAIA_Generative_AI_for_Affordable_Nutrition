"""
Adiciona coluna nova_group (1–4) ao unified_foods.csv.

Classificação NOVA seguindo:
  Monteiro et al. (2019). "Ultra-processed foods: what they are and how to
  identify them." Public Health Nutrition, 22(5), 936–941.
  doi:10.1017/S1368980018003762

Grupos:
  1 — Alimentos não processados ou minimamente processados
        (carnes frescas, frutas, legumes, cereais em grão, ovos, leite)
  2 — Ingredientes culinários processados
        (óleos vegetais, manteiga, farinha, açúcar, sal)
  3 — Alimentos processados
        (queijos, iogurte natural, pão, linguiça, presunto, leite condensado)
  4 — Alimentos ultraprocessados
        (biscoitos recheados, bolo pronto, cereal matinal açucarado, mortadela,
         achocolatado, iogurte sabor, petit suisse, macarrão instantâneo, etc.)

Metodologia:
  - Todos os itens recebem nova_group=1 por padrão.
  - Itens claramente do Grupo 2, 3 ou 4 são sobrepostos por mapeamento explícito.
  - Itens ambíguos são classificados conservadoramente no grupo menor (menos restritivo).

Executar: python scripts/add_nova_groups.py
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(BASE, "data", "processed", "unified_foods.csv")


# ── Grupo 2: Ingredientes culinários processados ──────────────────────────────
# Óleos, gorduras, farinhas puras, açúcar refinado.
# Presença no banco: óleos vegetais, manteiga, margarina sem óleo hidrogenado.
NOVA_2: set[str] = {
    "TACO_259",  # Azeite, de dendê
    "TACO_260",  # Azeite, de oliva, extra virgem
    "TACO_261",  # Manteiga, com sal
    "TACO_262",  # Manteiga, sem sal
    "TACO_265",  # Margarina, com óleo interesterificado, com sal
    "TACO_266",  # Margarina, com óleo interesterificado, sem sal
    "TACO_267",  # Óleo, de babaçu
    "TACO_268",  # Óleo, de canola
    "TACO_269",  # Óleo, de girassol
    "TACO_270",  # Óleo, de milho
    "TACO_271",  # Óleo, de pequi
    "TACO_272",  # Óleo, de soja
    "TACO_31",   # Farinha, de arroz, enriquecida
    "TACO_32",   # Farinha, de centeio, integral
    "TACO_33",   # Farinha, de milho, amarela
    "TACO_35",   # Farinha, de trigo
    "TACO_42",   # Milho, amido, cru
    "TACO_43",   # Milho, fubá, cru
}


# ── Grupo 3: Alimentos processados ───────────────────────────────────────────
# Fabricados com adição de sal, açúcar, óleo ou vinagre a alimentos integrais.
# Processo de cura, fermentação, defumação ou conserva reconhecíveis.
NOVA_3: set[str] = {
    # Pães (contêm aditivos de panificação, mas processo reconhecível)
    "TACO_48",   # Pão, aveia, forma
    "TACO_49",   # Pão, de soja
    "TACO_50",   # Pão, glúten, forma
    "TACO_51",   # Pão, milho, forma
    "TACO_52",   # Pão, trigo, forma, integral
    "TACO_53",   # Pão, trigo, francês
    "TACO_54",   # Pão, trigo, sovado
    "TACO_63",   # Torrada, pão francês
    "TACO_34",   # Farinha, de rosca
    # Laticínios processados
    "TACO_447",  # Creme de Leite (apenas creme centrifugado — sem aditivos declarados no TACO)
    "TACO_448",  # Iogurte, natural
    "TACO_449",  # Iogurte, natural, desnatado
    "TACO_453",  # Leite, condensado (leite + açúcar — processo tradicional)
    "TACO_454",  # Leite, de cabra
    "TACO_456",  # Leite, de vaca, desnatado, pó
    "TACO_459",  # Leite, de vaca, integral, pó
    "TACO_461",  # Queijo, minas, frescal
    "TACO_462",  # Queijo, minas, meia cura
    "TACO_463",  # Queijo, mozarela
    "TACO_464",  # Queijo, parmesão
    "TACO_465",  # Queijo, pasteurizado
    "TACO_467",  # Queijo, prato
    "TACO_468",  # Queijo, requeijão, cremoso
    "TACO_469",  # Queijo, ricota
    # Carnes processadas tradicionais (cura/defumação sem aditivos industriais complexos)
    "TACO_338",  # Carne, bovina, charque, cozido
    "TACO_339",  # Carne, bovina, charque, cru
    "TACO_384",  # Carne, bovina, seca, cozida
    "TACO_385",  # Carne, bovina, seca, crua
    "TACO_418",  # Linguiça, frango, crua
    "TACO_419",  # Linguiça, frango, frita
    "TACO_420",  # Linguiça, frango, grelhada
    "TACO_421",  # Linguiça, porco, crua
    "TACO_422",  # Linguiça, porco, frita
    "TACO_423",  # Linguiça, porco, grelhada
    "TACO_438",  # Presunto, com capa de gordura
    "TACO_439",  # Presunto, sem capa de gordura
    "TACO_443",  # Salame
    "TACO_444",  # Toucinho, cru
    "TACO_445",  # Toucinho, frito
    # Conservas simples
    "TACO_45",   # Milho, verde, enlatado, drenado
    # Mingaus em pó tradicionais (processados mas sem formulação industrial complexa)
    "TACO_27",   # Creme de arroz, pó
    "TACO_28",   # Creme de milho, pó
    "TACO_46",   # Mingau tradicional, pó
    # Pastéis e salgados caseiros (farinha + recheio — sem aditivos declarados no TACO)
    "TACO_55",   # Pastel, de carne, cru
    "TACO_56",   # Pastel, de carne, frito
    "TACO_57",   # Pastel, de queijo, cru
    "TACO_58",   # Pastel, de queijo, frito
    "TACO_59",   # Pastel, massa, crua
    "TACO_60",   # Pastel, massa, frita
    "TACO_61",   # Pipoca, com óleo de soja, sem sal
    # Hambúrguer (carne moída + sal — TACO descreve produto simples, não industrial)
    "TACO_415",  # Hambúrguer, bovino, cru
    "TACO_416",  # Hambúrguer, bovino, frito
    "TACO_417",  # Hambúrguer, bovino, grelhado
    # Frango à milanesa (carne + farinha de rosca — processado, não ultraprocessado)
    "TACO_401",  # Frango, filé, à milanesa
    "TACO_340",  # Carne, bovina, contra-filé, à milanesa
    # Almôndegas (carne + farinha + ovo)
    "TACO_330",  # Carne, bovina, almôndegas, cruas
    "TACO_331",  # Carne, bovina, almôndegas, fritas
    # Cereal matinal sem açúcar (limítrofe — classificado conservadoramente como 3)
    "TACO_25",   # Cereais, milho, flocos, com sal
    "TACO_22",   # Cereais, milho, flocos, sem sal
    # Cereais mistos (aveia + cevada + trigo — ingredientes reconhecíveis)
    "TACO_24",   # Cereais, mistura para vitamina, trigo, cevada e aveia
    # Margarina interesterificada (processo industrial mas sem trans — limítrofe 3/4)
    # → já marcada no Grupo 2 acima (TACO_265, TACO_266)
}


# ── Grupo 4: Alimentos ultraprocessados ──────────────────────────────────────
# Formulações industriais com 5+ ingredientes, incluindo aditivos (emulsificantes,
# corantes, aromatizantes artificiais, conservantes não usados em cozinha doméstica).
# Fonte: Monteiro et al. (2019); Guia Alimentar MS (2014).
NOVA_4: set[str] = {
    # Biscoitos industrializados
    "TACO_8",    # Biscoito, doce, maisena
    "TACO_9",    # Biscoito, doce, recheado com chocolate
    "TACO_10",   # Biscoito, doce, recheado com morango
    "TACO_11",   # Biscoito, doce, wafer, recheado de chocolate
    "TACO_12",   # Biscoito, doce, wafer, recheado de morango
    "TACO_13",   # Biscoito, salgado, cream cracker
    # Bolos industrializados
    "TACO_14",   # Bolo, mistura para
    "TACO_15",   # Bolo, pronto, aipim
    "TACO_16",   # Bolo, pronto, chocolate
    "TACO_17",   # Bolo, pronto, coco
    "TACO_18",   # Bolo, pronto, milho
    # Cereais matinais industriais
    "TACO_23",   # Cereais, mingau, milho, infantil (produto comercial com aditivos)
    "TACO_26",   # Cereal matinal, milho, açúcar
    # Farinha láctea (produto infantil ultraprocessado — Nestlé Farinha Láctea)
    "TACO_36",   # Farinha, láctea, de cereais
    # Macarrão instantâneo
    "TACO_39",   # Macarrão, instantâneo
    # Caldo em tablete (MSG, sal, aromatizantes artificiais)
    "TACO_324",  # Caldo de carne, tablete
    "TACO_325",  # Caldo de galinha, tablete
    # Embutidos ultraprocessados (têm amido, antioxidantes, corante, estabilizantes)
    "TACO_323",  # Apresuntado (≠ presunto — tem amido de trigo, proteína de soja)
    "TACO_424",  # Mortadela
    # Salgados industriais fritos
    "TACO_386",  # Coxinha de frango, frita
    "TACO_387",  # Croquete, de carne, cru
    "TACO_388",  # Croquete, de carne, frito
    "TACO_389",  # Empada de frango, pré-cozida, assada
    "TACO_390",  # Empada, de frango, pré-cozida
    # Laticínios ultraprocessados
    "TACO_446",  # Bebida láctea, pêssego (aromatizante artificial, espessantes)
    "TACO_450",  # Iogurte, sabor abacaxi
    "TACO_451",  # Iogurte, sabor morango
    "TACO_452",  # Iogurte, sabor pêssego
    "TACO_455",  # Leite, de vaca, achocolatado
    "TACO_460",  # Leite, fermentado (Yakult/Actimel — adoçantes, aromatizantes)
    "TACO_466",  # Queijo, petit suisse, morango
    # Margarina com óleo hidrogenado (ácidos graxos trans industriais)
    "TACO_263",  # Margarina, com óleo hidrogenado, com sal
    "TACO_264",  # Margarina, com óleo hidrogenado, sem sal
}


def main():
    print(f"Carregando {CSV} ...")
    df = pd.read_csv(CSV)
    print(f"Total: {len(df)} itens")

    # Padrão: Grupo 1 (não processado / minimamente processado)
    df["nova_group"] = 1

    applied_2 = applied_3 = applied_4 = 0
    for food_id, group in (
        [(fid, 2) for fid in NOVA_2] +
        [(fid, 3) for fid in NOVA_3] +
        [(fid, 4) for fid in NOVA_4]
    ):
        mask = df["food_id"] == food_id
        if mask.any():
            df.loc[mask, "nova_group"] = group
            if group == 2: applied_2 += 1
            elif group == 3: applied_3 += 1
            elif group == 4: applied_4 += 1
        else:
            print(f"  AVISO: {food_id} não encontrado no CSV")

    print(f"\nGrupo 1 (padrão): {(df['nova_group'] == 1).sum()} itens")
    print(f"Grupo 2 aplicado: {applied_2} itens")
    print(f"Grupo 3 aplicado: {applied_3} itens")
    print(f"Grupo 4 aplicado: {applied_4} itens")

    # Itens is_eligible=False do suplemento proteico mantidos como estão
    g4_eligible = df[(df["nova_group"] == 4) & (df.get("is_eligible", pd.Series(True, index=df.index)) != False)]
    print(f"\nItens Grupo 4 elegíveis (serão excluídos dos planos): {len(g4_eligible)}")
    for _, row in g4_eligible.sort_values("food_name").iterrows():
        print(f"  [{row['food_id']}] {row['food_name']}")

    df.to_csv(CSV, index=False)
    print(f"\nCSV salvo: {CSV} (coluna nova_group adicionada)")


if __name__ == "__main__":
    main()
