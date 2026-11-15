"""
Tabela de preços de referência para alimentos do banco GAIA.

Fonte: Carrefour Online (mercado.carrefour.com.br), consulta em 2026-11-08.
Valores em R$/100g do alimento tal como registrado no TACO/TBCA (porção consumida).

Para alimentos cozidos cujo preço comercial é do item cru, aplica-se fator de
rendimento para converter R$/100g cru → R$/100g cozido.
"""

from __future__ import annotations

import re
from typing import Optional

# ── tabela de preços: (padrão regex, R$/100g, produto de referência) ──────────
# Os padrões são testados em ordem; o primeiro que casar é usado.
# Mojibake tolerada: padrões usam \w ou sequências parciais que casam em ambas
# as codificações (UTF-8 correta e Latin-1 double-encoded).

_PRICE_TABLE: list[tuple[re.Pattern, float, str]] = [
    # Açaí / acai — cobre: TACO (polpa congelada), TBCA (Euterpe oleracea, média amostras)
    # .{1,2} tolera mojibake para ç e í
    (re.compile(r"\bacai\b|polpa.*congelada|congelada.*polpa|Euterpe|polpa.*amostra|amostra.*polpa", re.IGNORECASE),
     4.66, "Açaí polpa congelada 1kg = R$46,59"),
    # Alface
    (re.compile(r"alface", re.IGNORECASE),
     1.93, "Alface crespa ~150g = R$2,89"),
    # Biscoito maisena / maizena (doce — deve vir antes de polvilho)
    (re.compile(r"biscoito.*maisena|biscoito.*maizena|maisena.*biscoito", re.IGNORECASE),
     1.37, "Biscoito Maisena Piraquê 400g = R$5,49"),
    # Bolo pronto (industrializado)
    (re.compile(r"bolo.*pronto|bolo.*chocolat|bolo.*industriali", re.IGNORECASE),
     3.33, "Bolo pronto Bauducco 270g ≈ R$9,00"),
    # Cereal matinal com açúcar (milho, flocos, froot loops etc.)
    (re.compile(r"cereal.*matinal|matinal.*cereal|flocos.*a.{1,2}.{1,2}car|sucrilhos|corn.*flakes", re.IGNORECASE),
     4.33, "Kellogg's Sucrilhos 300g = R$13,00"),
    # Ervilha (fresca, em vagem ou enlatada)
    (re.compile(r"ervilha", re.IGNORECASE),
     2.94, "Ervilha lata Bonduelle 170g = R$5,00"),
    # Laranja in natura (fruta — diferente de suco)
    (re.compile(r"laranja.*natura|laranja.*pera|laranja.*lima|laranja.*bahia|laranja.*bahiana", re.IGNORECASE),
     0.50, "Laranja pera ~R$5,00/kg"),
    # Nhoque de batata
    (re.compile(r"nhoque", re.IGNORECASE),
     1.80, "Nhoque batata fresco 500g ≈ R$9,00"),
    # Sardinha em conserva
    (re.compile(r"sardinha", re.IGNORECASE),
     5.52, "Sardinha Coqueiro lata 125g = R$6,90"),
    # Tofu / queijo de soja
    (re.compile(r"tofu|queijo.*soja|soja.*queijo", re.IGNORECASE),
     3.48, "Tofu Otofu 300g = R$10,44"),
    # Paçoca de amendoim (doce de amendoim prensado)
    (re.compile(r"pa.{1,2}oca|pacoca", re.IGNORECASE),
     3.40, "Paçoca Santa Helena 200g ≈ R$6,79"),
    # Soja, extrato solúvel / leite de soja (bebida vegetal)
    (re.compile(r"soja.*extrato|extrato.*soja|leite.*soja|soja.*fluido|soja.*sol.{1,2}vel", re.IGNORECASE),
     0.82, "Leite de soja Ades 1L ≈ R$8,19"),
    # Abobrinha (todas as preparações)
    (re.compile(r"abobrinha", re.IGNORECASE),
     0.56, "Abobrinha italiana ~350g/unid = R$1,95 → R$5,57/kg"),
    # Arroz cozido (rendimento 1:2.5 cru→cozido)
    (re.compile(r"arroz.*cozido|cozido.*arroz", re.IGNORECASE),
     0.25, "Arroz Camil Tipo 1 5kg = R$30,69 (÷2.5 rendimento)"),
    # Batata inglesa cozida
    (re.compile(r"batata.*inglesa|inglesa.*batata", re.IGNORECASE),
     0.80, "Batata inglesa = R$7,98/kg"),
    # Banana
    (re.compile(r"banana", re.IGNORECASE),
     0.47, "Banana nanica = R$4,73/kg"),
    # Biscoito polvilho
    (re.compile(r"biscoito.*polvilho|polvilho.*biscoito", re.IGNORECASE),
     6.00, "Biscoito polvilho ~150g = R$8,99"),
    # Cará / cara (tubérculo — similar à batata)
    (re.compile(r"car[a\xe1\xc1]", re.IGNORECASE),
     0.80, "Cará cozido (est.) ≈ batata = R$7,98/kg"),
    # Cereais / mistura vitamina (Mucilon)
    (re.compile(r"cereais.*vitamina|vitamina.*cereal|mistura.*vitamina", re.IGNORECASE),
     6.80, "Mucilon 400g = R$27,19"),
    # Cenoura
    (re.compile(r"cenoura", re.IGNORECASE),
     0.63, "Cenoura = R$6,29/kg"),
    # Seleta de legumes enlatada (mix de vegetais em lata)
    (re.compile(r"seleta.*legume|legume.*seleta", re.IGNORECASE),
     2.17, "Seleta de Legumes Bulnez Lata 170g = R$3,69"),
    # Macarrão (qualquer tipo / molho) — deve vir antes de "manteiga"
    # pois nomes como "c/ molho branco (leite, manteiga e farinha)" contêm "manteiga"
    (re.compile(r"macarr", re.IGNORECASE),
     0.56, "Macarrão espaguete 500g = R$6,95 (÷2.5 rendimento)"),
    # Manteiga — padrão ancorado ao início: evita capturar "Couve, manteiga" ou
    # "c/ molho branco (leite, manteiga e farinha)" onde manteiga é ingrediente
    (re.compile(r"^manteiga", re.IGNORECASE),
     8.40, "Manteiga Président 200g = R$16,79"),
    # Carne bovina fraldinha
    (re.compile(r"fraldinha", re.IGNORECASE),
     5.00, "Fraldinha bovina = R$50,00/kg (est.)"),
    # Acém / músculo / carne moída bovina
    (re.compile(r"ac.{1,2}m|musculo|mo.{1,2}da.*boi|boi.*mo.{1,2}da", re.IGNORECASE),
     3.20, "Acém moído bovino ~R$32,00/kg (÷0.7 rendimento cozido)"),
    # Creme de arroz (pó / mingau)
    (re.compile(r"creme de arroz", re.IGNORECASE),
     1.60, "Creme de Arroz Yoki 500g ≈ R$8,00 (est.)"),
    # Feijão carioca cozido (rendimento 1:2.5 cru→cozido)
    # .{1,2} tolera mojibake: ã pode ser U+00E3 (1 char) ou U+00C3+U+00A3 (2 chars)
    (re.compile(r"feij.{1,2}o.*carioca|carioca.*feij", re.IGNORECASE),
     0.41, "Feijão carioca 1kg = R$10,19 (÷2.5 rendimento)"),
    # Feijão genérico (outras variedades — carioca é o mais barato; preto ~R$13/kg)
    (re.compile(r"feij.{1,2}o", re.IGNORECASE),
     0.52, "Feijão (genérico) ~R$13,00/kg (÷2.5 rendimento)"),
    # Fígado de frango
    (re.compile(r"f.{1,2}gado.*frango|frango.*f.{1,2}gado", re.IGNORECASE),
     1.20, "Fígado de frango 500g ≈ R$5,99"),
    # Fígado bovino
    (re.compile(r"f.{1,2}gado|figado", re.IGNORECASE),
     2.20, "Fígado bovino ~R$22,00/kg (est.)"),
    # Frango caipira
    (re.compile(r"frango.*caipira|caipira", re.IGNORECASE),
     2.45, "Frango passarinho IQF 1kg = R$15,89 (÷0.65 rendimento)"),
    # Frango sobrecoxa
    (re.compile(r"frango.*sobrecoxa|sobrecoxa", re.IGNORECASE),
     2.14, "Sobrecoxa = R$13,90/kg (÷0.65 rendimento)"),
    # Iogurte natural (qualquer variante)
    (re.compile(r"iogurte", re.IGNORECASE),
     2.41, "Iogurte natural Nestlé 170g = R$4,09"),
    # Laranja suco / suco de laranja
    (re.compile(r"laranja.*suco|suco.*laranja", re.IGNORECASE),
     1.00, "Laranja pêra 3kg = R$14,99 (÷2 rendimento suco)"),
    # Leite pó desnatado
    (re.compile(r"leite.*desnatado.*p.?|desnatado.*p.?.*leite", re.IGNORECASE),
     4.72, "Leite pó desnatado ~400g ≈ R$18,89"),
    # Leite pó integral
    (re.compile(r"leite.*integral.*p.?|leite.*p.?.*integral", re.IGNORECASE),
     4.50, "Leite pó integral Piracanjuba 400g = R$17,99"),
    # Linhaça semente
    # .{1,2} tolera mojibake para ç
    (re.compile(r"linha.{1,2}a|linhaca", re.IGNORECASE),
     5.88, "Linhaça semente 250g ≈ R$14,69"),
    # Mamão
    # .{1,2} tolera mojibake para ã
    (re.compile(r"mam.{1,2}o|mamao", re.IGNORECASE),
     1.29, "Mamão papaya ~500g = R$6,44 (R$12,89/kg)"),
    # Maracujá suco
    (re.compile(r"maracuj", re.IGNORECASE),
     1.50, "Maracujá (est.) ≈ R$15,00/kg suco concentrado"),
    # Melancia
    (re.compile(r"melancia", re.IGNORECASE),
     0.40, "Melancia ~8kg = R$31,92 (R$3,99/kg)"),
    # Mexerica / tangerina
    (re.compile(r"mexerica|tangerina", re.IGNORECASE),
     0.80, "Mexerica = R$3,49/kg (est.)"),
    # Morango
    (re.compile(r"morango", re.IGNORECASE),
     4.66, "Morango ~300g = R$13,99"),
    # Ovo (clara ou inteiro, cozido)
    (re.compile(r"\bovo\b|\bovos\b", re.IGNORECASE),
     2.80, "Ovos dúzia = R$16,79 (~50g/ovo)"),
    # Pão (qualquer tipo)
    # .{1,2} tolera mojibake para ã
    (re.compile(r"p.{1,2}o\b|p[aã]o\b", re.IGNORECASE),
     2.30, "Pão francês = R$1,15/50g"),
    # Pescada
    (re.compile(r"pescada", re.IGNORECASE),
     3.06, "Pescada filé = R$30,59/kg"),
    # Polenta / fubá
    (re.compile(r"polenta|fub.{1,2}", re.IGNORECASE),
     0.99, "Fubá 1kg = R$9,89"),
    # Queijo mozarela / mussarela (todas as grafias)
    (re.compile(r"queijo.*mo[sz]arel|mo[sz]arel.*queijo", re.IGNORECASE),
     3.89, "Queijo Mussarela Carrefour R$38,90/kg"),
    # Abóbora / moranga
    # .{1,2} tolera mojibake para ó
    (re.compile(r"ab.{1,2}bora|moranga|caboti", re.IGNORECASE),
     0.63, "Abóbora cabotiá = R$6,29/kg"),
    # Berinjela cozida
    (re.compile(r"berinjela", re.IGNORECASE),
     0.50, "Berinjela = R$4,99/kg"),
    # Castanha-de-caju (qualquer preparo)
    (re.compile(r"castanha.*caju|caju.*castanha", re.IGNORECASE),
     8.00, "Castanha-de-caju torrada 150g ≈ R$12,00"),
    # Couve-flor cozida
    (re.compile(r"couve.?flor|couve flor", re.IGNORECASE),
     1.00, "Couve-flor ~500g ≈ R$5,00 (R$1,00/100g cozida)"),
    # Costela bovina assada (30% perda no assado: R$30/kg raw → R$4,30/100g assada)
    (re.compile(r"costela", re.IGNORECASE),
     4.30, "Costela bovina ~R$30/kg raw (÷0.70 rendimento assado)"),
    # Goiabada / goiava, doce em pasta
    (re.compile(r"goiab.*pasta|goiab.*doce|goiabada|pasta.*goiab", re.IGNORECASE),
     1.55, "Goiabada Predilecta 800g ≈ R$12,39 (R$1,55/100g)"),
    # Pêssego enlatado em calda
    (re.compile(r"p.{1,2}ssego.*enlat|p.{1,2}ssego.*calda|enlat.*p.{1,2}ssego|calda.*p.{1,2}ssego", re.IGNORECASE),
     2.50, "Pêssego enlatado em calda Predilecta 400g = R$10,00 (R$2,50/100g)"),
    # Leite fermentado (kefir, cultured milk — tipo Yakult/Ninho fermentado)
    (re.compile(r"leite.*fermentado|fermentado.*leite", re.IGNORECASE),
     1.00, "Leite fermentado Nestlé 1L ≈ R$10,00 (R$1,00/100g)"),
    # Canjica com leite integral (milho branco cozido + leite)
    (re.compile(r"canjica|canjic.{1,2}", re.IGNORECASE),
     1.40, "Canjica (milho branco + leite integral) — R$1,40/100g porção pronta"),
    # Leite de vaca (genérico — último recurso para laticínios)
    (re.compile(r"leite.*vaca|vaca.*leite", re.IGNORECASE),
     0.40, "Leite vaca (genérico) ≈ R$4/L"),
]

# Preço padrão para alimentos sem correspondência na tabela
_DEFAULT_PRICE_PER_100G = 2.00  # R$20/kg — valor médio conservador


def get_price(food_name: str) -> float:
    """
    Retorna R$/100g para um alimento pelo nome.

    Usa correspondência por regex em ordem de especificidade.
    Retorna _DEFAULT_PRICE_PER_100G se nenhum padrão casar.
    """
    for pattern, price, _ in _PRICE_TABLE:
        if pattern.search(food_name):
            return price
    return _DEFAULT_PRICE_PER_100G


def get_price_label(food_name: str) -> str:
    """Retorna produto de referência usado para o preço."""
    for pattern, _, label in _PRICE_TABLE:
        if pattern.search(food_name):
            return label
    return f"padrão genérico = R${_DEFAULT_PRICE_PER_100G:.2f}/100g"


def build_price_array(food_names: list[str]) -> list[float]:
    """Retorna lista de R$/100g para cada alimento na lista."""
    return [get_price(name) for name in food_names]
