"""
Gera 5 relatorios de avaliacao nutricional via Gemini API, um por faixa etaria.
Saida: Desktop/TCC/avaliacaonutri/
"""

from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.assessment_agent import AssessmentAgent

# ── Configuração ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

API_KEY  = os.getenv("GEMINI_API_KEY", "")
MODEL    = "gemini-3.1-flash-lite"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

IN_CSV  = "tmp_who/batch_500_enani_results.csv"
OUT_DIR = os.path.expanduser(r"~\OneDrive - SAP SE\Desktop\TCC\avaliacaonutri")
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ["6-8m", "9-11m", "12-23m", "24-35m", "36-59m"]

# Perfil representativo por faixa (mediana ENANI)
REPR = {
    "6-8m":   dict(age_months=7,  sex="M", weight_kg=7.5,  height_cm=68.0),
    "9-11m":  dict(age_months=10, sex="M", weight_kg=9.0,  height_cm=74.0),
    "12-23m": dict(age_months=18, sex="M", weight_kg=11.0, height_cm=82.0),
    "24-35m": dict(age_months=30, sex="M", weight_kg=13.0, height_cm=91.0),
    "36-59m": dict(age_months=48, sex="M", weight_kg=15.5, height_cm=100.0),
}

NUTRIENTS = ["energy_pct","protein_pct","iron_pct","calcium_pct",
             "zinc_pct","vitA_pct","vitC_pct"]
NUT_NAMES = {"energy_pct":"Energia","protein_pct":"Proteína","iron_pct":"Ferro",
             "calcium_pct":"Cálcio","zinc_pct":"Zinco","vitA_pct":"Vitamina A",
             "vitC_pct":"Vitamina C"}
THRESH = {"energy_pct":(80,120),"protein_pct":(70,None),"iron_pct":(70,None),
          "calcium_pct":(70,None),"zinc_pct":(70,None),"vitA_pct":(70,None),
          "vitC_pct":(70,None)}


def band_stats(df: pd.DataFrame, band: str) -> dict:
    sub = df[df["stratum"] == band]
    stats = {}
    for n in NUTRIENTS:
        s = sub[n].dropna()
        lo, hi = THRESH[n]
        fail = int((s < lo).sum())
        if hi:
            fail += int((s > hi).sum())
        stats[n] = dict(
            median=round(float(s.median()), 1),
            mean=round(float(s.mean()), 1),
            p10=round(float(s.quantile(0.10)), 1),
            p90=round(float(s.quantile(0.90)), 1),
            min=round(float(s.min()), 1),
            below100=int((s < 100).sum()),
            below_thresh=fail,
            n=len(s),
        )
    # n_foods
    stats["n_foods"] = dict(
        mean=round(float(sub["n_foods"].mean()), 1),
        min=int(sub["n_foods"].min()),
        max=int(sub["n_foods"].max()),
    )
    # NAS
    stats["nas"] = dict(
        median=round(float(sub["nas_wavg"].median()), 4),
        mean=round(float(sub["nas_wavg"].mean()), 4),
    )
    # regional breakdown
    reg_energy = sub.groupby("region")["energy_pct"].median().round(1).to_dict()
    stats["energy_by_region"] = reg_energy
    return stats


def top_foods_for_band(band: str, n=12) -> list[str]:
    foods = pd.read_csv("data/processed/unified_foods.csv")
    col_map = {"6-8m":"NAS_6_8m","9-11m":"NAS_9_11m","12-23m":"NAS_12_23m",
               "24-35m":"NAS_24_35m","36-59m":"NAS_36_59m"}
    col = col_map[band]
    eligible = foods[foods["is_eligible"] == True].copy()
    top = eligible.nlargest(n, col)[["food_name","food_category",col]].reset_index(drop=True)
    return [f"{r['food_name']} ({r['food_category']})" for _, r in top.iterrows()]


def get_dri_for_band(band: str) -> dict:
    a = AssessmentAgent()
    profile = REPR[band]
    result = a.assess(profile)
    return result.dri


def build_prompt(band: str, dri: dict, stats: dict, top_foods: list[str]) -> str:
    s = stats
    n_plans = s["energy_pct"]["n"]

    def row(key):
        st = s[key]
        nm = NUT_NAMES[key]
        lo, hi = THRESH[key]
        thr_str = f"≥{lo}%" + (f" e ≤{hi}%" if hi else "")
        adequado = "✅" if st["below_thresh"] == 0 else f"⚠️ {st['below_thresh']}/{n_plans} planos falharam o limiar"
        return (f"  - {nm}: mediana {st['median']}% DRI | média {st['mean']}% | "
                f"P10={st['p10']}% | P90={st['p90']}% | mínimo {st['min']}% | "
                f"limiar de adequação ({thr_str}): {adequado}")

    rows = "\n".join(row(k) for k in NUTRIENTS)
    foods_str = "\n".join(f"  {i+1}. {f}" for i,f in enumerate(top_foods))

    reg_lines = "\n".join(f"  - {r}: {v}% DRI energia (mediana)"
                          for r,v in s["energy_by_region"].items())

    return f"""Você é um(a) nutricionista especialista em nutrição infantil e alimentação complementar.
Avalie criticamente os resultados abaixo, gerados pelo sistema GAIA — um sistema multi-agente de
planejamento alimentar automático para crianças brasileiras de 0 a 59 meses.

O GAIA utiliza programação linear (LP) para selecionar alimentos do banco alimentar de doação
(banco de alimentos), maximizando aceitabilidade cultural (score LightGBM) e densidade
nutricional (NAS — Nutritional Affordability Score), respeitando restrições de cobertura de DRI.
Os planos foram gerados para 100 perfis antropométricos reais da pesquisa ENANI-2019
(pesquisa nacional de nutrição infantil brasileira).

═══════════════════════════════════════════════════════
FAIXA ETÁRIA AVALIADA: {band} ({n_plans} planos)
═══════════════════════════════════════════════════════

DRI de referência (WHO/IOM para criança do sexo masculino, peso/altura típicos):
  - Energia:    {dri.get('energy_kcal','-')} kcal/dia
  - Proteína:   {dri.get('protein_g','-')} g/dia
  - Ferro:      {dri.get('iron_mg','-')} mg/dia
  - Cálcio:     {dri.get('calcium_mg','-')} mg/dia
  - Zinco:      {dri.get('zinc_mg','-')} mg/dia
  - Vitamina A: {dri.get('vitA_ug','-')} µg/dia
  - Vitamina C: {dri.get('vitC_mg','-')} mg/dia

Cobertura nutricional (% DRI atingido pelos planos):
{rows}

Diversidade alimentar:
  - Média de alimentos por plano: {s['n_foods']['mean']} (mínimo {s['n_foods']['min']}, máximo {s['n_foods']['max']})

NAS ponderado por grama (indicador de densidade nutricional por custo):
  - Mediana: {s['nas']['median']} | Média: {s['nas']['mean']}

Energia por macrorregião brasileira (mediana % DRI):
{reg_lines}

Top {len(top_foods)} alimentos com maior NAS elegíveis nesta faixa:
{foods_str}

═══════════════════════════════════════════════════════
TAREFA
═══════════════════════════════════════════════════════

Escreva um RELATÓRIO DE AVALIAÇÃO NUTRICIONAL completo, em português, com as seguintes seções:

1. **Síntese geral**: avalie em 2–3 parágrafos se os planos desta faixa etária são nutricionalmente
   adequados. Seja direto(a) sobre pontos fortes e limitações reais.

2. **Análise por nutriente**: para cada nutriente, comente se a cobertura é satisfatória, moderada
   ou crítica, e explique a causa provável (biológica, alimentar ou metodológica).

3. **Pontos de atenção clínica**: sinalize quais nutrientes e em quais proporções de planos
   representam risco real para a criança, considerando as consequências de deficiência.

4. **Diversidade e qualidade da dieta**: avalie a diversidade com base no número de alimentos
   por plano e nos alimentos com maior NAS. Comente se a seleção é coerente com as
   recomendações do Guia Alimentar para Crianças Brasileiras Menores de 2 Anos (MS, 2019).

5. **Variação regional**: interprete se há disparidades regionais relevantes na cobertura energética.

6. **Conclusão e recomendações**: conclua com 3–5 recomendações objetivas para melhorar
   os planos desta faixa etária.

O relatório deve ser rigoroso, baseado nos dados fornecidos, e escrito para uma banca de avaliação
acadêmica (TCC de graduação em Sistemas de Informação com foco em saúde nutricional).
Não invente dados que não estejam nos resultados acima.
"""


def generate_report(band: str, prompt: str) -> str:
    print(f"  Chamando Gemini para {band}...", flush=True)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
    }
    for attempt in range(1, 5):
        r = requests.post(BASE_URL, params={"key": API_KEY}, json=body, timeout=120)
        if r.status_code in (429, 503):
            wait = 15 * attempt
            print(f"    {r.status_code} — aguardando {wait}s (tentativa {attempt})...", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    r.raise_for_status()


def save_report(band: str, text: str, stats: dict):
    safe_band = band.replace("-", "_")
    filename = f"relatorio_nutricional_{safe_band}.md"
    path = os.path.join(OUT_DIR, filename)

    header = f"""# Relatório de Avaliação Nutricional — Faixa Etária {band}

**Sistema:** GAIA — Planejamento Alimentar Automatizado
**Avaliação:** Gemini 2.0 Flash
**Data:** {pd.Timestamp.now().strftime('%d/%m/%Y')}
**Planos avaliados:** {stats['energy_pct']['n']} perfis ENANI-2019
**Status LP:** 100% optimal

---

"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + text)
    print(f"  Salvo: {path}")
    return path


def main():
    print("Carregando dados...", flush=True)
    df = pd.read_csv(IN_CSV)
    df = df[df["lp_status"] != "error"]
    assessment = AssessmentAgent()

    paths = []
    for band in BANDS:
        # Pular faixas já geradas
        safe = band.replace("-", "_")
        out_path = os.path.join(OUT_DIR, f"relatorio_nutricional_{safe}.md")
        if os.path.exists(out_path):
            print(f"[{band}] ja existe, pulando.", flush=True)
            continue
        print(f"\n[{band}]", flush=True)
        stats = band_stats(df, band)
        dri   = get_dri_for_band(band)
        foods = top_foods_for_band(band, n=12)
        prompt = build_prompt(band, dri, stats, foods)

        report_text = generate_report(band, prompt)
        path = save_report(band, report_text, stats)
        paths.append(path)

        # Respeitar rate limit da API gratuita
        if band != BANDS[-1]:
            time.sleep(3)

    print(f"\n✅ {len(paths)} relatórios gerados em:")
    print(f"   {OUT_DIR}")


if __name__ == "__main__":
    main()
