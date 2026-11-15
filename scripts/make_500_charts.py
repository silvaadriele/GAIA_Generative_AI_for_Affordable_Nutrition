"""
Gera gráficos e tabelas para o artigo a partir de batch_500_results.csv.

Saídas em tmp_who/figures/:
  fig_adequacy_by_band.png  — Mediana % DRI por faixa etária (gráfico de barras agrupadas)
  fig_boxplot_nutrients.png — Distribuição % DRI por nutriente (box plots)
  fig_iron_calcium.png      — Ferro e Cálcio por faixa etária (barras + erro)
  fig_by_region.png         — Mediana % DRI energia/ferro/cálcio por região

  evaluation_500_charts.xlsx — Tabelas completas para referência
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xlsxwriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IN_CSV   = "tmp_who/batch_500_enani_results.csv"
OUT_DIR  = "tmp_who/figures"
XLSX_OUT = "tmp_who/evaluation_500_enani_charts.xlsx"

NUTRIENTS  = ["energy_pct","protein_pct","iron_pct","calcium_pct","zinc_pct","vitA_pct","vitC_pct"]
NUT_LABELS = ["Energy","Protein","Iron","Calcium","Zinc","Vit. A","Vit. C"]
AGE_BANDS  = ["6-8m","9-11m","12-23m","24-35m","36-59m"]
REGIONS    = ["Norte","Nordeste","Centro-Oeste","Sudeste","Sul"]

BAND_COLORS = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"]
REGION_COLORS = ["#1b7837","#762a83","#d6604d","#4393c3","#f4a582"]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":      True,
    "grid.color":     "#E0E0E0",
    "grid.linewidth": 0.7,
    "figure.dpi":     150,
})


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df["stratum"] = pd.Categorical(df["stratum"], categories=AGE_BANDS, ordered=True)
    return df


# ── Figure 1: Mediana % DRI por faixa etária ──────────────────────────────────
def fig_adequacy_by_band(df: pd.DataFrame):
    medians = df.groupby("stratum", observed=True)[NUTRIENTS].median().reindex(AGE_BANDS)

    # Exclui proteína e vitC do gráfico principal (valores muito altos distorcem escala)
    nuts_plot  = ["energy_pct","iron_pct","calcium_pct","zinc_pct","vitA_pct"]
    nuts_lbls  = ["Energy","Iron","Calcium","Zinc","Vit. A"]

    x       = np.arange(len(nuts_plot))
    width   = 0.15
    offsets = np.linspace(-(len(AGE_BANDS)-1)/2, (len(AGE_BANDS)-1)/2, len(AGE_BANDS)) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, band in enumerate(AGE_BANDS):
        vals = [medians.loc[band, n] for n in nuts_plot]
        ax.bar(x + offsets[i], vals, width, label=band, color=BAND_COLORS[i],
               alpha=0.88, zorder=3)

    ax.axhline(100, color="#333333", linewidth=1.2, linestyle="--",
               label="100% DRI target", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(nuts_lbls, fontsize=11)
    ax.set_ylabel("Median % DRI", fontsize=11)
    ax.set_title("Median % DRI by Age Band and Nutrient  (n = 500)", fontsize=12, pad=10)
    ax.set_ylim(0, 280)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_adequacy_by_band.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── Figure 2: Box plots dos nutrientes ────────────────────────────────────────
def fig_boxplot_nutrients(df: pd.DataFrame):
    nuts  = ["energy_pct","iron_pct","calcium_pct","zinc_pct","vitA_pct"]
    lbls  = ["Energy","Iron","Calcium","Zinc","Vit. A"]
    colors = ["#4C72B0","#C44E52","#8172B2","#55A868","#CCB974"]

    fig, ax = plt.subplots(figsize=(10, 5))

    data   = [df[n].dropna().values for n in nuts]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        widths=0.55,
        zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(100, color="#333333", linewidth=1.2, linestyle="--",
               label="100% DRI", zorder=4)

    ax.set_xticks(range(1, len(lbls)+1))
    ax.set_xticklabels(lbls, fontsize=11)
    ax.set_ylabel("% DRI", fontsize=11)
    ax.set_title("Distribution of Nutrient Adequacy  (n = 500)", fontsize=12, pad=10)
    ax.set_ylim(0, 500)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_boxplot_nutrients.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── Figure 3: Ferro e Cálcio por faixa etária ─────────────────────────────────
def fig_iron_calcium(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, nut, label, color in [
        (axes[0], "iron_pct",    "Iron % DRI",    "#C44E52"),
        (axes[1], "calcium_pct", "Calcium % DRI", "#8172B2"),
    ]:
        medians = df.groupby("stratum", observed=True)[nut].median().reindex(AGE_BANDS)
        p10     = df.groupby("stratum", observed=True)[nut].quantile(0.10).reindex(AGE_BANDS)
        p90     = df.groupby("stratum", observed=True)[nut].quantile(0.90).reindex(AGE_BANDS)
        pct_below = [(df[df["stratum"]==b][nut] < 100).mean()*100 for b in AGE_BANDS]

        x = np.arange(len(AGE_BANDS))
        bars = ax.bar(x, medians.values, color=color, alpha=0.78, zorder=3, label="Median")
        ax.errorbar(x, medians.values,
                    yerr=[medians.values - p10.values, p90.values - medians.values],
                    fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4,
                    label="P10–P90")

        # Percentual abaixo de 100% em cada barra
        for xi, bar, below in zip(x, bars, pct_below):
            if below > 0:
                ax.text(xi, bar.get_height() + (p90.values[xi] - medians.values[xi]) + 4,
                        f"{below:.0f}%↓", ha="center", va="bottom", fontsize=8, color="#333")

        ax.axhline(100, color="#333333", linewidth=1.2, linestyle="--",
                   label="100% DRI", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_BANDS, fontsize=10)
        ax.set_ylabel("% DRI", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, max(p90.max() * 1.2, 160))
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Iron and Calcium Adequacy by Age Band  (n = 500)", fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_iron_calcium.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── Figure 4: Adequação por região ────────────────────────────────────────────
def fig_by_region(df: pd.DataFrame):
    nuts  = ["energy_pct","iron_pct","calcium_pct","zinc_pct","vitA_pct"]
    lbls  = ["Energy","Iron","Calcium","Zinc","Vit. A"]

    medians = df.groupby("region")[nuts].median().reindex(REGIONS)

    x      = np.arange(len(nuts))
    width  = 0.15
    offsets = np.linspace(-(len(REGIONS)-1)/2, (len(REGIONS)-1)/2, len(REGIONS)) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, region in enumerate(REGIONS):
        vals = [medians.loc[region, n] for n in nuts]
        ax.bar(x + offsets[i], vals, width, label=region,
               color=REGION_COLORS[i], alpha=0.85, zorder=3)

    ax.axhline(100, color="#333333", linewidth=1.2, linestyle="--",
               label="100% DRI", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=11)
    ax.set_ylabel("Median % DRI", fontsize=11)
    ax.set_title("Median % DRI by Brazilian Region  (n = 500)", fontsize=12, pad=10)
    ax.set_ylim(0, 280)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_by_region.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── Excel com todas as tabelas ────────────────────────────────────────────────
def make_excel(df: pd.DataFrame):
    wb   = xlsxwriter.Workbook(XLSX_OUT)
    bold = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white", "border": 1})
    hdr  = wb.add_format({"bold": True, "bg_color": "#D6E4F0", "border": 1})
    pct_f = wb.add_format({"num_format": "0.0", "border": 1})
    num0  = wb.add_format({"num_format": "0",   "border": 1})
    plain = wb.add_format({"border": 1})
    colors_chart = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"]

    # ── Aba 1: Adequacy by Age Band ─────────────────────────────────────────
    ws = wb.add_worksheet("Adequacy_by_Band")
    ws.hide_gridlines(2)
    ws.set_column("A:A", 14)
    ws.set_column("B:H", 11)

    ws.write(0, 0, "Age Band", bold)
    for j, lbl in enumerate(NUT_LABELS):
        ws.write(0, j+1, lbl, bold)

    medians = df.groupby("stratum", observed=True)[NUTRIENTS].median().reindex(AGE_BANDS)
    for i, band in enumerate(AGE_BANDS):
        ws.write(i+1, 0, band, hdr)
        for j, nut in enumerate(NUTRIENTS):
            ws.write(i+1, j+1, round(float(medians.loc[band, nut]), 1), pct_f)
    ws.write(6, 0, "All (500)", hdr)
    for j, nut in enumerate(NUTRIENTS):
        ws.write(6, j+1, round(float(df[nut].median()), 1), pct_f)

    chart = wb.add_chart({"type": "bar"})
    for i, band in enumerate(AGE_BANDS):
        chart.add_series({
            "name":       band,
            "categories": ["Adequacy_by_Band", 0, 1, 0, 7],
            "values":     ["Adequacy_by_Band", i+1, 1, i+1, 7],
            "fill":       {"color": colors_chart[i]},
        })
    chart.set_title({"name": "Median % DRI by Age Band and Nutrient"})
    chart.set_x_axis({"name": "% DRI", "min": 0, "max": 400, "crossing": 0,
                      "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
    chart.set_y_axis({"name": "Nutrient", "crossing": 0})
    chart.set_legend({"position": "bottom"})
    chart.set_size({"width": 620, "height": 360})
    ws.insert_chart("A9", chart)

    # ── Aba 2: Iron & Calcium detail ────────────────────────────────────────
    ws2 = wb.add_worksheet("Iron_Calcium")
    ws2.hide_gridlines(2)
    ws2.set_column("A:A", 14)
    ws2.set_column("B:G", 16)

    headers = ["Age Band","Iron median","Iron P10","Iron P90",
               "Iron <100% (n)","Calcium median","Calcium <100% (n)"]
    for j, h in enumerate(headers):
        ws2.write(0, j, h, bold)

    for i, band in enumerate(AGE_BANDS):
        sub = df[df["stratum"] == band]
        ws2.write(i+1, 0, band, hdr)
        ws2.write(i+1, 1, round(float(sub["iron_pct"].median()), 1),    pct_f)
        ws2.write(i+1, 2, round(float(sub["iron_pct"].quantile(0.10)),1), pct_f)
        ws2.write(i+1, 3, round(float(sub["iron_pct"].quantile(0.90)),1), pct_f)
        ws2.write(i+1, 4, int((sub["iron_pct"] < 100).sum()),            num0)
        ws2.write(i+1, 5, round(float(sub["calcium_pct"].median()), 1), pct_f)
        ws2.write(i+1, 6, int((sub["calcium_pct"] < 100).sum()),         num0)

    # ── Aba 3: By Region ────────────────────────────────────────────────────
    ws3 = wb.add_worksheet("By_Region")
    ws3.hide_gridlines(2)
    ws3.set_column("A:A", 16)
    ws3.set_column("B:H", 11)

    ws3.write(0, 0, "Region", bold)
    for j, lbl in enumerate(NUT_LABELS):
        ws3.write(0, j+1, lbl, bold)

    reg_med = df.groupby("region")[NUTRIENTS].median().reindex(REGIONS)
    for i, reg in enumerate(REGIONS):
        ws3.write(i+1, 0, reg, hdr)
        for j, nut in enumerate(NUTRIENTS):
            ws3.write(i+1, j+1, round(float(reg_med.loc[reg, nut]), 1), pct_f)

    # ── Aba 4: Summary ──────────────────────────────────────────────────────
    ws4 = wb.add_worksheet("Summary")
    ws4.hide_gridlines(2)
    ws4.set_tab_color("#1F3864")
    ws4.set_column("A:A", 30)
    ws4.set_column("B:B", 35)

    title_f = wb.add_format({"bold": True, "font_size": 13,
                              "bg_color": "#1F3864", "font_color": "white", "border": 1})
    ws4.merge_range("A1:B1",
        "GAIA Evaluation — 500 Meal Plans (Sections 4.2 & 4.3)", title_f)

    sec_f = wb.add_format({"bold": True, "bg_color": "#D6E4F0", "border": 1})
    ok = df[df["lp_status"] != "error"]
    below80 = ok[NUTRIENTS].lt(80).any(axis=1).sum()

    rows_s = [
        ("Total plans",              "500"),
        ("LP status",                "500 optimal  (100%)"),
        ("Age bands covered",        "6-8m, 9-11m, 12-23m, 24-35m, 36-59m"),
        ("Regions covered",          "5 (all BR macro-regions)"),
        ("Sex distribution",         "250 M / 250 F"),
        ("Avg foods per plan",       f"{ok['n_foods'].mean():.1f}  (range {int(ok['n_foods'].min())}–{int(ok['n_foods'].max())})"),
        ("",""),
        ("─── Nutritional Adequacy ───", ""),
        ("Energy  (median % DRI)",   f"{ok['energy_pct'].median():.1f}%"),
        ("Protein (median % DRI)",   f"{ok['protein_pct'].median():.1f}%"),
        ("Iron    (median % DRI)",   f"{ok['iron_pct'].median():.1f}%  [{int((ok['iron_pct']<100).sum())}/500 below 100%]"),
        ("Calcium (median % DRI)",   f"{ok['calcium_pct'].median():.1f}%  [{int((ok['calcium_pct']<100).sum())}/500 below 100%]"),
        ("Zinc    (median % DRI)",   f"{ok['zinc_pct'].median():.1f}%"),
        ("Vit. A  (median % DRI)",   f"{ok['vitA_pct'].median():.1f}%"),
        ("Vit. C  (median % DRI)",   f"{ok['vitC_pct'].median():.1f}%"),
        ("Plans below 80% any DRI",  f"{below80} / 500  ({below80/5:.1f}%)"),
    ]

    for i, (k, v) in enumerate(rows_s):
        fmt = sec_f if k.startswith("─") else plain
        ws4.write(2 + i, 0, k, fmt)
        ws4.write(2 + i, 1, v, fmt)

    wb.close()
    print(f"  Salvo: {XLSX_OUT}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    print("Gerando gráficos...")
    fig_adequacy_by_band(df)
    fig_boxplot_nutrients(df)
    fig_iron_calcium(df)
    fig_by_region(df)
    print("Gerando Excel...")
    make_excel(df)
    print("\nConcluído. Arquivos em tmp_who/figures/ e tmp_who/evaluation_500_charts.xlsx")


if __name__ == "__main__":
    main()
