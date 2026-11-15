import pandas as pd
import numpy as np
import xlsxwriter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv("tmp_who/batch_results.csv")
df["age_band"] = pd.cut(df["age_months"], bins=[0,5.5,8.5,11.5,23.5,60],
                        labels=["0-5m","6-8m","9-11m","12-23m","24-59m"])
foods = pd.read_csv("data/processed/unified_foods.csv")

nutrients  = ["energy_pct","protein_pct","iron_pct","calcium_pct","zinc_pct","vitA_pct","vitC_pct"]
nut_labels = ["Energy","Protein","Iron","Calcium","Zinc","Vit. A","Vit. C"]
age_bands  = ["6-8m","9-11m","12-23m","24-35m","36-59m"]
nas_cols   = {"6-8m":"NAS_6_8m","9-11m":"NAS_9_11m",
              "12-23m":"NAS_12_23m","24-35m":"NAS_24_35m","36-59m":"NAS_36_59m"}

wb   = xlsxwriter.Workbook("tmp_who/evaluation_charts.xlsx")
bold = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white", "border": 1})
hdr  = wb.add_format({"bold": True, "bg_color": "#D6E4F0", "border": 1})
pct_f = wb.add_format({"num_format": "0.0", "border": 1})
nas_f = wb.add_format({"num_format": "0.000", "border": 1})
num0  = wb.add_format({"num_format": "0", "border": 1})
plain = wb.add_format({"border": 1})
colors = ["#4CAF50","#2196F3","#F44336","#9C27B0","#FF9800","#00BCD4","#E91E63"]

# ── Sheet 1: Adequacy ──────────────────────────────────────────────────────────
ws_a = wb.add_worksheet("Adequacy")
ws_a.hide_gridlines(2)
ws_a.set_column("A:A", 14)
ws_a.set_column("B:H", 11)

ws_a.write(0, 0, "Age Band", bold)
for j, lbl in enumerate(nut_labels):
    ws_a.write(0, j+1, lbl, bold)

medians = df.groupby("age_band", observed=True)[nutrients].median().reindex(age_bands)
for i, band in enumerate(age_bands):
    ws_a.write(i+1, 0, band, hdr)
    for j, nut in enumerate(nutrients):
        ws_a.write(i+1, j+1, round(float(medians.loc[band, nut]), 1), pct_f)

ws_a.write(6, 0, "All Plans", hdr)
for j, nut in enumerate(nutrients):
    ws_a.write(6, j+1, round(float(df[nut].median()), 1), pct_f)

chart_a = wb.add_chart({"type": "bar"})
for i, band in enumerate(age_bands):
    chart_a.add_series({
        "name":       band,
        "categories": ["Adequacy", 0, 1, 0, 7],
        "values":     ["Adequacy", i+1, 1, i+1, 7],
        "fill":       {"color": colors[i]},
    })
chart_a.set_title({"name": "Median % DRI by Age Band and Nutrient"})
chart_a.set_x_axis({"name": "% DRI", "min": 0, "max": 400, "crossing": 0,
                    "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
chart_a.set_y_axis({"name": "Nutrient", "crossing": 0})
chart_a.set_legend({"position": "bottom"})
chart_a.set_size({"width": 620, "height": 360})
chart_a.set_chartarea({"border": {"none": True}, "fill": {"color": "white"}})
chart_a.set_plotarea({"fill": {"color": "white"}})
ws_a.insert_chart("A9", chart_a)

# ── Sheet 2: Iron & Calcium ────────────────────────────────────────────────────
ws_ic = wb.add_worksheet("Iron_Calcium")
ws_ic.hide_gridlines(2)
ws_ic.set_column("A:A", 14)
ws_ic.set_column("B:E", 16)

ws_ic.write(0, 0, "Age Band",           bold)
ws_ic.write(0, 1, "Iron median",        bold)
ws_ic.write(0, 2, "Iron below 100%",    bold)
ws_ic.write(0, 3, "Calcium median",     bold)
ws_ic.write(0, 4, "Calcium below 100%", bold)

for i, band in enumerate(age_bands):
    sub = df[df["age_band"] == band]
    ws_ic.write(i+1, 0, band, hdr)
    ws_ic.write(i+1, 1, round(float(sub["iron_pct"].median()), 1),    pct_f)
    ws_ic.write(i+1, 2, int((sub["iron_pct"] < 100).sum()),            num0)
    ws_ic.write(i+1, 3, round(float(sub["calcium_pct"].median()), 1), pct_f)
    ws_ic.write(i+1, 4, int((sub["calcium_pct"] < 100).sum()),         num0)

chart_ic = wb.add_chart({"type": "column"})
chart_ic.add_series({
    "name": "Iron median % DRI",
    "categories": ["Iron_Calcium", 1, 0, 5, 0],
    "values":     ["Iron_Calcium", 1, 1, 5, 1],
    "fill": {"color": "#F44336"},
})
chart_ic.add_series({
    "name": "Calcium median % DRI",
    "categories": ["Iron_Calcium", 1, 0, 5, 0],
    "values":     ["Iron_Calcium", 1, 3, 5, 3],
    "fill": {"color": "#9C27B0"},
})
chart_ic.set_title({"name": "Iron & Calcium Median % DRI by Age Band"})
chart_ic.set_x_axis({"name": "Age Band", "crossing": 0})
chart_ic.set_y_axis({"name": "% DRI", "min": 0, "max": 200, "crossing": 0,
                     "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
chart_ic.set_legend({"position": "bottom"})
chart_ic.set_size({"width": 500, "height": 340})
chart_ic.set_chartarea({"border": {"none": True}, "fill": {"color": "white"}})
chart_ic.set_plotarea({"fill": {"color": "white"}})
ws_ic.insert_chart("A8", chart_ic)

# ── Sheet 3: NAS ──────────────────────────────────────────────────────────────
ws_n = wb.add_worksheet("NAS")
ws_n.hide_gridlines(2)
ws_n.set_column("A:A", 14)
ws_n.set_column("B:D", 18)

ws_n.write(0, 0, "Age Band",          bold)
ws_n.write(0, 1, "Food Bank (mean)",  bold)
ws_n.write(0, 2, "Plans (wavg mean)", bold)
ws_n.write(0, 3, "Plans (wavg med)",  bold)

for i, band in enumerate(age_bands):
    col    = nas_cols[band]
    bank_m = float(foods[col].dropna().mean())
    sub    = df[df["age_band"] == band]["nas_wavg"]
    ws_n.write(i+1, 0, band, hdr)
    ws_n.write(i+1, 1, round(bank_m, 4),             nas_f)
    ws_n.write(i+1, 2, round(float(sub.mean()), 4),   nas_f)
    ws_n.write(i+1, 3, round(float(sub.median()), 4), nas_f)

chart_n = wb.add_chart({"type": "column"})
chart_n.add_series({
    "name": "Food Bank (mean NAS)",
    "categories": ["NAS", 1, 0, 5, 0],
    "values":     ["NAS", 1, 1, 5, 1],
    "fill": {"color": "#90CAF9"},
})
chart_n.add_series({
    "name": "Meal Plans (wavg mean)",
    "categories": ["NAS", 1, 0, 5, 0],
    "values":     ["NAS", 1, 2, 5, 2],
    "fill": {"color": "#1565C0"},
})
chart_n.set_title({"name": "NAS: Food Bank vs. Meal Plans by Age Band"})
chart_n.set_x_axis({"name": "Age Band", "crossing": 0})
chart_n.set_y_axis({"name": "NAS (gram-weighted avg)", "min": 0, "max": 0.50,
                    "crossing": 0,
                    "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
chart_n.set_legend({"position": "bottom"})
chart_n.set_size({"width": 500, "height": 340})
chart_n.set_chartarea({"border": {"none": True}, "fill": {"color": "white"}})
chart_n.set_plotarea({"fill": {"color": "white"}})
ws_n.insert_chart("A8", chart_n)

# ── Sheet 4: Overview ─────────────────────────────────────────────────────────
ws_ov = wb.add_worksheet("Overview")
ws_ov.hide_gridlines(2)
ws_ov.set_tab_color("#1F3864")

def clone_bar():
    c = wb.add_chart({"type": "bar"})
    for i, band in enumerate(age_bands):
        c.add_series({
            "name":       band,
            "categories": ["Adequacy", 0, 1, 0, 7],
            "values":     ["Adequacy", i+1, 1, i+1, 7],
            "fill":       {"color": colors[i]},
        })
    c.set_title({"name": "Median % DRI by Nutrient"})
    c.set_x_axis({"min": 0, "max": 400, "crossing": 0,
                  "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
    c.set_y_axis({"crossing": 0})
    c.set_legend({"position": "bottom"})
    c.set_size({"width": 440, "height": 320})
    c.set_chartarea({"border": {"none": True}, "fill": {"color": "white"}})
    c.set_plotarea({"fill": {"color": "white"}})
    return c

def clone_col():
    c = wb.add_chart({"type": "column"})
    c.add_series({"name": "Iron",
                  "categories": ["Iron_Calcium", 1, 0, 5, 0],
                  "values":     ["Iron_Calcium", 1, 1, 5, 1],
                  "fill": {"color": "#F44336"}})
    c.add_series({"name": "Calcium",
                  "categories": ["Iron_Calcium", 1, 0, 5, 0],
                  "values":     ["Iron_Calcium", 1, 3, 5, 3],
                  "fill": {"color": "#9C27B0"}})
    c.set_title({"name": "Iron & Calcium Median % DRI"})
    c.set_x_axis({"crossing": 0})
    c.set_y_axis({"min": 0, "max": 200, "crossing": 0,
                  "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
    c.set_legend({"position": "bottom"})
    c.set_size({"width": 380, "height": 320})
    c.set_chartarea({"border": {"none": True}, "fill": {"color": "white"}})
    c.set_plotarea({"fill": {"color": "white"}})
    return c

def clone_nas():
    c = wb.add_chart({"type": "column"})
    c.add_series({"name": "Food Bank",
                  "categories": ["NAS", 1, 0, 5, 0],
                  "values":     ["NAS", 1, 1, 5, 1],
                  "fill": {"color": "#90CAF9"}})
    c.add_series({"name": "Meal Plans",
                  "categories": ["NAS", 1, 0, 5, 0],
                  "values":     ["NAS", 1, 2, 5, 2],
                  "fill": {"color": "#1565C0"}})
    c.set_title({"name": "NAS: Food Bank vs Meal Plans"})
    c.set_x_axis({"crossing": 0})
    c.set_y_axis({"min": 0, "max": 0.50, "crossing": 0,
                  "major_gridlines": {"visible": True, "line": {"color": "#E0E0E0"}}})
    c.set_legend({"position": "bottom"})
    c.set_size({"width": 380, "height": 320})
    c.set_chartarea({"border": {"none": True}, "fill": {"color": "white"}})
    c.set_plotarea({"fill": {"color": "white"}})
    return c

ws_ov.insert_chart("B2",  clone_bar())
ws_ov.insert_chart("P2",  clone_col())
ws_ov.insert_chart("AB2", clone_nas())

# ── Sheet 5: Summary ──────────────────────────────────────────────────────────
ws_s = wb.add_worksheet("Summary")
ws_s.hide_gridlines(2)
ws_s.set_column("A:A", 26)
ws_s.set_column("B:B", 32)

title_f = wb.add_format({"bold": True, "font_size": 13,
                          "bg_color": "#1F3864", "font_color": "white", "border": 1})
ws_s.merge_range("A1:B1", "GAIA Evaluation — 100 Meal Plans (Sections 4.2 & 4.3)", title_f)

rows = [
    ("Total plans",                "100"),
    ("LP status",                  "100% optimal"),
    ("Age bands covered",          "0-5m, 6-8m, 9-11m, 12-23m, 24-59m"),
    ("Regions covered",            "5 (all BR macro-regions)"),
    ("Avg foods per plan",         "8.1 (range 4–12)"),
    ("",                           ""),
    ("─── Nutritional Adequacy ───",""),
    ("Energy  (median % DRI)",     "109.9%  [range 108–112%]"),
    ("Protein (median % DRI)",     "311.2%"),
    ("Iron    (median % DRI)",     "100.4%  [39/100 below 100%]"),
    ("Calcium (median % DRI)",     "100.9%  [45/100 below 100%]"),
    ("Zinc    (median % DRI)",     "139.0%"),
    ("Vit. A  (median % DRI)",     "113.0%"),
    ("Vit. C  (median % DRI)",     "670.5%"),
    ("Plans below 77% DRI",        "0 / 100"),
    ("",                           ""),
    ("─── Affordability (NAS) ───", ""),
    ("NAS plan mean",               "0.187"),
    ("NAS plan median",             "0.176"),
    ("NAS plan range",              "0.135 – 0.259"),
]

sec_f = wb.add_format({"bold": True, "bg_color": "#D6E4F0", "border": 1})
for i, (k, v) in enumerate(rows):
    fmt = sec_f if k.startswith("─") else plain
    ws_s.write(2 + i, 0, k, fmt)
    ws_s.write(2 + i, 1, v, fmt)

wb.close()
print("evaluation_charts.xlsx criado com sucesso.")
