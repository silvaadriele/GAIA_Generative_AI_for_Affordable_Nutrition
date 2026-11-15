"""
Combina os 5 planos GAIA em um unico documento Word para avaliacao nutricional.
Saida: avaliacaonutri/planos_gaia_avaliacao_nutricional.docx
"""

import os
import re
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

AVALIA_DIR = os.path.join(
    os.path.expanduser("~"), "OneDrive - SAP SE", "Desktop", "TCC", "avaliacaonutri"
)
OUT_FILE = os.path.join(AVALIA_DIR, "planos_gaia_avaliacao_nutricional.docx")

PLAN_FILES = [
    "plano_01_6_8m_6m_F_S.txt",
    "plano_02_9_11m_11m_M_CO.txt",
    "plano_03_12_23m_19m_F_CO.txt",
    "plano_04_24_35m_31m_F_S.txt",
    "plano_05_36_59m_57m_M_NE.txt",
]

BAND_LABELS = {
    "plano_01": "Faixa 1 — 6 a 8 meses",
    "plano_02": "Faixa 2 — 9 a 11 meses",
    "plano_03": "Faixa 3 — 12 a 23 meses",
    "plano_04": "Faixa 4 — 24 a 35 meses",
    "plano_05": "Faixa 5 — 36 a 59 meses",
}

BLUE     = RGBColor(31, 73, 125)
DARKGRAY = RGBColor(60, 60, 60)
GRAY     = RGBColor(100, 100, 100)
LIGHTGRAY= RGBColor(140, 140, 140)
GREEN    = RGBColor(0, 112, 0)
ORANGE   = RGBColor(180, 90, 0)


def font(run, size=11, bold=False, italic=False, color=None):
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color


def add_rule(doc, color="AAAAAA", thickness="6"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), thickness)
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_page_break(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_break(__import__("docx.enum.text", fromlist=["WD_BREAK"]).WD_BREAK.PAGE)


# ── parsers ────────────────────────────────────────────────────────────────────

def parse_plan(text):
    """Split raw plan text into labeled sections."""
    sections = []
    current_label = None
    current_lines = []

    SECTION_MARKERS = {
        "── PERFIL DA CRIANÇA": "perfil",
        "── AVALIAÇÃO ANTROPOMÉTRICA": "antropometria",
        "── METAS NUTRICIONAIS": "metas",
        "── PLANO DE REFEIÇÕES": "refeicoes",
        "── COBERTURA NUTRICIONAL": "cobertura",
        "── CUSTO DIÁRIO": "custo",
        "── VALIDAÇÃO": "validacao",
        "── ALEITAMENTO MATERNO": "aleitamento",
        "── ORIENTAÇÕES AO CUIDADOR": "orientacoes",
    }

    for line in text.splitlines():
        matched = None
        for marker, label in SECTION_MARKERS.items():
            if marker in line:
                matched = label
                break
        if matched:
            if current_label:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = matched
            current_lines = []
        else:
            if current_label:
                current_lines.append(line)

    if current_label:
        sections.append((current_label, "\n".join(current_lines).strip()))

    return dict(sections)


def render_plan(doc, plan_num, filename, text):
    key = f"plano_{plan_num:02d}"
    band_label = BAND_LABELS.get(key, f"Plano {plan_num}")

    sec = parse_plan(text)

    # ── cabeçalho do plano ─────────────────────────────────────────────────────
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run(f"Plano {plan_num}  —  {band_label}")
    font(r, size=15, bold=True, color=BLUE)

    add_rule(doc, color="1F497D", thickness="12")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── perfil ─────────────────────────────────────────────────────────────────
    _section_header(doc, "Perfil da Criança")
    for line in sec.get("perfil", "").splitlines():
        line = line.strip()
        if line:
            _info_line(doc, line)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── antropometria ──────────────────────────────────────────────────────────
    _section_header(doc, "Avaliação Antropométrica (z-scores OMS)")
    for line in sec.get("antropometria", "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "→" in line:
            parts = line.split("→")
            label_part = parts[0].strip().rstrip(":")
            status = parts[1].strip() if len(parts) > 1 else ""
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_after = Pt(1)
            r1 = p.add_run(f"{label_part}  →  ")
            font(r1, size=10.5, color=DARKGRAY)
            col = GREEN if "adequado" in status.lower() or "normal" in status.lower() else ORANGE
            r2 = p.add_run(status)
            font(r2, size=10.5, bold=True, color=col)
        else:
            _info_line(doc, line, bold=("CLASSIFICAÇÃO" in line or "Ajuste" in line))

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── metas DRI ─────────────────────────────────────────────────────────────
    _section_header(doc, "Metas Nutricionais Diárias (DRI Ajustado)")
    for line in sec.get("metas", "").splitlines():
        line = line.strip()
        if line:
            _info_line(doc, line)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── plano de refeições ─────────────────────────────────────────────────────
    _section_header(doc, "Plano de Refeições")
    refeicoes_text = sec.get("refeicoes", "")
    current_meal = None
    for line in refeicoes_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Status"):
            _info_line(doc, stripped, italic=True)
            continue
        if stripped.startswith("[") and "]" in stripped:
            current_meal = stripped
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.3)
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after  = Pt(1)
            r = p.add_run(stripped)
            font(r, size=11, bold=True, color=BLUE)
        elif stripped.startswith("•"):
            _food_line(doc, stripped)
        else:
            _info_line(doc, stripped)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── cobertura ──────────────────────────────────────────────────────────────
    _section_header(doc, "Cobertura Nutricional")
    for line in sec.get("cobertura", "").splitlines():
        line = line.strip()
        if not line:
            continue
        pct_match = re.search(r"\((\d+[\.,]\d+)%\)", line)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_after = Pt(1)
        if pct_match:
            pct = float(pct_match.group(1).replace(",", "."))
            col = GREEN if 80 <= pct <= 150 else ORANGE
            r = p.add_run(line)
            font(r, size=10.5, color=col)
        else:
            r = p.add_run(line)
            font(r, size=10.5, color=DARKGRAY)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── custo ──────────────────────────────────────────────────────────────────
    _section_header(doc, "Custo Diário Estimado")
    for line in sec.get("custo", "").splitlines():
        line = line.strip()
        if line:
            _info_line(doc, line, bold="Custo diário" in line)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── validação ──────────────────────────────────────────────────────────────
    _section_header(doc, "Validação")
    for line in sec.get("validacao", "").splitlines():
        line = line.strip()
        if not line:
            continue
        col = GREEN if "APROVADO" in line else ORANGE
        _info_line(doc, line, bold="Resultado" in line, color=col if "Resultado" in line else None)

    # ── aleitamento (se houver) ────────────────────────────────────────────────
    if sec.get("aleitamento"):
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        _section_header(doc, "Aleitamento Materno")
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(sec["aleitamento"].strip())
        font(r, size=10, italic=True, color=GRAY)

    # ── orientações ao cuidador ───────────────────────────────────────────────
    if sec.get("orientacoes"):
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        _section_header(doc, "Orientações ao Cuidador (geradas por IA)")
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(sec["orientacoes"].strip())
        font(r, size=10.5, color=DARKGRAY)


def _section_header(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(3)
    r = p.add_run(title)
    font(r, size=11, bold=True, color=RGBColor(50, 50, 50))
    # underline leve
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "CCCCCC")
    pBdr.append(bottom)
    pPr.append(pBdr)


def _info_line(doc, text, bold=False, italic=False, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text)
    font(r, size=10.5, bold=bold, italic=italic, color=color or DARKGRAY)


def _food_line(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text)
    font(r, size=10.5, color=DARKGRAY)


# ── main ───────────────────────────────────────────────────────────────────────

def build():
    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    # capa / título geral
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("GAIA — Planos Alimentares Pediátricos")
    font(r, size=18, bold=True, color=BLUE)

    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = s.add_run("5 Planos para Avaliação Nutricional  |  ENANI-2019  |  TCC UNISINOS 2026")
    font(rs, size=11, italic=True, color=GRAY)

    doc.add_paragraph()

    # índice simples
    idx_hdr = doc.add_paragraph()
    ri = idx_hdr.add_run("Planos incluídos neste documento:")
    font(ri, size=11, bold=True, color=DARKGRAY)

    labels = [
        "Plano 1  —  Faixa 1: 6 a 8 meses",
        "Plano 2  —  Faixa 2: 9 a 11 meses",
        "Plano 3  —  Faixa 3: 12 a 23 meses",
        "Plano 4  —  Faixa 4: 24 a 35 meses",
        "Plano 5  —  Faixa 5: 36 a 59 meses",
    ]
    for lbl in labels:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(lbl)
        font(r, size=10.5, color=DARKGRAY)

    add_rule(doc, color="1F497D", thickness="8")

    # planos
    for i, fname in enumerate(PLAN_FILES, 1):
        fpath = os.path.join(AVALIA_DIR, fname)
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        if i > 1:
            add_page_break(doc)

        render_plan(doc, i, fname, text)

    doc.save(OUT_FILE)
    print(f"Documento salvo em:\n  {OUT_FILE}")


if __name__ == "__main__":
    build()
