"""
Gera documento Word com perguntas para avaliacao nutricional dos planos GAIA.
Saida: avaliacaonutri/perguntas_avaliacao_nutricionista.docx
"""

import os
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT_DIR = os.path.join(
    os.path.expanduser("~"), "OneDrive - SAP SE", "Desktop", "TCC", "avaliacaonutri"
)
OUT_FILE = os.path.join(OUT_DIR, "perguntas_avaliacao_nutricionista.docx")

TITLE   = "GAIA — Avaliação Nutricional por Especialista"
SUBTITLE = "Perguntas para Avaliação dos Planos Alimentares Gerados"

INTRO = (
    "Os arquivos anexos contêm cinco planos alimentares pediátricos gerados pelo GAIA "
    "(Sistema Multiagente para Planejamento Alimentar Pediátrico), um sistema desenvolvido "
    "como Trabalho de Conclusão de Curso na UNISINOS (2026). Cada plano corresponde a uma "
    "faixa etária distinta (6–8 meses, 9–11 meses, 12–23 meses, 24–35 meses e 36–59 meses) "
    "e foi gerado automaticamente a partir do perfil antropométrico de crianças reais do "
    "inquérito ENANI-2019, utilizando programação linear com restrições nutricionais baseadas "
    "em DRI/OMS e alimentos típicos de banco de alimentos brasileiro.\n\n"
    "Para fins desta avaliação, gostaríamos de sua opinião como profissional de nutrição "
    "sobre os planos apresentados, respondendo às perguntas abaixo."
)

PERGUNTAS = [
    {
        "num": "1",
        "titulo": "Adequação nutricional e escolha dos alimentos",
        "texto": (
            "Considerando o perfil de cada criança (idade, estado nutricional e metas de DRI "
            "ajustadas apresentadas no plano), os alimentos selecionados e as quantidades "
            "propostas são clinicamente adequados? Você faria alterações significativas em "
            "algum plano — e se sim, quais?"
        ),
        "objetivo": (
            "Validar se o sistema produz planos nutricionalmente viáveis, não apenas "
            "matematicamente corretos."
        ),
    },
    {
        "num": "2",
        "titulo": "Adequação para a faixa etária (texturas, combinações e distribuição)",
        "texto": (
            "As escolhas alimentares, as texturas implícitas e a distribuição entre as "
            "refeições são compatíveis com as recomendações de alimentação complementar para "
            "cada faixa etária avaliada? Há algum alimento que você consideraria inapropriado "
            "ou ausente para a idade?"
        ),
        "objetivo": (
            "Avaliar se a seleção de alimentos e o fracionamento por refeição respeitam as "
            "diretrizes de introdução alimentar (Guia Alimentar para Crianças Brasileiras "
            "Menores de 2 Anos / PNAN)."
        ),
    },
    {
        "num": "3",
        "titulo": "Utilidade como ferramenta de apoio profissional",
        "texto": (
            "Um sistema que gera planos individualizados automaticamente a partir de dados "
            "antropométricos, respeitando restrições nutricionais e o contexto de banco de "
            "alimentos, seria útil como ponto de partida no seu trabalho clínico ou em "
            "programas de nutrição em saúde pública? O que precisaria ser diferente para que "
            "você o utilizasse — ou recomendasse — na prática?"
        ),
        "objetivo": (
            "Validar o valor do GAIA como ferramenta de suporte à decisão profissional, "
            "não como substituto ao nutricionista."
        ),
    },
    {
        "num": "4",
        "titulo": "Limitações e o que falta para uso real",
        "texto": (
            "Quais são as principais limitações que impediriam esses planos de serem "
            "entregues diretamente às famílias sem revisão profissional? O que o sistema "
            "precisaria incluir ou melhorar para ter aplicabilidade clínica real?"
        ),
        "objetivo": (
            "Identificar lacunas concretas que possam ser documentadas como trabalhos futuros "
            "no artigo."
        ),
    },
]

RODAPE = (
    "GAIA — Sistema Multiagente para Planejamento Alimentar Pediátrico  |  "
    "TCC UNISINOS 2026  |  Adriele Silva"
)


def set_run_font(run, size=11, bold=False, italic=False, color=None):
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_horizontal_rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "CCCCCC")
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p


def build_doc():
    doc = Document()

    # margens
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    # título
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(TITLE)
    set_run_font(r, size=16, bold=True, color=(31, 73, 125))

    # subtítulo
    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = s.add_run(SUBTITLE)
    set_run_font(rs, size=12, italic=True, color=(89, 89, 89))

    doc.add_paragraph()  # espaço

    # mini formulário identificação
    form_hdr = doc.add_paragraph()
    form_hdr.paragraph_format.space_before = Pt(4)
    form_hdr.paragraph_format.space_after  = Pt(6)
    rfh = form_hdr.add_run("Identificação do Avaliador")
    set_run_font(rfh, size=12, bold=True, color=(31, 73, 125))

    for label, width_chars in [("Nome completo:", 55), ("CRN:", 30)]:
        row = doc.add_paragraph()
        row.paragraph_format.left_indent = Cm(0.5)
        row.paragraph_format.space_after = Pt(6)
        rl = row.add_run(f"{label}  ")
        set_run_font(rl, size=11, bold=True, color=(60, 60, 60))
        rline = row.add_run("_" * width_chars)
        set_run_font(rline, size=11, color=(180, 180, 180))

    add_horizontal_rule(doc)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # intro
    intro_p = doc.add_paragraph()
    intro_p.paragraph_format.space_after = Pt(12)
    ri = intro_p.add_run(INTRO)
    set_run_font(ri, size=11)

    add_horizontal_rule(doc)
    doc.add_paragraph()

    # perguntas
    for perg in PERGUNTAS:
        # cabeçalho da pergunta
        hdr = doc.add_paragraph()
        hdr.paragraph_format.space_before = Pt(6)
        hdr.paragraph_format.space_after  = Pt(4)
        rnum = hdr.add_run(f"Pergunta {perg['num']}  —  ")
        set_run_font(rnum, size=12, bold=True, color=(31, 73, 125))
        rtit = hdr.add_run(perg["titulo"])
        set_run_font(rtit, size=12, bold=True, color=(31, 73, 125))

        # texto da pergunta
        body = doc.add_paragraph()
        body.paragraph_format.left_indent  = Cm(0.5)
        body.paragraph_format.space_after  = Pt(6)
        rb = body.add_run(perg["texto"])
        set_run_font(rb, size=11)

        # linha de resposta (espaço em branco para preencher)
        resp_label = doc.add_paragraph()
        resp_label.paragraph_format.left_indent = Cm(0.5)
        rl = resp_label.add_run("Resposta:")
        set_run_font(rl, size=10, bold=True, color=(89, 89, 89))

        for _ in range(5):
            blank = doc.add_paragraph("_" * 90)
            blank.paragraph_format.left_indent = Cm(0.5)
            blank.paragraph_format.space_after = Pt(2)
            for run in blank.runs:
                set_run_font(run, size=9, color=(180, 180, 180))

        # objetivo (em itálico, menor)
        obj_p = doc.add_paragraph()
        obj_p.paragraph_format.left_indent  = Cm(0.5)
        obj_p.paragraph_format.space_before = Pt(4)
        obj_p.paragraph_format.space_after  = Pt(14)
        ro = obj_p.add_run(f"Objetivo desta pergunta: {perg['objetivo']}")
        set_run_font(ro, size=9, italic=True, color=(120, 120, 120))

        add_horizontal_rule(doc)

    # seção de comentários adicionais
    doc.add_paragraph()
    com_hdr = doc.add_paragraph()
    com_hdr.paragraph_format.space_before = Pt(6)
    com_hdr.paragraph_format.space_after  = Pt(4)
    rch = com_hdr.add_run("Comentários Adicionais  (opcional)")
    set_run_font(rch, size=12, bold=True, color=(31, 73, 125))

    com_desc = doc.add_paragraph()
    com_desc.paragraph_format.left_indent = Cm(0.5)
    com_desc.paragraph_format.space_after = Pt(6)
    rcd = com_desc.add_run(
        "Sinta-se à vontade para registrar qualquer observação geral sobre os planos, "
        "sugestões de melhoria para o sistema ou aspectos que não tenham sido cobertos "
        "pelas perguntas anteriores."
    )
    set_run_font(rcd, size=11)

    for _ in range(8):
        blank = doc.add_paragraph("_" * 90)
        blank.paragraph_format.left_indent = Cm(0.5)
        blank.paragraph_format.space_after = Pt(2)
        for run in blank.runs:
            set_run_font(run, size=9, color=(180, 180, 180))

    add_horizontal_rule(doc)

    # rodapé no último parágrafo
    doc.add_paragraph()
    rf_p = doc.add_paragraph()
    rf_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rf = rf_p.add_run(RODAPE)
    set_run_font(rf, size=9, italic=True, color=(120, 120, 120))

    doc.save(OUT_FILE)
    print(f"Documento salvo em:\n  {OUT_FILE}")


if __name__ == "__main__":
    build_doc()
