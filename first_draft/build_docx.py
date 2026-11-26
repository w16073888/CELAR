from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CELAR_first_draft_body.txt"
OUTPUT = ROOT / "CELAR_first_draft_body.docx"
FONT = "Times New Roman"

H1 = {
    "Introduction",
    "Related Work",
    "Method",
    "Experimental Protocol",
    "Results",
    "Discussion",
    "Limitations and Next Steps",
    "Conclusion",
}
H2 = {
    "Problem Formulation",
    "Training Objectives and Safety",
    "MuJoCo Engineering Dataset",
    "Representation and Control Evaluation",
    "Data and Rendering Validation",
    "Action Representation and Behavior Cloning",
    "Online Reinforcement Learning",
    "Held-Out Embodiment Adaptation",
}


def set_run_font(run, *, bold: bool | None = None, size: float = 12) -> None:
    run.font.name = FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold


def style_font(style, *, bold: bool | None = None, size: float = 12) -> None:
    style.font.name = FONT
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        style.font.bold = bold


def first_line_indent(paragraph) -> None:
    paragraph.paragraph_format.first_line_indent = Inches(0.5)
    p_pr = paragraph._p.get_or_add_pPr()
    ind = p_pr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        p_pr.append(ind)
    ind.set(qn("w:firstLine"), str(int(0.5 * 1440)))
    ind.set(qn("w:firstLineChars"), "200")


def remove_paragraph_borders(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is not None:
        p_pr.remove(borders)


def remove_style_borders(style) -> None:
    p_pr = style._element.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is not None:
        p_pr.remove(borders)


def configure_paragraph(paragraph, *, alignment, indent: bool, bold: bool = False) -> None:
    paragraph.alignment = alignment
    paragraph.paragraph_format.line_spacing = 2.0
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.widow_control = True
    if indent:
        first_line_indent(paragraph)
    for run in paragraph.runs:
        set_run_font(run, bold=bold, size=12)


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    style_font(normal, bold=False, size=12)
    normal.paragraph_format.line_spacing = 2.0
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)

    title_style = doc.styles["Title"]
    style_font(title_style, bold=True, size=12)
    title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_style.paragraph_format.line_spacing = 2.0
    title_style.paragraph_format.space_before = Pt(0)
    title_style.paragraph_format.space_after = Pt(0)
    remove_style_borders(title_style)

    heading1 = doc.styles["Heading 1"]
    style_font(heading1, bold=True, size=12)
    heading1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading1.paragraph_format.line_spacing = 2.0
    heading1.paragraph_format.space_before = Pt(0)
    heading1.paragraph_format.space_after = Pt(0)

    heading2 = doc.styles["Heading 2"]
    style_font(heading2, bold=True, size=12)
    heading2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    heading2.paragraph_format.line_spacing = 2.0
    heading2.paragraph_format.space_before = Pt(0)
    heading2.paragraph_format.space_after = Pt(0)

    lines = [line.strip() for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    first_content = True
    for line in lines:
        if not line:
            continue
        if first_content:
            paragraph = doc.add_paragraph(style="Title")
            paragraph.add_run(line)
            configure_paragraph(
                paragraph,
                alignment=WD_ALIGN_PARAGRAPH.CENTER,
                indent=False,
                bold=True,
            )
            remove_paragraph_borders(paragraph)
            first_content = False
        elif line in H1:
            paragraph = doc.add_paragraph(style="Heading 1")
            paragraph.add_run(line)
            configure_paragraph(
                paragraph,
                alignment=WD_ALIGN_PARAGRAPH.CENTER,
                indent=False,
                bold=True,
            )
        elif line in H2:
            paragraph = doc.add_paragraph(style="Heading 2")
            paragraph.add_run(line)
            configure_paragraph(
                paragraph,
                alignment=WD_ALIGN_PARAGRAPH.LEFT,
                indent=False,
                bold=True,
            )
        else:
            paragraph = doc.add_paragraph(style="Normal")
            paragraph.add_run(line)
            configure_paragraph(
                paragraph,
                alignment=WD_ALIGN_PARAGRAPH.LEFT,
                indent=True,
                bold=False,
            )

    doc.core_properties.title = (
        "CELAR: A Two-Stage Feasibility Study of Cross-Embodiment Latent Action Reinforcement Learning"
    )
    doc.core_properties.subject = "First draft paper body"
    doc.core_properties.author = ""
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
