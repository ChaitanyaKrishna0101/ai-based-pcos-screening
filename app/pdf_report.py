"""
pdf_report.py — premium, professional PDF summary of a screening result.
Pure-Python (reportlab), no system deps, HF-Spaces friendly.

Design system:
- Header band (brand color) with title + subtitle
- Colored tier badge (rounded rect) instead of plain colored text
- Section cards with left accent rule + consistent spacing scale
- Clean table with zebra striping and a real header row
- Footer with page numbers + generation timestamp on every page
- One typography scale, one color palette — nothing ad-hoc
"""

from __future__ import annotations
import io
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem, Flowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas as pdfcanvas

# ---------------------------------------------------------------- palette --
INK = HexColor("#1f2937")        # body text
MUTED = HexColor("#6b7280")      # secondary text
BRAND = HexColor("#5b21b6")      # deep violet
BRAND_LIGHT = HexColor("#f5f3ff")
BRAND_LINE = HexColor("#ddd6fe")
CARD_BG = HexColor("#faf9fc")
LINE = HexColor("#e5e7eb")

TIER_COLORS = {
    1: (HexColor("#15803d"), HexColor("#dcfce7")),   # green fg/bg
    2: (HexColor("#b45309"), HexColor("#fef3c7")),   # amber fg/bg
    3: (HexColor("#b91c1c"), HexColor("#fee2e2")),   # red fg/bg
}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

# Fallback/internal-status phrases that should NEVER reach the user.
# If your backend appends something like "(AI guidance service unavailable —
# showing standard guidance.)" to the summary string, this strips it here as
# a safety net. Fix the root cause in the backend too — see chat.
_LEAK_PATTERNS = [
    re.compile(r"\(\s*AI guidance service unavailable[^)]*\)", re.IGNORECASE),
    re.compile(r"\(\s*showing standard guidance\.?\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*fallback[^)]*\)", re.IGNORECASE),
]


def _sanitize(text: str) -> str:
    """Remove internal/fallback status phrases that shouldn't be user-facing."""
    if not text:
        return text
    cleaned = text
    for pat in _LEAK_PATTERNS:
        cleaned = pat.sub("", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# ------------------------------------------------------------- flowables --
class Badge(Flowable):
    """Rounded-rect pill badge, e.g. 'Tier 2 — Moderate Risk'."""

    def __init__(self, text, fg, bg, font_size=11, pad_x=10, pad_y=5):
        super().__init__()
        self.text = text
        self.fg = fg
        self.bg = bg
        self.font_size = font_size
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.font_name = "Helvetica-Bold"
        c = pdfcanvas.Canvas(io.BytesIO())
        self.text_w = c.stringWidth(text, self.font_name, font_size)
        self.width = self.text_w + 2 * pad_x
        self.height = font_size + 2 * pad_y

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width, self.height, self.height / 2, stroke=0, fill=1)
        c.setFillColor(self.fg)
        c.setFont(self.font_name, self.font_size)
        c.drawCentredString(self.width / 2, self.pad_y + 1.5, self.text)
        c.restoreState()


class HRule(Flowable):
    """Thin horizontal divider."""

    def __init__(self, width, color=LINE, thickness=0.75):
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.color = color
        self.height = thickness

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


# ------------------------------------------------------------------- doc --
def _header_footer(canvas_obj, doc, meta):
    canvas_obj.saveState()

    # --- header band ---
    band_h = 26 * mm
    canvas_obj.setFillColor(BRAND)
    canvas_obj.rect(0, PAGE_H - band_h, PAGE_W, band_h, stroke=0, fill=1)

    canvas_obj.setFillColor(white)
    canvas_obj.setFont("Helvetica-Bold", 17)
    canvas_obj.drawString(MARGIN, PAGE_H - 14 * mm, meta["title"])

    canvas_obj.setFont("Helvetica", 9.5)
    canvas_obj.setFillColor(HexColor("#e9d8fd"))
    canvas_obj.drawString(MARGIN, PAGE_H - 20 * mm, meta["subtitle"])

    canvas_obj.setFont("Helvetica", 8.5)
    canvas_obj.drawRightString(PAGE_W - MARGIN, PAGE_H - 14 * mm, meta["generated"])
    canvas_obj.drawRightString(PAGE_W - MARGIN, PAGE_H - 20 * mm, f"Report ID: {meta['report_id']}")

    # --- footer ---
    canvas_obj.setStrokeColor(LINE)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)

    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(MUTED)
    canvas_obj.drawString(MARGIN, 10 * mm, meta["footer_left"])
    canvas_obj.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")

    canvas_obj.restoreState()


def _styles():
    ss = getSampleStyleSheet()
    styles = {}
    styles["section"] = ParagraphStyle(
        "Section", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=12.5, textColor=BRAND, spaceBefore=0, spaceAfter=6,
        leading=15,
    )
    styles["body"] = ParagraphStyle(
        "Body", parent=ss["BodyText"], fontName="Helvetica",
        fontSize=10, textColor=INK, leading=14.5,
    )
    styles["muted"] = ParagraphStyle(
        "Muted", parent=ss["BodyText"], fontName="Helvetica",
        fontSize=9, textColor=MUTED, leading=13,
    )
    styles["bullet"] = ParagraphStyle(
        "Bullet", parent=ss["BodyText"], fontName="Helvetica",
        fontSize=10, textColor=INK, leading=14, spaceAfter=3,
    )
    styles["disclaimer"] = ParagraphStyle(
        "Disclaimer", parent=ss["BodyText"], fontName="Helvetica-Oblique",
        fontSize=8, textColor=MUTED, leading=11.5,
    )
    styles["cardlabel"] = ParagraphStyle(
        "CardLabel", parent=ss["BodyText"], fontName="Helvetica-Bold",
        fontSize=9, textColor=MUTED, leading=11,
    )
    styles["stat"] = ParagraphStyle(
        "Stat", parent=ss["BodyText"], fontName="Helvetica-Bold",
        fontSize=18, textColor=BRAND, leading=20,
    )
    return styles


def _section_card(title, body_flowables, styles, content_width):
    """A titled card: accent rule + heading, then content, wrapped so it
    doesn't split awkwardly across a page break."""
    inner = [Paragraph(title, styles["section"]), HRule(content_width, color=BRAND_LINE, thickness=1.2),
             Spacer(1, 6)]
    inner.extend(body_flowables)
    inner.append(Spacer(1, 14))
    return KeepTogether(inner)


# Canonical grouping for the intake form fields. Order here is the order
# they render in, regardless of what order the input dict arrives in — never
# trust upstream key order (it can get alphabetized by JSON.stringify,
# sort_keys, etc. before it reaches this function).
_DETAIL_GROUPS = [
    ("Basic Measurements", ["Age", "Height (cm)", "Weight (kg)", "BMI (auto)"]),
    ("Vital Signs", ["Systolic BP", "Diastolic BP", "Pulse rate"]),
    ("Cycle & Lifestyle", ["Period cycle", "Avg. gap between periods (days)",
                            "Weight change >5kg recently?"]),
    ("Clinical History", ["Marital status", "Ultrasound: PCOD detected?"]),
]


def _detail_groups_flowables(user_details, styles, content_width):
    """Build a grouped grid of label/value 'chips' instead of one flat table.
    Fields not matched to a known group land in a trailing 'Additional
    Details' group so nothing silently disappears if the form grows."""
    if isinstance(user_details, dict):
        remaining = dict(user_details)
    else:
        remaining = dict(user_details)  # list of pairs -> dict, order kept (py3.7+)

    group_style = ParagraphStyle(
        "GroupLabel", parent=styles["cardlabel"], fontName="Helvetica-Bold",
        fontSize=8.5, textColor=BRAND, spaceAfter=4, leading=11,
    )

    flowables = []
    cols = 3
    col_w = content_width / cols

    def chip(label, value):
        txt = (f'<font size="7" color="#6b7280">{label.upper()}</font><br/>'
               f'<font size="10.5" color="#1f2937"><b>{value}</b></font>')
        return Paragraph(txt, styles["body"])

    def render_group(title, pairs):
        flowables.append(Paragraph(title.upper(), group_style))
        rows = []
        row = []
        for label, value in pairs:
            row.append(chip(label, value))
            if len(row) == cols:
                rows.append(row)
                row = []
        if row:
            while len(row) < cols:
                row.append("")
            rows.append(row)
        t = Table(rows, colWidths=[col_w] * cols, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.6, LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        flowables.append(t)
        flowables.append(Spacer(1, 10))

    for title, fields in _DETAIL_GROUPS:
        pairs = [(f, remaining.pop(f)) for f in fields if f in remaining]
        if pairs:
            render_group(title, pairs)

    if remaining:
        render_group("Additional Details", list(remaining.items()))

    if flowables:
        flowables.pop()  # drop trailing spacer
    return flowables


def build_pdf(result: dict) -> bytes:
    buf = io.BytesIO()
    content_width = PAGE_W - 2 * MARGIN

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=32 * mm, bottomMargin=20 * mm,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title="PCOS Screening Report",
    )
    styles = _styles()

    tier = result.get("tier", 2)
    fg, bg = TIER_COLORS.get(tier, TIER_COLORS[2])
    label = result.get("label", "")
    probability = result.get("probability", 0)
    advice = result.get("advice", {})

    meta = {
        "title": "PCOS Screening Report",
        "subtitle": "AI-assisted risk screening summary",
        "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "report_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "footer_left": "Confidential — for personal health reference only",
    }

    story = []

    # ---- result summary block (badge + probability stat) --------------
    badge = Badge(f"Tier {tier} — {label}" if label else f"Tier {tier}", fg, bg, font_size=12)
    stat_tbl = Table(
        [[badge, Paragraph(f"Estimated risk probability&nbsp;&nbsp;<font color='#5b21b6'><b>{probability:.0%}</b></font>",
                            styles["muted"])]],
        colWidths=[badge.width + 4, content_width - badge.width - 4],
    )
    stat_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(stat_tbl)
    story.append(HRule(content_width, color=LINE, thickness=0.75))
    story.append(Spacer(1, 12))

    # ---- summary card ----------------------------------------------------
    if advice.get("summary"):
        story.append(_section_card(
            "Summary", [Paragraph(_sanitize(advice["summary"]), styles["body"])], styles, content_width
        ))

    # ---- Medical Profile (grouped, not a flat alphabetized dump) ----
    user_details = result.get("user_details")
    if user_details:
        story.append(_section_card(
            "Medical Profile", _detail_groups_flowables(user_details, styles, content_width),
            styles, content_width,
        ))

    # ---- top contributing factors table ----------------------------------
    top_factors = result.get("top_factors", [])
    if top_factors:
        rows = [["FACTOR", "EFFECT"]]
        row_colors = []
        for f in top_factors:
            raises = f.get("direction") == "increases_risk"
            rows.append([f.get("label", f.get("feature", "")), "▲ Raises risk" if raises else "▼ Lowers risk"])

        t = Table(rows, colWidths=[content_width * 0.68, content_width * 0.32], hAlign="LEFT")
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), BRAND),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9.5),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, BRAND),
            ("LINEBELOW", (0, 1), (-1, -2), 0.5, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), CARD_BG))
        t.setStyle(TableStyle(style_cmds))
        story.append(_section_card("Top Contributing Factors", [t], styles, content_width))

    # ---- bullet sections ---------------------------------------------
    def bullets(items):
        return ListFlowable(
            [ListItem(Paragraph(str(i), styles["bullet"]), bulletColor=BRAND, value="circle")
             for i in items],
            bulletType="bullet", leftIndent=14, bulletFontSize=6,
        )

    def bullet_section(title, items):
        if not items:
            return
        story.append(_section_card(title, [bullets(items)], styles, content_width))

    bullet_section("Lifestyle Suggestions", advice.get("lifestyle"))
    bullet_section("Ayurvedic / Natural Support (supportive only)", advice.get("ayurvedic"))
    bullet_section("Mental Wellbeing", advice.get("mental_wellbeing"))
    bullet_section("Avoid", advice.get("avoid"))
    bullet_section("Suggested Lab Tests", result.get("lab_suggestions"))

    if advice.get("doctor_action"):
        story.append(_section_card(
            "Recommended Doctor Action",
            [Paragraph(_sanitize(advice["doctor_action"]), styles["body"])],
            styles, content_width,
        ))

    # ---- disclaimer box -------------------------------------------------
    disc_tbl = Table(
        [[Paragraph(
            "<b>Disclaimer:</b> This is a screening tool based on self-reported information, "
            "not a medical diagnosis. It does not replace professional medical evaluation. "
            "Please consult a qualified gynecologist for any diagnosis or treatment decisions.",
            styles["disclaimer"],
        )]],
        colWidths=[content_width],
    )
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.75, BRAND_LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(Spacer(1, 4))
    story.append(disc_tbl)

    doc.build(
        story,
        onFirstPage=lambda c, d: _header_footer(c, d, meta),
        onLaterPages=lambda c, d: _header_footer(c, d, meta),
    )
    return buf.getvalue()


# ---------------------------------------------------------------- demo ----
if __name__ == "__main__":
    demo_result = {
        "tier": 2,
        "label": "Moderate Risk",
        "probability": 0.62,
        "user_details": {
            "Age": "27",
            "Height (cm)": "160",
            "Weight (kg)": "62",
            "BMI (auto)": "24.2",
            "Systolic BP": "118",
            "Diastolic BP": "78",
            "Pulse rate": "76",
            "Marital status": "Unmarried",
            "Ultrasound: PCOD detected?": "Not done",
            "Period cycle": "Regular",
            "Avg. gap between periods (days)": "29",
            "Weight change >5kg recently?": "No",
        },
        "advice": {
            "summary": (
                "Based on your responses, several indicators associated with PCOS risk were "
                "identified. This is not a diagnosis — it's a prompt to get evaluated by a "
                "qualified professional and to start monitoring the factors below. "
                "(AI guidance service unavailable — showing standard guidance.)"
            ),
            "lifestyle": [
                "Aim for 150 minutes/week of moderate exercise (brisk walking, cycling, swimming).",
                "Prioritize a low-glycemic diet: whole grains, legumes, vegetables over refined sugar.",
                "Maintain a consistent sleep schedule of 7–8 hours per night.",
            ],
            "ayurvedic": [
                "Cinnamon and fenugreek are traditionally used to support insulin sensitivity.",
                "Consult an Ayurvedic practitioner before starting any herbal regimen.",
            ],
            "mental_wellbeing": [
                "PCOS is linked with higher anxiety/depression risk — don't dismiss mood changes.",
                "Consider journaling or a brief daily mindfulness practice to track stress.",
            ],
            "avoid": [
                "Crash diets or extreme calorie restriction.",
                "Self-medicating with hormonal supplements without medical guidance.",
            ],
            "doctor_action": (
                "Schedule a consultation with a gynecologist or endocrinologist within the next "
                "4–6 weeks. Bring this report and a record of your menstrual cycle for the past "
                "3 months."
            ),
        },
        "top_factors": [
            {"label": "Irregular menstrual cycle", "direction": "increases_risk"},
            {"label": "Elevated BMI", "direction": "increases_risk"},
            {"label": "Regular physical activity", "direction": "decreases_risk"},
            {"label": "Family history of PCOS", "direction": "increases_risk"},
        ],
        "lab_suggestions": [
            "Fasting insulin and glucose",
            "LH/FSH ratio",
            "Total & free testosterone",
            "Pelvic ultrasound",
        ],
    }
    pdf_bytes = build_pdf(demo_result)
    with open("/home/claude/demo_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("wrote", len(pdf_bytes), "bytes")