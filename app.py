from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader
import io
from PIL import Image
from datetime import date

# ── Colour palette ───────────────────────────────────────────────────────────
C_PRIMARY    = colors.HexColor("#1A3557")   # deep navy
C_ACCENT     = colors.HexColor("#2E86C1")   # medium blue
C_LIGHT      = colors.HexColor("#D6EAF8")   # pale blue
C_SUCCESS    = colors.HexColor("#1E8449")   # green
C_WARNING    = colors.HexColor("#B7950B")   # amber
C_DANGER     = colors.HexColor("#922B21")   # red
C_MUTED      = colors.HexColor("#707B7C")   # grey
C_WHITE      = colors.white
C_BG         = colors.HexColor("#F4F6F7")   # near-white background

action_color = {
    "No_Treatment": C_WARNING,
    "Medication":   C_ACCENT,
    "Surgery":      C_SUCCESS,
}

def _draw_footer(c, page_num, total_pages, width, margin):
    c.setStrokeColor(C_MUTED)
    c.setLineWidth(0.5)
    c.line(margin, 40, width - margin, 40)
    c.setFont("Helvetica", 8)
    c.setFillColor(C_MUTED)
    c.drawString(margin, 28, "Healthcare Decision Support System  |  Confidential – Academic Simulation Only")
    c.drawRightString(width - margin, 28, f"Page {page_num}")

def _section_header(c, text, y, width, margin):
    """Bold section title with a colored rule."""
    c.setFillColor(C_PRIMARY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, text.upper())
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1.5)
    c.line(margin, y - 4, width - margin, y - 4)
    return y - 22

def _badge(c, text, x, y, color):
    """Colored pill-shaped badge."""
    w = c.stringWidth(text, "Helvetica-Bold", 9) + 12
    c.setFillColor(color)
    c.roundRect(x, y - 2, w, 14, 4, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 6, y + 2, text)
    return x + w + 6

def generate_pdf_report(profile, optimal_policy, V_star, diagram_path):
    buffer    = io.BytesIO()
    c         = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin    = 55
    page_num  = 1

    # ── PAGE 1 : COVER ───────────────────────────────────────────────────────
    # Top navy banner
    c.setFillColor(C_PRIMARY)
    c.rect(0, height - 110, width, 110, fill=1, stroke=0)

    # Title block
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin, height - 52, "Healthcare Decision Support System")
    c.setFont("Helvetica", 13)
    c.drawString(margin, height - 74, "Optimal Treatment Plan  ·  MDP Policy Report")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#AED6F1"))
    c.drawRightString(width - margin, height - 88, f"Generated: {date.today().strftime('%B %d, %Y')}")

    # Accent stripe
    c.setFillColor(C_ACCENT)
    c.rect(0, height - 116, width, 6, fill=1, stroke=0)

    y = height - 150

    # ── Patient Profile Card ─────────────────────────────────────────────────
    card_h = 130
    c.setFillColor(C_BG)
    c.roundRect(margin, y - card_h, width - 2*margin, card_h, 8, fill=1, stroke=0)
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1)
    c.roundRect(margin, y - card_h, width - 2*margin, card_h, 8, fill=0, stroke=1)

    # Left accent bar
    c.setFillColor(C_ACCENT)
    c.rect(margin, y - card_h, 5, card_h, fill=1, stroke=0)

    cy = y - 22
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(C_PRIMARY)
    c.drawString(margin + 18, cy, "PATIENT PROFILE")
    cy -= 20

    fields = [
        ("Current Health State", profile.health),
        ("Age Group",            profile.age),
        ("Comorbidity Level",    profile.comorbidity),
        ("Recommended Action",   optimal_policy[profile.health].replace("_", " ")),
        ("Expected Value (V*)",  f"{V_star[profile.health]:.2f}"),
    ]
    col_x = [margin + 18, margin + 200]
    for label, value in fields:
        c.setFont("Helvetica", 9)
        c.setFillColor(C_MUTED)
        c.drawString(col_x[0], cy, label)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(C_PRIMARY)
        c.drawString(col_x[1], cy, value)
        cy -= 16

    y -= card_h + 25

    # ── Disclaimer box ───────────────────────────────────────────────────────
    c.setFillColor(colors.HexColor("#FDFEFE"))
    c.setStrokeColor(C_WARNING)
    c.setLineWidth(1)
    c.roundRect(margin, y - 38, width - 2*margin, 38, 5, fill=1, stroke=1)
    c.setFillColor(C_WARNING)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 10, y - 14, "⚠  DISCLAIMER")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.black)
    c.drawString(margin + 10, y - 28,
        "This report is an academic simulation only. It does NOT constitute medical advice and must not be used for clinical decision-making.")
    y -= 60

    # ── Purpose ─────────────────────────────────────────────────────────────
    y = _section_header(c, "1.  Purpose & Methodology", y, width, margin)
    body = (
        "This report presents the output of a Markov Decision Process (MDP) optimisation engine "
        "applied to patient treatment planning under uncertainty. The system models health state "
        "transitions, treatment costs, and clinical risk to derive a personalised, long-term optimal "
        "treatment policy via Policy Iteration with a discount factor γ = 0.90.\n"
        "Transition probabilities are adjusted dynamically for the patient's age group and comorbidity "
        "level, ensuring recommendations reflect individual clinical context."
    )
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    tw = width - 2 * margin
    # simple word-wrap
    words = body.replace("\n", " \n ").split(" ")
    line, lines = "", []
    for w in words:
        if w == "\n":
            lines.append(line); line = ""; continue
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 10) < tw:
            line = test
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    for l in lines:
        c.drawString(margin, y, l); y -= 14
    y -= 10

    # ── States & Actions Table ───────────────────────────────────────────────
    y = _section_header(c, "2.  Model Components", y, width, margin)

    table_data = [
        ["Health States", ", ".join(health_states)],
        ["Actions Available", ", ".join(a.replace("_"," ") for a in actions)],
        ["Terminal States", "Recovered, Deceased"],
        ["Discount Factor (γ)", "0.90"],
        ["Algorithm", "Policy Iteration (convergence δ < 1×10⁻⁶)"],
    ]
    tbl = Table(table_data, colWidths=[160, tw - 160])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  C_LIGHT),
        ("BACKGROUND",  (0,0),(0,-1),  C_LIGHT),
        ("TEXTCOLOR",   (0,0),(0,-1),  C_PRIMARY),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (1,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_WHITE, C_BG]),
        ("GRID",        (0,0),(-1,-1), 0.4, C_MUTED),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    tbl_w, tbl_h = tbl.wrapOn(c, tw, 300)
    tbl.drawOn(c, margin, y - tbl_h)
    y -= tbl_h + 15

    _draw_footer(c, page_num, "—", width, margin)
    c.showPage(); page_num += 1

    # ── PAGE 2 : OPTIMAL POLICY TABLE ────────────────────────────────────────
    # Header stripe
    c.setFillColor(C_PRIMARY)
    c.rect(0, height - 52, width, 52, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, height - 30, "Optimal Treatment Policy")
    c.setFont("Helvetica", 10)
    c.drawString(margin, height - 46, "Per-state recommended actions and expected long-term values")
    c.setFillColor(C_ACCENT)
    c.rect(0, height - 58, width, 6, fill=1, stroke=0)

    y = height - 80
    y = _section_header(c, "3.  Per-State Policy & Value Function", y, width, margin)

    # Build policy table
    pol_data = [["Health State", "Recommended Action", "Expected Value (V*)", "Action Badge"]]
    for state in health_states:
        act = optimal_policy[state]
        pol_data.append([state, act.replace("_"," "), f"{V_star[state]:.2f}", act])

    pol_tbl = Table(pol_data[:-1+1], colWidths=[120, 160, 130, 0.01])  # last col hidden, drawn manually
    # Remove last col – we'll draw badges manually
    pol_data_display = [[r[0], r[1], r[2]] for r in pol_data]
    pol_tbl2 = Table(pol_data_display, colWidths=[140, 180, 120])
    pol_tbl2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  C_PRIMARY),
        ("TEXTCOLOR",   (0,0),(-1,0),  C_WHITE),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_BG]),
        ("GRID",        (0,0),(-1,-1), 0.4, colors.HexColor("#BDC3C7")),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING", (0,0),(-1,-1), 10),
        ("ALIGN",       (2,0),(-1,-1), "CENTER"),
    ]))
    # Color rows by action
    action_row_colors = {
        "No_Treatment": colors.HexColor("#FEF9E7"),
        "Medication":   colors.HexColor("#EBF5FB"),
        "Surgery":      colors.HexColor("#EAFAF1"),
    }
    for i, state in enumerate(health_states, 1):
        act = optimal_policy[state]
        bg = action_row_colors.get(act, C_WHITE)
        pol_tbl2.setStyle(TableStyle([("BACKGROUND",(0,i),(-1,i), bg)]))
        # Value cell: color based on V*
        v = V_star[state]
        vc = C_SUCCESS if v > 5 else (C_WARNING if v > 0 else C_DANGER)
        pol_tbl2.setStyle(TableStyle([("TEXTCOLOR",(2,i),(2,i), vc),
                                       ("FONTNAME", (2,i),(2,i),"Helvetica-Bold")]))

    pw, ph = pol_tbl2.wrapOn(c, tw, 500)
    pol_tbl2.drawOn(c, margin, y - ph)
    y -= ph + 20

    # Legend
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(C_PRIMARY)
    c.drawString(margin, y, "Action Legend:")
    lx = margin + 105
    for act, col in [("No Treatment", C_WARNING), ("Medication", C_ACCENT), ("Surgery", C_SUCCESS)]:
        lx = _badge(c, act, lx, y - 1, col)
    y -= 25

    # ── Value Bar Chart ──────────────────────────────────────────────────────
    y -= 10
    y = _section_header(c, "4.  Expected Value Distribution (V*)", y, width, margin)

    bar_area_w = width - 2*margin - 80
    bar_h      = 16
    bar_gap    = 8
    max_v      = max(abs(v) for v in V_star.values()) or 1
    zero_x     = margin + 80 + (bar_area_w * 0.5)  # centre = 0

    for state in health_states:
        v = V_star[state]
        bar_w = abs(v) / max_v * (bar_area_w * 0.48)
        col   = C_SUCCESS if v > 5 else (C_WARNING if v > 0 else C_DANGER)

        # State label
        c.setFont("Helvetica", 9)
        c.setFillColor(C_PRIMARY)
        c.drawRightString(margin + 75, y + 4, state)

        # Bar
        bx = zero_x if v >= 0 else zero_x - bar_w
        c.setFillColor(col)
        c.roundRect(bx, y, bar_w if bar_w > 2 else 2, bar_h, 2, fill=1, stroke=0)

        # Value label
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(C_PRIMARY)
        lbl = f"{v:.1f}"
        if v >= 0:
            c.drawString(bx + bar_w + 4, y + 4, lbl)
        else:
            c.drawRightString(bx - 4, y + 4, lbl)

        y -= bar_h + bar_gap

    # Zero line
    c.setStrokeColor(C_MUTED)
    c.setLineWidth(0.8)
    c.line(zero_x, y, zero_x, y + (bar_h + bar_gap) * len(health_states))

    _draw_footer(c, page_num, "—", width, margin)
    c.showPage(); page_num += 1

    # ── PAGE 3 : MDP DIAGRAM ─────────────────────────────────────────────────
    if diagram_path:
        c.setFillColor(C_PRIMARY)
        c.rect(0, height - 52, width, 52, fill=1, stroke=0)
        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(margin, height - 30, "MDP State Transition Diagram")
        c.setFont("Helvetica", 10)
        c.drawString(margin, height - 46, "Patient-specific transition probabilities after age & comorbidity adjustment")
        c.setFillColor(C_ACCENT)
        c.rect(0, height - 58, width, 6, fill=1, stroke=0)

        try:
            img    = Image.open(diagram_path)
            dw     = width - 2*margin
            dh     = dw * (img.size[1] / img.size[0])
            img_y  = max(margin + 30, (height - 58 - dh) / 2)
            c.drawImage(diagram_path, margin, img_y, width=dw, height=dh, preserveAspectRatio=True)
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.setFillColor(C_DANGER)
            c.drawString(margin, height//2, f"Diagram could not be rendered: {e}")

        _draw_footer(c, page_num, "—", width, margin)
        c.showPage(); page_num += 1

    # ── PAGE 4 : NOTES & APPENDIX ────────────────────────────────────────────
    c.setFillColor(C_PRIMARY)
    c.rect(0, height - 52, width, 52, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, height - 30, "Appendix & Technical Notes")
    c.setFillColor(C_ACCENT)
    c.rect(0, height - 58, width, 6, fill=1, stroke=0)

    y = height - 85
    y = _section_header(c, "5.  Reward Function", y, width, margin)
    reward_rows = [
        ["Component", "Description", "Values"],
        ["Health Reward",    "Base reward per state",             "Healthy:+8, Mild:+5, Moderate:+2, Severe:−3, Critical:−8, Recovered:+15, Deceased:−20"],
        ["Treatment Cost",   "Cost subtracted per action",        "No Treatment: 0  |  Medication: −2  |  Surgery: −6"],
        ["Risk Penalty",     "Risk charge per action",            "No Treatment: −1  |  Medication: −2  |  Surgery: −4"],
        ["Surgery Risk Adj.","Extra penalty for age/comorbidity", "Elderly: −3  |  Severe Comorb: −3  |  Adult: −1  |  Moderate Comorb: −1"],
    ]
    r_tbl = Table(reward_rows, colWidths=[110, 155, tw-265])
    r_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  C_PRIMARY),
        ("TEXTCOLOR",   (0,0),(-1,0),  C_WHITE),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_BG]),
        ("GRID",        (0,0),(-1,-1), 0.4, C_MUTED),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    rw, rh = r_tbl.wrapOn(c, tw, 300)
    r_tbl.drawOn(c, margin, y - rh)
    y -= rh + 25

    y = _section_header(c, "6.  Patient-Specific Adjustments", y, width, margin)
    adj_rows = [
        ["Factor", "Level", "Deterioration Penalty Applied"],
        ["Age", "Young",   "None (baseline)"],
        ["Age", "Adult",   "+0.05 shift toward worse state"],
        ["Age", "Elderly", "+0.12 shift toward worse state"],
        ["Comorbidity", "None",     "None (baseline)"],
        ["Comorbidity", "Moderate", "+0.06 shift toward worse state"],
        ["Comorbidity", "Severe",   "+0.14 shift toward worse state"],
    ]
    a_tbl = Table(adj_rows, colWidths=[100, 100, tw-200])
    a_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  C_ACCENT),
        ("TEXTCOLOR",   (0,0),(-1,0),  C_WHITE),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_BG]),
        ("GRID",        (0,0),(-1,-1), 0.4, C_MUTED),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    aw, ah = a_tbl.wrapOn(c, tw, 300)
    a_tbl.drawOn(c, margin, y - ah)
    y -= ah + 25

    # Final disclaimer
    c.setFillColor(C_DANGER)
    c.roundRect(margin, y - 48, width - 2*margin, 48, 5, fill=0, stroke=1)
    c.setFillColor(C_DANGER)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 10, y - 16, "IMPORTANT NOTICE")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.black)
    c.drawString(margin + 10, y - 30, "This document is produced solely for academic and educational purposes. All results are simulated")
    c.drawString(margin + 10, y - 42, "and do not represent real medical data or clinical recommendations. Do not use for patient care.")

    _draw_footer(c, page_num, "—", width, margin)
    c.save()
    buffer.seek(0)
    return buffer
