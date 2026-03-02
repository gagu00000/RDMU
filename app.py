import streamlit as st
import random
from dataclasses import dataclass
from graphviz import Digraph
from PIL import Image
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from datetime import date

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
health_states     = ['Healthy', 'Mild', 'Moderate', 'Severe', 'Critical', 'Recovered', 'Deceased']
non_terminal_states = health_states[:-2]
terminal_states   = ['Recovered', 'Deceased']
actions           = ['No_Treatment', 'Medication', 'Surgery']
age_groups        = ['Young', 'Adult', 'Elderly']
comorbidities     = ['None', 'Moderate', 'Severe']

# Fix #14: γ raised to 0.95; terminal values computed as one-time reward (no self-loop inflation)
gamma         = 0.95
MAX_EVAL_ITERS = 2000

# State desirability index used for penalty targeting (Fix #15)
STATE_VALUE = {
    'Healthy': 6, 'Mild': 5, 'Moderate': 4, 'Severe': 3,
    'Critical': 2, 'Recovered': 7, 'Deceased': 0
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Colour Palette (PDF)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C_PRIMARY = colors.HexColor("#1A3557")
C_ACCENT  = colors.HexColor("#2E86C1")
C_LIGHT   = colors.HexColor("#D6EAF8")
C_SUCCESS = colors.HexColor("#1E8449")
C_WARNING = colors.HexColor("#B7950B")
C_DANGER  = colors.HexColor("#922B21")
C_MUTED   = colors.HexColor("#707B7C")
C_WHITE   = colors.white
C_BG      = colors.HexColor("#F4F6F7")

action_row_colors = {
    "No_Treatment": colors.HexColor("#FEF9E7"),
    "Medication":   colors.HexColor("#EBF5FB"),
    "Surgery":      colors.HexColor("#EAFAF1"),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Patient Profile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class PatientProfile:
    health: str
    age: str
    comorbidity: str

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Base Transition Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P_base = {
    'Healthy': {
        'No_Treatment': {'Healthy': 0.8,  'Mild': 0.2},
        'Medication':   {'Healthy': 0.9,  'Mild': 0.1},
        'Surgery':      {'Healthy': 0.85, 'Mild': 0.1, 'Critical': 0.05},
    },
    'Mild': {
        'No_Treatment': {'Mild': 0.6,    'Moderate': 0.3, 'Healthy': 0.1},
        'Medication':   {'Healthy': 0.6, 'Mild': 0.3,    'Moderate': 0.1},
        'Surgery':      {'Healthy': 0.7, 'Mild': 0.2,    'Critical': 0.1},
    },
    'Moderate': {
        'No_Treatment': {'Moderate': 0.5, 'Severe': 0.4,   'Mild': 0.1},
        'Medication':   {'Mild': 0.5,     'Moderate': 0.3, 'Severe': 0.2},
        'Surgery':      {'Healthy': 0.5,  'Moderate': 0.3, 'Critical': 0.2},
    },
    'Severe': {
        'No_Treatment': {'Severe': 0.4,    'Critical': 0.6},
        'Medication':   {'Moderate': 0.4,  'Severe': 0.4,  'Critical': 0.2},
        'Surgery':      {'Moderate': 0.4,  'Severe': 0.3,  'Critical': 0.3},
    },
    'Critical': {
        'No_Treatment': {'Critical': 0.7,  'Deceased': 0.3},
        'Medication':   {'Critical': 0.5,  'Severe': 0.3,    'Deceased': 0.2},
        'Surgery':      {'Severe': 0.4,    'Recovered': 0.4, 'Deceased': 0.2},
    },
}
# Terminal states self-loop (absorbing)
for t in terminal_states:
    P_base[t] = {a: {t: 1.0} for a in actions}


def _normalize(dist: dict) -> dict:
    total = sum(dist.values())
    if total <= 0:
        raise ValueError(f"Zero-sum distribution — check transition logic: {dist}")
    return {k: v / total for k, v in dist.items()}


def build_transitions(profile: PatientProfile) -> dict:
    """
    Personalise transition probabilities for age and comorbidity.
    Penalty shifts probability away from the BEST outcome state
    (Fix #15: using STATE_VALUE so Recovered is correctly penalised too).
    """
    age_penalty    = {'Young': 0.0, 'Adult': 0.05, 'Elderly': 0.12}
    comorb_penalty = {'None':  0.0, 'Moderate': 0.06, 'Severe': 0.14}
    total_penalty  = age_penalty[profile.age] + comorb_penalty[profile.comorbidity]

    worse_of = {
        'Healthy': 'Mild', 'Mild': 'Moderate', 'Moderate': 'Severe',
        'Severe': 'Critical', 'Critical': 'Deceased'
    }

    P_patient = {}
    for state in health_states:
        P_patient[state] = {}
        for action in actions:
            dist = dict(P_base[state][action])

            if state in terminal_states or total_penalty == 0:
                P_patient[state][action] = dist
                continue

            # Fix #15: pick best outcome by STATE_VALUE (includes Recovered)
            best = max(dist.keys(), key=lambda s: STATE_VALUE[s])
            worse = worse_of.get(state)

            if worse and best:
                shift = min(total_penalty, dist.get(best, 0) * 0.8)
                dist[best]  = dist.get(best, 0) - shift
                # If the worse state isn't already in dist, add it
                dist[worse] = dist.get(worse, 0) + shift

            dist = {k: max(v, 0) for k, v in dist.items() if max(v, 0) > 0}
            P_patient[state][action] = _normalize(dist)

    return P_patient


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reward Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
health_reward = {
    'Healthy': 8, 'Mild': 5, 'Moderate': 2,
    'Severe': -3, 'Critical': -8,
    'Recovered': 15, 'Deceased': -20
}
# Fix #13: No_Treatment penalty removed (progression risk already in transitions)
treatment_cost = {'No_Treatment': 0, 'Medication': -2, 'Surgery': -6}
risk_penalty   = {'No_Treatment': 0, 'Medication': -2, 'Surgery': -4}

surgery_risk = {
    'age':        {'Young': 0, 'Adult': -1, 'Elderly': -3},
    'comorbidity':{'None': 0,  'Moderate': -1, 'Severe': -3},
}


def reward(state: str, action: str, profile: PatientProfile) -> float:
    # Fix #12: terminal states return health reward only — no treatment costs
    if state in terminal_states:
        return health_reward[state]
    base = health_reward[state] + treatment_cost[action] + risk_penalty[action]
    if action == 'Surgery':
        base += surgery_risk['age'][profile.age]
        base += surgery_risk['comorbidity'][profile.comorbidity]
    return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Policy Evaluation & Iteration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def policy_evaluation(policy: dict, P: dict, profile: PatientProfile) -> dict:
    V = {s: 0.0 for s in health_states}

    # Fix #2: Seed terminal state values analytically (no self-loop drift)
    for t in terminal_states:
        V[t] = reward(t, policy[t], profile) / (1 - gamma)

    for _ in range(MAX_EVAL_ITERS):
        delta = 0.0
        for s in health_states:
            if s in terminal_states:
                continue  # Fix #2: skip terminal — already set analytically
            a = policy[s]
            v = V[s]
            V[s] = sum(
                p * (reward(s, a, profile) + gamma * V[s2])
                for s2, p in P[s][a].items()
            )
            delta = max(delta, abs(v - V[s]))
        if delta < 1e-8:
            break
    return V


def policy_iteration(profile: PatientProfile):
    P = build_transitions(profile)

    # Fix #1: deterministic initial policy
    policy = {s: 'No_Treatment' for s in health_states}

    for _ in range(500):
        V = policy_evaluation(policy, P, profile)
        stable = True
        for s in health_states:
            if s in terminal_states:
                continue
            old = policy[s]
            # Fix #3: renamed lambda param to `act` to avoid shadowing
            policy[s] = max(
                actions,
                key=lambda act: sum(
                    p * (reward(s, act, profile) + gamma * V[s2])
                    for s2, p in P[s][act].items()
                )
            )
            if old != policy[s]:
                stable = False
        if stable:
            break
    return policy, V, P


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def simulate(start_state: str, policy: dict, P: dict, steps: int = 5, seed: int = 42):
    random.seed(seed)
    state, history = start_state, []
    for _ in range(steps):
        if state in terminal_states:
            break
        action = policy[state]
        next_state = random.choices(
            list(P[state][action].keys()),
            list(P[state][action].values())
        )[0]
        history.append((state, action, next_state))
        state = next_state
    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MDP Diagram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_mdp_diagram(P: dict) -> Digraph:
    dot = Digraph(comment='Healthcare MDP', format='png')
    dot.attr(
        engine='dot',
        rankdir='TB',
        bgcolor='#FFFFFF',
        pad='0.8',
        nodesep='0.8',
        ranksep='1.2',
        splines='ortho',
        fontname='Helvetica',
        dpi='180',
    )

    node_styles = {
        'Healthy':   {'fillcolor': '#E8F8F5', 'color': '#1E8449', 'fontcolor': '#1E8449'},
        'Mild':      {'fillcolor': '#FEF9E7', 'color': '#D4AC0D', 'fontcolor': '#9A7D0A'},
        'Moderate':  {'fillcolor': '#FEF0E6', 'color': '#E67E22', 'fontcolor': '#A04000'},
        'Severe':    {'fillcolor': '#FDEDEC', 'color': '#E74C3C', 'fontcolor': '#C0392B'},
        'Critical':  {'fillcolor': '#F9EBEA', 'color': '#C0392B', 'fontcolor': '#922B21'},
        'Recovered': {'fillcolor': '#EBF5FB', 'color': '#2E86C1', 'fontcolor': '#1A5276'},
        'Deceased':  {'fillcolor': '#F2F3F4', 'color': '#808B96', 'fontcolor': '#566573'},
    }

    rank_groups = [
        ['Healthy'],
        ['Mild'],
        ['Moderate'],
        ['Severe'],
        ['Critical'],
        ['Recovered', 'Deceased'],
    ]
    for group in rank_groups:
        with dot.subgraph() as s:
            s.attr(rank='same')
            for node in group:
                s.node(node)

    for state in health_states:
        st = node_styles[state]
        is_terminal = state in terminal_states
        dot.node(
            state,
            label=state,
            shape='doublecircle' if is_terminal else 'circle',
            style='filled',
            fillcolor=st['fillcolor'],
            color=st['color'],
            fontcolor=st['fontcolor'],
            fontname='Helvetica-Bold',
            fontsize='13',
            width='1.2',
            height='1.2',
            fixedsize='true',
            penwidth='2.5' if is_terminal else '2.0',
        )

    # Aggregate all actions into one edge per (s→s2)
    edge_map: dict = {}
    for s in health_states:
        for a in actions:
            for s2, p in P[s][a].items():
                if p > 0:
                    edge_map.setdefault((s, s2), []).append((a, p))

    action_abbr  = {'No_Treatment': 'NT', 'Medication': 'Med', 'Surgery': 'Sur'}
    action_color = {'No_Treatment': '#E67E22', 'Medication': '#2980B9', 'Surgery': '#27AE60'}

    for (s, s2), entries in edge_map.items():
        dominant = max(entries, key=lambda x: x[1])[0]
        edge_col = action_color[dominant]
        label    = "\n".join(
            f"{action_abbr[a]}: {p:.2f}"
            for a, p in sorted(entries, key=lambda x: x[1], reverse=True)
        )
        is_self = (s == s2)
        dot.edge(
            s, s2,
            label=f" {label} ",
            color=edge_col,
            fontcolor='#2C3E50',
            fontname='Helvetica',
            fontsize='8',
            penwidth='1.8' if not is_self else '1.2',
            arrowsize='0.6',
            style='solid' if not is_self else 'dashed',
            constraint='true' if not is_self else 'false',
        )

    with dot.subgraph(name='cluster_legend') as leg:
        leg.attr(
            label='  Edge Label Key  ',
            style='filled,rounded',
            fillcolor='#F8F9FA',
            color='#BDC3C7',
            fontname='Helvetica-Bold',
            fontsize='10',
            margin='12',
            rank='sink',
        )
        for key, label, fc in [
            ('l1', 'NT = No Treatment', '#E67E22'),
            ('l2', 'Med = Medication',  '#2980B9'),
            ('l3', 'Sur = Surgery',     '#27AE60'),
        ]:
            leg.node(key, label, shape='plaintext', fontcolor=fc,
                     fontname='Helvetica', fontsize='9')
        leg.edge('l1', 'l2', style='invis')
        leg.edge('l2', 'l3', style='invis')

    return dot


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PDF Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _new_page(c, width, height):
    c.setFillColor(C_WHITE)
    c.rect(0, 0, width, height, fill=1, stroke=0)


def _draw_footer(c, page_num, width, margin):
    c.setStrokeColor(C_MUTED)
    c.setLineWidth(0.5)
    c.line(margin, 40, width - margin, 40)
    c.setFont("Helvetica", 8)
    c.setFillColor(C_MUTED)
    c.drawString(margin, 28, "Healthcare Decision Support System  |  Confidential – Academic Simulation Only")
    c.drawRightString(width - margin, 28, f"Page {page_num}")
    c.setFillColor(colors.black)


def _section_header(c, text, y, width, margin):
    c.setFillColor(C_PRIMARY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, text.upper())
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1.5)
    c.line(margin, y - 4, width - margin, y - 4)
    c.setFillColor(colors.black)
    return y - 22


def _badge(c, text, x, y, color):
    w = c.stringWidth(text, "Helvetica-Bold", 9) + 12
    c.setFillColor(color)
    c.roundRect(x, y - 2, w, 14, 4, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 6, y + 2, text)
    c.setFillColor(colors.black)
    return x + w + 6


def _wrap_text(c, text, font, size, max_width):
    words = text.replace("\n", " \n ").split(" ")
    line, lines = "", []
    for w in words:
        if w == "\n":
            lines.append(line); line = ""; continue
        test = (line + " " + w).strip()
        if c.stringWidth(test, font, size) < max_width:
            line = test
        else:
            # Fix #7: hard-break single words that exceed max_width
            if not line and c.stringWidth(w, font, size) >= max_width:
                lines.append(w); line = ""
            else:
                lines.append(line); line = w
    if line:
        lines.append(line)
    return lines


def _page_header(c, title, subtitle, width, height, margin):
    c.setFillColor(C_PRIMARY)
    c.rect(0, height - 52, width, 52, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, height - 30, title)
    c.setFont("Helvetica", 10)
    c.drawString(margin, height - 46, subtitle)
    c.setFillColor(C_ACCENT)
    c.rect(0, height - 58, width, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PDF Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_pdf_report(profile: PatientProfile, optimal_policy: dict,
                        V_star: dict, diagram_path: str | None) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin   = 55
    tw       = width - 2 * margin
    page_num = 1

    # ── PAGE 1 : COVER ───────────────────────────────────────────────────────
    _new_page(c, width, height)

    c.setFillColor(C_PRIMARY)
    c.rect(0, height - 110, width, 110, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin, height - 52, "Healthcare Decision Support System")
    c.setFont("Helvetica", 13)
    c.drawString(margin, height - 74, "Optimal Treatment Plan  ·  MDP Policy Report")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#AED6F1"))
    c.drawRightString(width - margin, height - 88,
                      f"Generated: {date.today().strftime('%B %d, %Y')}")
    c.setFillColor(C_ACCENT)
    c.rect(0, height - 116, width, 6, fill=1, stroke=0)

    y = height - 150

    # Patient profile card
    card_h = 130
    c.setFillColor(C_BG)
    c.roundRect(margin, y - card_h, tw, card_h, 8, fill=1, stroke=0)
    c.setStrokeColor(C_ACCENT)
    c.setLineWidth(1)
    c.roundRect(margin, y - card_h, tw, card_h, 8, fill=0, stroke=1)
    c.setFillColor(C_ACCENT)
    c.rect(margin, y - card_h, 5, card_h, fill=1, stroke=0)

    cy = y - 22
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(C_PRIMARY)
    c.drawString(margin + 18, cy, "PATIENT PROFILE")
    cy -= 20

    for label, value in [
        ("Current Health State", profile.health),
        ("Age Group",            profile.age),
        ("Comorbidity Level",    profile.comorbidity),
        ("Recommended Action",   optimal_policy[profile.health].replace("_", " ")),
        ("Expected Value (V*)",  f"{V_star[profile.health]:.2f}"),
    ]:
        c.setFont("Helvetica", 9)
        c.setFillColor(C_MUTED)
        c.drawString(margin + 18, cy, label)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(C_PRIMARY)
        c.drawString(margin + 210, cy, value)
        cy -= 16

    y -= card_h + 25

    # Disclaimer
    c.setFillColor(colors.HexColor("#FFFDF0"))
    c.setStrokeColor(C_WARNING)
    c.setLineWidth(1.5)
    c.roundRect(margin, y - 40, tw, 40, 5, fill=1, stroke=1)
    c.setFillColor(C_WARNING)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 10, y - 14, "⚠  DISCLAIMER")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.black)
    c.drawString(margin + 10, y - 28,
        "This report is an academic simulation only. It does NOT constitute medical advice "
        "and must not be used for clinical decision-making.")
    y -= 60

    # Section 1
    y = _section_header(c, "1.  Purpose & Methodology", y, width, margin)
    body = (
        "This report presents the output of a Markov Decision Process (MDP) optimisation engine "
        "applied to patient treatment planning under uncertainty. The system models health state "
        "transitions, treatment costs, and clinical risk to derive a personalised, long-term optimal "
        "treatment policy via Policy Iteration with a discount factor γ = 0.95.\n"
        "Transition probabilities are adjusted dynamically for the patient's age group and comorbidity "
        "level. The penalty correctly targets the most desirable outcome state (including Recovery), "
        "ensuring that high-risk patients realistically face reduced recovery probabilities."
    )
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    for line in _wrap_text(c, body, "Helvetica", 10, tw):
        c.drawString(margin, y, line); y -= 14
    y -= 10

    # Section 2
    y = _section_header(c, "2.  Model Components", y, width, margin)
    table_data = [
        ["Health States",       ", ".join(health_states)],
        ["Actions Available",   ", ".join(a.replace("_", " ") for a in actions)],
        ["Terminal States",     "Recovered, Deceased"],
        ["Discount Factor (γ)", "0.95"],
        ["Algorithm",           "Policy Iteration (convergence δ < 1×10⁻⁸)"],
        ["No-Treatment Penalty","Removed — progression risk captured in transitions only"],
        ["Terminal Rewards",    "One-time health reward only (no treatment cost applied)"],
    ]
    tbl = Table(table_data, colWidths=[175, tw - 175])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (0, -1),  C_LIGHT),
        ("TEXTCOLOR",      (0, 0), (0, -1),  C_PRIMARY),
        ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",       (1, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_WHITE, C_BG]),
        ("GRID",           (0, 0), (-1, -1), 0.4, C_MUTED),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
    ]))
    tw2, th = tbl.wrapOn(c, tw, 300)
    tbl.drawOn(c, margin, y - th)
    y -= th + 15

    _draw_footer(c, page_num, width, margin)
    c.showPage(); page_num += 1

    # ── PAGE 2 : POLICY TABLE + BAR CHART ────────────────────────────────────
    _new_page(c, width, height)
    _page_header(c, "Optimal Treatment Policy",
                 "Per-state recommended actions and expected long-term values",
                 width, height, margin)

    y = height - 80
    y = _section_header(c, "3.  Per-State Policy & Value Function", y, width, margin)

    pol_data = [["Health State", "Recommended Action", "Expected Value (V*)"]]
    for state in health_states:
        pol_data.append([
            state,
            optimal_policy[state].replace("_", " "),
            f"{V_star[state]:.2f}"
        ])

    pol_tbl = Table(pol_data, colWidths=[150, 200, 150])
    pol_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_PRIMARY),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_BG]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
        ("LEFTPADDING",    (0, 0), (-1, -1), 10),
        ("ALIGN",          (2, 0), (-1, -1), "CENTER"),
    ]))
    for i, state in enumerate(health_states, 1):
        act = optimal_policy[state]
        v   = V_star[state]
        bg  = action_row_colors.get(act, C_WHITE)
        vc  = C_SUCCESS if v > 5 else (C_WARNING if v > 0 else C_DANGER)
        pol_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, i), (-1, i), bg),
            ("TEXTCOLOR",  (2, i), (2, i),  vc),
            ("FONTNAME",   (2, i), (2, i),  "Helvetica-Bold"),
        ]))

    pw, ph = pol_tbl.wrapOn(c, tw, 500)
    pol_tbl.drawOn(c, margin, y - ph)
    y -= ph + 16

    # Legend
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(C_PRIMARY)
    c.drawString(margin, y, "Action Legend:")
    lx = margin + 105
    for act_label, col in [
        ("No Treatment", C_WARNING),
        ("Medication",   C_ACCENT),
        ("Surgery",      C_SUCCESS),
    ]:
        lx = _badge(c, act_label, lx, y - 1, col)
    y -= 30

    # Bar chart
    y -= 5
    y = _section_header(c, "4.  Expected Value Distribution (V*)", y, width, margin)

    bar_area_w = tw - 80
    bar_h, bar_gap = 16, 8
    max_v  = max(abs(v) for v in V_star.values()) or 1
    zero_x = margin + 80 + bar_area_w * 0.5
    chart_top = y + bar_h

    for state in health_states:
        v     = V_star[state]
        bar_w = abs(v) / max_v * (bar_area_w * 0.48)
        col   = C_SUCCESS if v > 5 else (C_WARNING if v > 0 else C_DANGER)

        c.setFont("Helvetica", 9)
        c.setFillColor(C_PRIMARY)
        c.drawRightString(margin + 75, y + 4, state)

        bx = zero_x if v >= 0 else zero_x - bar_w
        c.setFillColor(col)
        c.roundRect(bx, y, max(bar_w, 2), bar_h, 2, fill=1, stroke=0)

        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(C_PRIMARY)
        if v >= 0:
            c.drawString(bx + bar_w + 4, y + 4, f"{v:.1f}")
        else:
            c.drawRightString(bx - 4, y + 4, f"{v:.1f}")

        y -= bar_h + bar_gap

    c.setStrokeColor(C_MUTED)
    c.setLineWidth(0.8)
    c.line(zero_x, y, zero_x, chart_top)

    _draw_footer(c, page_num, width, margin)
    c.showPage(); page_num += 1

    # ── PAGE 3 : MDP DIAGRAM ─────────────────────────────────────────────────
    if diagram_path:
        _new_page(c, width, height)
        _page_header(c, "MDP State Transition Diagram",
                     "Patient-specific transition probabilities after age & comorbidity adjustment",
                     width, height, margin)
        try:
            img   = Image.open(diagram_path)
            dw    = tw
            dh    = dw * (img.size[1] / img.size[0])
            img_y = max(margin + 30, (height - 58 - dh) / 2)
            c.drawImage(diagram_path, margin, img_y, width=dw, height=dh,
                        preserveAspectRatio=True)
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.setFillColor(C_DANGER)
            c.drawString(margin, height // 2, f"Diagram could not be rendered: {e}")
        _draw_footer(c, page_num, width, margin)
        c.showPage(); page_num += 1

    # ── PAGE 4 : APPENDIX ────────────────────────────────────────────────────
    _new_page(c, width, height)
    _page_header(c, "Appendix & Technical Notes", "", width, height, margin)

    y = height - 80
    y = _section_header(c, "5.  Reward Function", y, width, margin)

    reward_rows = [
        ["Component",        "Description",                       "Values"],
        ["Health Reward",    "Base reward per state",              "Healthy:+8  Mild:+5  Moderate:+2  Severe:−3  Critical:−8  Recovered:+15  Deceased:−20"],
        ["Treatment Cost",   "Cost per active treatment",          "No Treatment: 0  |  Medication: −2  |  Surgery: −6"],
        ["Risk Penalty",     "Procedural risk per active action",  "No Treatment: 0  |  Medication: −2  |  Surgery: −4"],
        ["Surgery Risk Adj.","Extra surgical penalty (profile)",   "Elderly: −3  |  Severe Comorb: −3  |  Adult: −1  |  Moderate Comorb: −1"],
        ["Terminal Reward",  "Applied once, no treatment cost",    "Recovered: +15  |  Deceased: −20  (health reward only)"],
    ]
    r_tbl = Table(reward_rows, colWidths=[120, 155, tw - 275])
    r_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_PRIMARY),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_BG]),
        ("GRID",           (0, 0), (-1, -1), 0.4, C_MUTED),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
    ]))
    rw, rh = r_tbl.wrapOn(c, tw, 300)
    r_tbl.drawOn(c, margin, y - rh)
    y -= rh + 25

    y = _section_header(c, "6.  Patient-Specific Adjustments", y, width, margin)
    adj_rows = [
        ["Factor",       "Level",    "Effect"],
        ["Age",          "Young",    "No adjustment (baseline)"],
        ["Age",          "Adult",    "+0.05 probability shift from best to next-worse state"],
        ["Age",          "Elderly",  "+0.12 probability shift from best to next-worse state"],
        ["Comorbidity",  "None",     "No adjustment (baseline)"],
        ["Comorbidity",  "Moderate", "+0.06 probability shift from best to next-worse state"],
        ["Comorbidity",  "Severe",   "+0.14 probability shift from best to next-worse state"],
        ["Penalty Target","All",     "Best outcome by STATE_VALUE (includes Recovered) — fixes critical-surgery underestimation"],
    ]
    a_tbl = Table(adj_rows, colWidths=[100, 90, tw - 190])
    a_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_ACCENT),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_BG]),
        ("GRID",           (0, 0), (-1, -1), 0.4, C_MUTED),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
    ]))
    aw, ah = a_tbl.wrapOn(c, tw, 300)
    a_tbl.drawOn(c, margin, y - ah)
    y -= ah + 30

    # Final disclaimer
    c.setFillColor(colors.HexColor("#FDF2F2"))
    c.setStrokeColor(C_DANGER)
    c.setLineWidth(1.5)
    c.roundRect(margin, y - 52, tw, 52, 5, fill=1, stroke=1)
    c.setFillColor(C_DANGER)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 10, y - 16, "IMPORTANT NOTICE")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.black)
    c.drawString(margin + 10, y - 31,
        "This document is produced solely for academic and educational purposes.")
    c.drawString(margin + 10, y - 44,
        "All results are simulated and do not represent real medical data or clinical recommendations.")

    _draw_footer(c, page_num, width, margin)
    c.save()
    buffer.seek(0)
    return buffer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streamlit UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(page_title="Healthcare DSS", page_icon="🩺", layout="wide")
st.title("🩺 Healthcare Decision Support System")
st.caption("MDP-based intelligent agent for personalised treatment planning under uncertainty")
st.info("⚠️ Academic simulation — does **not** provide medical advice.")
st.markdown("---")

left, right = st.columns([1.2, 2])

with left:
    st.subheader("Patient Profile")
    health   = st.selectbox("Current Health Condition", non_terminal_states)
    age      = st.selectbox("Age Group", age_groups)
    comorb   = st.selectbox("Comorbidity Level", comorbidities)
    sim_seed = st.number_input("Simulation Seed", min_value=0, max_value=9999, value=42, step=1)
    generate = st.button("Generate Optimal Treatment Plan", use_container_width=True)

with right:
    if generate:
        profile = PatientProfile(health=health, age=age, comorbidity=comorb)
        with st.spinner("Running personalised policy iteration…"):
            optimal_policy, V_star, P_patient = policy_iteration(profile)

        action = optimal_policy[health]
        st.subheader("Recommended Decision")
        st.success(f"**Optimal Treatment:** {action.replace('_', ' ')}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Health State", health)
        m2.metric("Recommended Action", action.replace("_", " "))
        m3.metric("Expected Value", f"{V_star[health]:.2f}")
        st.markdown("---")

        with st.expander("Why was this treatment chosen?", expanded=True):
            st.write(
                f"The MDP was personalised for a **{age}** patient with **{comorb}** comorbidities. "
                "Age and comorbidity shift transition probabilities away from the best possible outcome "
                "(including Recovery), correctly penalising high-risk patients. "
                "Terminal states use a one-time health reward with no treatment cost. "
                "Policy iteration with γ=0.95 selects the action maximising long-term expected reward."
            )

        with st.expander("Full Optimal Policy (all states)"):
            for s, a in optimal_policy.items():
                st.markdown(f"- **{s}** → `{a.replace('_', ' ')}` *(V = {V_star[s]:.2f})*")

        with st.expander("Simulated Patient Trajectory"):
            sim = simulate(health, optimal_policy, P_patient, seed=int(sim_seed))
            if not sim:
                st.info("Patient is already in a terminal state — no trajectory to simulate.")
            else:
                for i, (s, a, s2) in enumerate(sim, 1):
                    st.markdown(f"**Step {i}** — `{s}` → *{a.replace('_', ' ')}* → `{s2}`")

        diagram_path = None
        try:
            md = create_mdp_diagram(P_patient)
            md.render("/tmp/mdp_diagram", view=False)
            diagram_path = "/tmp/mdp_diagram.png"
            st.image(diagram_path, caption="Patient-Specific MDP Flow Diagram")
        except Exception as e:
            st.warning(f"Could not render MDP diagram (Graphviz may not be installed): {e}")

        pdf_buffer = generate_pdf_report(profile, optimal_policy, V_star, diagram_path)
        st.download_button(
            label="📄 Download Executive Report (PDF)",
            data=pdf_buffer,
            file_name="healthcare_report.pdf",
            mime="application/pdf"
        )

st.markdown("---")
st.caption("Markov Decision Processes · Policy Iteration · Streamlit")
