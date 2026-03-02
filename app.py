import streamlit as st
import random
from dataclasses import dataclass
from graphviz import Digraph
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

health_states = ['Healthy','Mild','Moderate','Severe','Critical','Recovered','Deceased']
non_terminal_states = health_states[:-2]
actions = ['No_Treatment', 'Medication', 'Surgery']
age_groups = ['Young', 'Adult', 'Elderly']
comorbidities = ['None', 'Moderate', 'Severe']
terminal_states = ['Recovered', 'Deceased']
gamma = 0.9
MAX_EVAL_ITERS = 1000

@dataclass
class PatientProfile:
    health: str
    age: str
    comorbidity: str

P_base = {
    'Healthy': {
        'No_Treatment': {'Healthy': 0.8, 'Mild': 0.2},
        'Medication':   {'Healthy': 0.9, 'Mild': 0.1},
        'Surgery':      {'Healthy': 0.85, 'Mild': 0.1, 'Critical': 0.05},
    },
    'Mild': {
        'No_Treatment': {'Mild': 0.6, 'Moderate': 0.3, 'Healthy': 0.1},
        'Medication':   {'Healthy': 0.6, 'Mild': 0.3, 'Moderate': 0.1},
        'Surgery':      {'Healthy': 0.7, 'Mild': 0.2, 'Critical': 0.1},
    },
    'Moderate': {
        'No_Treatment': {'Moderate': 0.5, 'Severe': 0.4, 'Mild': 0.1},
        'Medication':   {'Mild': 0.5, 'Moderate': 0.3, 'Severe': 0.2},
        'Surgery':      {'Healthy': 0.5, 'Moderate': 0.3, 'Critical': 0.2},
    },
    'Severe': {
        'No_Treatment': {'Severe': 0.4, 'Critical': 0.6},
        'Medication':   {'Moderate': 0.4, 'Severe': 0.4, 'Critical': 0.2},
        'Surgery':      {'Moderate': 0.4, 'Severe': 0.3, 'Critical': 0.3},
    },
    'Critical': {
        'No_Treatment': {'Critical': 0.7, 'Deceased': 0.3},
        'Medication':   {'Critical': 0.5, 'Severe': 0.3, 'Deceased': 0.2},
        'Surgery':      {'Severe': 0.4, 'Recovered': 0.4, 'Deceased': 0.2},
    },
}
for t in terminal_states:
    P_base[t] = {a: {t: 1.0} for a in actions}

def _normalize(dist):
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()} if total > 0 else dist

def build_transitions(profile):
    """Adjust transition probabilities based on patient age and comorbidity."""
    age_penalty    = {'Young': 0.0, 'Adult': 0.05, 'Elderly': 0.12}
    comorb_penalty = {'None': 0.0, 'Moderate': 0.06, 'Severe': 0.14}
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
            worse = worse_of.get(state)
            if worse:
                best = max(dist, key=lambda s: health_states.index(s) if s not in terminal_states else -1)
                shift = min(total_penalty, dist.get(best, 0) * 0.8)
                dist[best]  = dist.get(best, 0) - shift
                dist[worse] = dist.get(worse, 0) + shift
            dist = {k: max(v, 0) for k, v in dist.items() if max(v, 0) > 0}
            P_patient[state][action] = _normalize(dist)
    return P_patient

health_reward  = {'Healthy': 8, 'Mild': 5, 'Moderate': 2, 'Severe': -3,
                  'Critical': -8, 'Recovered': 15, 'Deceased': -20}
treatment_cost = {'No_Treatment': 0, 'Medication': -2, 'Surgery': -6}
risk_penalty   = {'No_Treatment': -1, 'Medication': -2, 'Surgery': -4}
surgery_risk   = {
    'age':        {'Young': 0, 'Adult': -1, 'Elderly': -3},
    'comorbidity':{'None': 0, 'Moderate': -1, 'Severe': -3},
}

def reward(state, action, profile):
    base = health_reward[state] + treatment_cost[action] + risk_penalty[action]
    if action == 'Surgery':
        base += surgery_risk['age'][profile.age]
        base += surgery_risk['comorbidity'][profile.comorbidity]
    return base

def policy_evaluation(policy, P, profile):
    V = {s: 0.0 for s in health_states}
    for _ in range(MAX_EVAL_ITERS):
        delta = 0.0
        for s in health_states:
            a = policy[s]
            v = V[s]
            V[s] = sum(p * (reward(s, a, profile) + gamma * V[s2]) for s2, p in P[s][a].items())
            delta = max(delta, abs(v - V[s]))
        if delta < 1e-6:
            break
    return V

def policy_iteration(profile):
    P = build_transitions(profile)
    policy = {s: random.choice(actions) for s in health_states}
    for _ in range(500):
        V = policy_evaluation(policy, P, profile)
        stable = True
        for s in health_states:
            old = policy[s]
            policy[s] = max(actions, key=lambda a: sum(
                p * (reward(s, a, profile) + gamma * V[s2]) for s2, p in P[s][a].items()))
            if old != policy[s]:
                stable = False
        if stable:
            break
    return policy, V, P

def simulate(start_state, policy, P, steps=5, seed=42):
    random.seed(seed)
    state, history = start_state, []
    for _ in range(steps):
        if state in terminal_states:
            break
        action = policy[state]
        next_state = random.choices(list(P[state][action].keys()), list(P[state][action].values()))[0]
        history.append((state, action, next_state))
        state = next_state
    return history

def create_mdp_diagram(P):
    dot = Digraph(comment='Healthcare MDP', format='png')
    dot.attr(rankdir='LR')
    for state in health_states:
        dot.node(state, shape='doublecircle' if state in terminal_states else 'circle')
    seen = set()
    for s in health_states:
        for a in actions:
            for s2, p in P[s][a].items():
                if p > 0 and (s, s2, a) not in seen:
                    dot.edge(s, s2, label=f"{a.replace('_',' ')} ({p:.2f})")
                    seen.add((s, s2, a))
    return dot

def generate_pdf_report(profile, optimal_policy, V_star, diagram_path):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin, y = 50, height - 50

    def check_space(needed=20):
        nonlocal y
        if y < margin + needed:
            c.showPage(); y = height - margin

    def write_line(text, font="Helvetica", size=11, indent=0, gap=15):
        nonlocal y
        check_space(gap + 10)
        c.setFont(font, size)
        c.drawString(margin + indent, y, text)
        y -= gap

    write_line("Healthcare Decision Support System", "Helvetica-Bold", 16, gap=25)
    write_line("Executive Summary", gap=30)
    write_line("Patient Profile:", "Helvetica-Bold", 12, gap=20)
    for label, val in [("Health State", profile.health), ("Age Group", profile.age), ("Comorbidity", profile.comorbidity)]:
        write_line(f"{label}: {val}", indent=20)
    y -= 10
    write_line("Optimal Policy by State:", "Helvetica-Bold", 12, gap=20)
    for state, action in optimal_policy.items():
        write_line(f"{state}: {action.replace('_',' ')}  (V = {V_star[state]:.2f})", indent=20)
    y -= 10
    write_line("Notes:", "Helvetica-Bold", 12, gap=20)
    for note in ["Terminal states: Recovered, Deceased",
                 "Policy personalised via age and comorbidity adjustments.",
                 "Academic simulation only — does NOT provide medical advice."]:
        write_line(f"- {note}", indent=20)

    if diagram_path:
        c.showPage()
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, height - margin, "MDP State Transition Diagram")
        try:
            img = Image.open(diagram_path)
            dw = width - 2 * margin
            dh = dw * (img.size[1] / img.size[0])
            c.drawImage(diagram_path, margin, max(margin, (height - dh) / 2), width=dw, height=dh)
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.drawString(margin, height - margin - 30, f"Diagram unavailable: {e}")

    c.save(); buffer.seek(0)
    return buffer

# ── UI ──────────────────────────────────────────────────────────────────────
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
                "Age and comorbidity shift transition probabilities toward worse outcomes and add "
                "surgical risk penalties to the reward function. Policy iteration selects the action "
                "that maximises long-term expected reward under these personalised conditions."
            )

        with st.expander("Full Optimal Policy (all states)"):
            for s, a in optimal_policy.items():
                st.markdown(f"- **{s}** → `{a.replace('_',' ')}` *(V = {V_star[s]:.2f})*")

        with st.expander("Simulated Patient Trajectory"):
            sim = simulate(health, optimal_policy, P_patient, seed=int(sim_seed))
            if not sim:
                st.info("Patient is already in a terminal state — no trajectory to simulate.")
            else:
                for i, (s, a, s2) in enumerate(sim, 1):
                    st.markdown(f"**Step {i}** — `{s}` → *{a.replace('_',' ')}* → `{s2}`")

        diagram_path = None
        try:
            md = create_mdp_diagram(P_patient)
            md.render("/tmp/mdp_diagram", view=False)
            diagram_path = "/tmp/mdp_diagram.png"
            st.image(diagram_path, caption="Patient-Specific MDP Flow Diagram")
        except Exception as e:
            st.warning(f"Could not render MDP diagram (Graphviz may not be installed): {e}")

        pdf_buffer = generate_pdf_report(profile, optimal_policy, V_star, diagram_path)
        st.download_button("📄 Download Executive Report (PDF)", data=pdf_buffer,
                           file_name="healthcare_report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Markov Decision Processes · Policy Iteration · Streamlit")
