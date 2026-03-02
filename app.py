import streamlit as st
import numpy as np
import random
from graphviz import Digraph
import io

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# States, Actions, Patient Profile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
health_states = [
    'Healthy', 'Mild', 'Moderate', 'Severe', 'Critical',
    'Recovered', 'Deceased'
]
actions = ['No_Treatment', 'Medication', 'Surgery']
age_groups = ['Young', 'Adult', 'Elderly']
comorbidities = ['None', 'Moderate', 'Severe']
terminal_states = ['Recovered', 'Deceased']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transition Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P_base = {
    'Healthy': {
        'No_Treatment': {'Healthy': 0.8, 'Mild': 0.2},
        'Medication': {'Healthy': 0.9, 'Mild': 0.1},
        'Surgery': {'Healthy': 0.85, 'Mild': 0.1, 'Critical': 0.05},
    },
    'Mild': {
        'No_Treatment': {'Mild': 0.6, 'Moderate': 0.3, 'Healthy': 0.1},
        'Medication': {'Healthy': 0.6, 'Mild': 0.3, 'Moderate': 0.1},
        'Surgery': {'Healthy': 0.7, 'Mild': 0.2, 'Critical': 0.1},
    },
    'Moderate': {
        'No_Treatment': {'Moderate': 0.5, 'Severe': 0.4, 'Mild': 0.1},
        'Medication': {'Mild': 0.5, 'Moderate': 0.3, 'Severe': 0.2},
        'Surgery': {'Healthy': 0.5, 'Moderate': 0.3, 'Critical': 0.2},
    },
    'Severe': {
        'No_Treatment': {'Severe': 0.4, 'Critical': 0.6},
        'Medication': {'Moderate': 0.4, 'Severe': 0.4, 'Critical': 0.2},
        'Surgery': {'Moderate': 0.4, 'Severe': 0.3, 'Critical': 0.3},
    },
    'Critical': {
        'No_Treatment': {'Critical': 0.7, 'Deceased': 0.3},
        'Medication': {'Critical': 0.5, 'Severe': 0.3, 'Deceased': 0.2},
        'Surgery': {'Severe': 0.4, 'Recovered': 0.4, 'Deceased': 0.2},
    },
}
for t in terminal_states:
    P_base[t] = {a: {t: 1.0} for a in actions}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reward Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
health_reward = {
    'Healthy': 8, 'Mild': 5, 'Moderate': 2,
    'Severe': -3, 'Critical': -8,
    'Recovered': 15, 'Deceased': -20
}
treatment_cost = {'No_Treatment': 0, 'Medication': -2, 'Surgery': -6}
risk_penalty = {'No_Treatment': -1, 'Medication': -2, 'Surgery': -4}

def reward(state, action):
    return health_reward[state] + treatment_cost[action] + risk_penalty[action]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Policy Iteration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gamma = 0.9

def policy_evaluation(policy):
    V = {s: 0 for s in health_states}
    while True:
        delta = 0
        for s in health_states:
            a = policy[s]
            v = V[s]
            V[s] = sum(
                P_base[s][a][s2] * (reward(s, a) + gamma * V[s2])
                for s2 in P_base[s][a]
            )
            delta = max(delta, abs(v - V[s]))
        if delta < 1e-6:
            break
    return V

def policy_iteration():
    policy = {s: random.choice(actions) for s in health_states}
    while True:
        V = policy_evaluation(policy)
        stable = True
        for s in health_states:
            old = policy[s]
            policy[s] = max(
                actions,
                key=lambda a: sum(
                    P_base[s][a][s2] * (reward(s, a) + gamma * V[s2])
                    for s2 in P_base[s][a]
                )
            )
            if old != policy[s]:
                stable = False
        if stable:
            break
    return policy, V

optimal_policy, V_star = policy_iteration()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def simulate(start_state, steps=5):
    state = start_state
    history = []
    for _ in range(steps):
        if state in terminal_states:
            break
        action = optimal_policy[state]
        next_state = random.choices(
            list(P_base[state][action].keys()),
            list(P_base[state][action].values())
        )[0]
        history.append((state, action, next_state))
        state = next_state
    return history

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MDP Diagram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_mdp_diagram():
    dot = Digraph(comment='Healthcare MDP', format='png')
    for state in health_states:
        dot.node(state)
    for s in health_states:
        for a in actions:
            for s2, p in P_base[s][a].items():
                if p > 0:
                    dot.edge(s, s2, label=f"{a.replace('_',' ')} ({p:.1f})")
    return dot

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_report(patient_profile, optimal_policy, V_star):
    health, age, comorb = patient_profile
    report = f"""
# Healthcare Decision Support System - Executive Summary

**Patient Profile:**  
- Health State: {health}  
- Age Group: {age}  
- Comorbidity Level: {comorb}  

**Purpose:**  
This system uses a Markov Decision Process (MDP) to recommend optimal treatment plans under uncertainty, balancing health outcomes, treatment cost, and medical risk.

**States & Actions:**  
- Health States: Healthy, Mild, Moderate, Severe, Critical, Recovered, Deceased  
- Actions: No Treatment, Medication, Surgery  

**Reward Structure:**  
- Health-based reward, treatment cost, and risk penalties are combined to calculate expected long-term reward.

**Optimal Policy (Selected Treatment by State):**  
"""
    for state, action in optimal_policy.items():
        report += f"- {state}: {action.replace('_',' ')} (Expected Value: {V_star[state]:.2f})\n"

    report += """

**MDP Simulation:**  
The system can simulate patient trajectories probabilistically to show potential outcomes following the optimal policy.

**Notes:**  
- Terminal states: Recovered, Deceased  
- The system is for academic simulation purposes and does NOT provide medical advice.

"""
    return report

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streamlit UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Healthcare Decision Support System",
    page_icon="🩺",
    layout="wide"
)

# Header
st.markdown(
    "<h1 style='text-align:center;'>Healthcare Decision Support System</h1>"
    "<p style='text-align:center; color: gray;'>An MDP-based intelligent agent for treatment planning under uncertainty</p>",
    unsafe_allow_html=True
)
st.markdown("---")
st.info("This application is an academic simulation and does NOT provide medical advice.")

# Layout
left, right = st.columns([1.2, 2])
with left:
    st.subheader("Patient Profile")
    health = st.selectbox("Current Health Condition", health_states[:-2])
    age = st.selectbox("Age Group", age_groups)
    comorb = st.selectbox("Comorbidity Level", comorbidities)
    generate = st.button("Generate Optimal Treatment Plan", use_container_width=True)

with right:
    if generate:
        action = optimal_policy[health]
        st.subheader("Recommended Decision")
        st.success(f"**Optimal Treatment:** {action.replace('_', ' ')}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Health State", health)
        m2.metric("Selected Action", action.replace("_", " "))
        m3.metric("Expected Value", f"{V_star[health]:.2f}")

        st.markdown("---")
        with st.expander("Why was this treatment chosen?", expanded=True):
            st.write(
                "The decision agent evaluates all possible treatment options using a Markov Decision Process "
                "and selects the action that maximizes **long-term expected reward**, considering probabilistic transitions, treatment costs, and risks."
            )
        with st.expander("Simulated Patient Trajectory (MDP Rollout)"):
            sim = simulate(health)
            for i, (s, a, s2) in enumerate(sim, 1):
                st.markdown(f"**Step {i}**  \nState: `{s}`  \nAction: `{a.replace('_',' ')}`  \nOutcome: `{s2}`")

        # ────────────── Download Executive Report ──────────────
        report = generate_report((health, age, comorb), optimal_policy, V_star)
        st.download_button(
            label="Download Executive Report (Markdown)",
            data=report.encode('utf-8'),
            file_name="executive_report.md",
            mime="text/markdown"
        )

        # ────────────── MDP Diagram ──────────────
        md = create_mdp_diagram()
        diagram_path = "/tmp/mdp_diagram"
        md.render(diagram_path, view=False)
        with open(diagram_path + ".png", "rb") as f:
            st.download_button(
                label="Download MDP Diagram",
                data=f,
                file_name="mdp_diagram.png",
                mime="image/png"
            )
        st.image(diagram_path + ".png", caption="Healthcare MDP Flow Diagram")

st.markdown("---")
st.caption("Developed using Markov Decision Processes, Policy Iteration, and Streamlit")
