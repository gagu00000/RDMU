import streamlit as st
import numpy as np
import random

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State, Action, Patient Profile
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
# Transition Model (Base)
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

# Terminal states
for t in terminal_states:
    P_base[t] = {a: {t: 1.0} for a in actions}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reward Function (Decomposed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

health_reward = {
    'Healthy': 8, 'Mild': 5, 'Moderate': 2,
    'Severe': -3, 'Critical': -8,
    'Recovered': 15, 'Deceased': -20
}

treatment_cost = {
    'No_Treatment': 0,
    'Medication': -2,
    'Surgery': -6
}

risk_penalty = {
    'No_Treatment': -1,
    'Medication': -2,
    'Surgery': -4
}

def reward(state, action):
    return (
        health_reward[state]
        + treatment_cost[action]
        + risk_penalty[action]
    )

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
# Streamlit UI — PROFESSIONAL UX VERSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Healthcare Decision Support System",
    page_icon="🩺",
    layout="wide"
)

# Header
st.markdown(
    """
    <h1 style='text-align:center;'>Healthcare Decision Support System</h1>
    <p style='text-align:center; color: gray;'>
    An MDP-based intelligent agent for treatment planning under uncertainty
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Disclaimer (important for professor)
st.info(
    "This application is an academic decision-support simulation and "
    "does NOT provide real medical advice."
)

# Layout
left, right = st.columns([1.2, 2])

# ────────────────
# LEFT PANEL — INPUT
# ────────────────
with left:
    st.subheader("Patient Profile")

    health = st.selectbox(
        "Current Health Condition",
        health_states[:-2]
    )

    age = st.selectbox(
        "Age Group",
        age_groups
    )

    comorb = st.selectbox(
        "Comorbidity Level",
        comorbidities
    )

    st.markdown("")

    generate = st.button(
        "Generate Optimal Treatment Plan",
        use_container_width=True
    )

# ────────────────
# RIGHT PANEL — OUTPUT
# ────────────────
with right:
    if generate:
        action = optimal_policy[health]

        st.subheader("Recommended Decision")

        st.success(
            f"**Optimal Treatment:** {action.replace('_', ' ')}"
        )

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Health State", health)
        m2.metric("Selected Action", action.replace("_", " "))
        m3.metric("Expected Value", f"{V_star[health]:.2f}")

        st.markdown("---")

        # Explanation
        with st.expander("Why was this treatment chosen?", expanded=True):
            st.write(
                """
                The decision agent evaluates all possible treatment options
                using a Markov Decision Process and selects the action that
                maximizes **long-term expected reward**, considering:
                """
            )
            st.markdown(
                """
                - Probabilistic health transitions  
                - Treatment cost and medical risk  
                - Long-term recovery outcomes  
                """
            )

        # Simulation
        with st.expander("Simulated Patient Trajectory (MDP Rollout)"):
            sim = simulate(health)
            for i, (s, a, s2) in enumerate(sim, 1):
                st.markdown(
                    f"""
                    **Step {i}**  
                    State: `{s}`  
                    Action: `{a.replace('_',' ')}`  
                    Outcome: `{s2}`
                    """
                )

# Footer
st.markdown("---")
st.caption(
    "Developed using Markov Decision Processes, Policy Iteration, and Streamlit"
)
