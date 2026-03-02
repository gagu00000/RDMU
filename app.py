import streamlit as st
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MDP Definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

states = ['Healthy', 'Sick', 'Critical']
actions = ['No_Treatment', 'Medication', 'Surgery']

# Transition probabilities P[s][a][s']
P = {
    'Healthy': {
        'No_Treatment': {'Healthy': 0.85, 'Sick': 0.15},
        'Medication':   {'Healthy': 0.90, 'Sick': 0.10},
        'Surgery':      {'Healthy': 0.88, 'Sick': 0.10, 'Critical': 0.02},
    },
    'Sick': {
        'No_Treatment': {'Sick': 0.65, 'Critical': 0.25, 'Healthy': 0.10},
        'Medication':   {'Healthy': 0.55, 'Sick': 0.35, 'Critical': 0.10},
        'Surgery':      {'Healthy': 0.65, 'Sick': 0.20, 'Critical': 0.15},
    },
    'Critical': {
        'No_Treatment': {'Critical': 0.70, 'Sick': 0.15, 'Healthy': 0.15},
        'Medication':   {'Critical': 0.50, 'Sick': 0.40, 'Healthy': 0.10},
        'Surgery':      {'Critical': 0.25, 'Sick': 0.35, 'Healthy': 0.40},
    },
}

# Rewards R[s][a]
R = {
    'Healthy': {'No_Treatment':  4, 'Medication':  3, 'Surgery': -2},
    'Sick':    {'No_Treatment': -6, 'Medication':  2, 'Surgery': -1},
    'Critical':{'No_Treatment':-12, 'Medication': -4, 'Surgery':  5},
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Policy Iteration Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def policy_evaluation(policy, gamma=0.9, theta=1e-6):
    V = {s: 0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            a = policy[s]
            V[s] = sum(
                P[s][a][s_next] * (R[s][a] + gamma * V[s_next])
                for s_next in P[s][a]
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration(gamma=0.9):
    policy = {s: np.random.choice(actions) for s in states}

    while True:
        V = policy_evaluation(policy, gamma)
        policy_stable = True

        for s in states:
            old_action = policy[s]
            policy[s] = max(
                actions,
                key=lambda a: sum(
                    P[s][a][s_next] * (R[s][a] + gamma * V[s_next])
                    for s_next in P[s][a]
                )
            )
            if old_action != policy[s]:
                policy_stable = False

        if policy_stable:
            break

    return policy, V

# Compute optimal policy once
optimal_policy, optimal_values = policy_iteration()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streamlit UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(page_title="Healthcare Decision Bot (MDP)", layout="centered")

st.title("Healthcare Decision Bot")
st.subheader("MDP-based Treatment Recommendation System")

st.markdown("""
This application uses a **Markov Decision Process (MDP)** to recommend an
**optimal treatment plan** based on the patient's current health condition.

> ⚠️ This is a **decision-support demonstration**, not a medical diagnostic tool.
""")

# User input
current_state = st.selectbox(
    "Select the patient's current health condition:",
    states
)

if st.button("Get Treatment Recommendation"):
    recommended_action = optimal_policy[current_state]

    st.success(f"Recommended Treatment: **{recommended_action.replace('_', ' ')}**")

    st.markdown("### Why this recommendation?")
    st.write(
        f"""
        - The system evaluates **long-term outcomes**, not just immediate effects.
        - For the **{current_state}** state, the action **{recommended_action.replace('_', ' ')}**
          maximizes expected future reward under uncertainty.
        - This accounts for recovery chances, risks, and treatment costs.
        """
    )

    st.markdown("### Expected Long-Term Value")
    st.write(
        f"Estimated value for state **{current_state}**: "
        f"**{optimal_values[current_state]:.2f}**"
    )

# Footer
st.markdown("---")
st.caption("Built using Markov Decision Processes and Policy Iteration")
