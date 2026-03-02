# Healthcare Decision Bot using Markov Decision Process (MDP)

This project implements a **Streamlit-based decision-support chatbot** that recommends an optimal healthcare treatment plan using **Markov Decision Process (MDP)** logic and **Policy Iteration**.

The system is designed as an academic demonstration of **decision-making under uncertainty**.

---

## Problem Statement

In healthcare, treatment decisions often involve uncertainty.  
This project models patient health progression as a **Markov Decision Process** to determine the best treatment action based on the patient’s current health condition.

---

## MDP Components

### States
- Healthy
- Sick
- Critical

### Actions
- No Treatment
- Medication
- Surgery

### Transition Probabilities
Each action leads to probabilistic transitions between health states, representing real-world medical uncertainty.

### Rewards
Rewards capture treatment effectiveness, cost, and risk:
- Positive rewards for recovery
- Negative rewards for deterioration or unnecessary treatment

---

## Solution Approach

- The MDP is solved using **Policy Iteration**
- The optimal policy maps each health state to the best treatment action
- Streamlit provides an interactive interface for user input and recommendations

---

## How to Run the Application

### 1. Install dependencies
```bash
pip install -r requirements.txt