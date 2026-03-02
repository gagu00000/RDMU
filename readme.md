# 🩺 Healthcare Decision Support System
### Markov Decision Process-Based Optimal Treatment Planning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF-blue?style=flat)
![License](https://img.shields.io/badge/License-Academic-green?style=flat)

> ⚠️ **Disclaimer:** This is an academic simulation only. It does **not** constitute medical advice and must not be used for any clinical decision-making.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [MDP Formulation](#mdp-formulation)
5. [Reward Function](#reward-function)
6. [Patient Personalisation](#patient-personalisation)
7. [Algorithm: Policy Iteration](#algorithm-policy-iteration)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Project Structure](#project-structure)
11. [Known Limitations](#known-limitations)
12. [Future Improvements](#future-improvements)
13. [Key Design Decisions & Bug Fixes](#key-design-decisions--bug-fixes)
14. [References](#references)

---

## 🔍 Overview

The **Healthcare Decision Support System (DSS)** is an interactive application that uses a **Markov Decision Process (MDP)** to compute optimal treatment recommendations for patients across different health states. The system accounts for:

- Probabilistic transitions between health states
- Treatment costs and procedural risks
- Patient-specific factors: **age group** and **comorbidity level**
- Long-term expected outcomes using **discounted rewards**

The application is built with **Streamlit** for the interactive UI, **Graphviz** for MDP visualisation, and **ReportLab** for professional PDF report generation.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Personalised MDP** | Transition probabilities adjusted per patient age and comorbidity |
| 📊 **Policy Iteration** | Deterministic convergence to globally optimal policy |
| 🗺️ **MDP Diagram** | Colour-coded, readable state transition graph via Graphviz |
| 📄 **PDF Report** | 4-page professional executive report with charts and tables |
| 🔁 **Trajectory Simulation** | Reproducible patient rollout with configurable seed |
| 📈 **Value Function Chart** | Visual bar chart of expected values across all states |

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  Patient Profile Input  →  Generate Button  →  Results UI   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    MDP Engine (Core Logic)                   │
│                                                             │
│  PatientProfile  →  build_transitions()                     │
│       │                    │                                │
│       │           Personalised P(s'|s,a)                   │
│       │                    │                                │
│       └────────►  policy_iteration()                        │
│                    │              │                         │
│             policy_evaluation()   └─► Optimal Policy π*    │
│                    │                        │               │
│               Value Function V*             │               │
└────────────────────┬────────────────────────┼───────────────┘
                     │                        │
          ┌──────────▼──────┐      ┌──────────▼──────────┐
          │  Graphviz Diagram│      │   PDF Report        │
          │  (PNG export)    │      │   (ReportLab)       │
          └──────────────────┘      └─────────────────────┘
```

---

## 🧮 MDP Formulation

The system models patient treatment as a finite MDP defined by the 5-tuple **(S, A, P, R, γ)**.

### States (S)

| State | Description | Type |
|---|---|---|
| `Healthy` | Patient in full health | Active |
| `Mild` | Minor symptoms, manageable condition | Active |
| `Moderate` | Noticeable deterioration requiring attention | Active |
| `Severe` | Serious condition requiring urgent care | Active |
| `Critical` | Life-threatening, immediate intervention needed | Active |
| `Recovered` | Patient has fully recovered | **Terminal** |
| `Deceased` | Patient has died | **Terminal** |

Terminal states are **absorbing** — once entered, the patient remains there. Their value is computed analytically as a one-time reward with no treatment cost:
```
V(terminal) = R_health(terminal) / (1 - γ)
```

### Actions (A)

| Action | Description | Typical Use |
|---|---|---|
| `No_Treatment` | Watchful waiting, no active intervention | Low-severity states |
| `Medication` | Pharmacological treatment | Mild to moderate conditions |
| `Surgery` | Invasive surgical procedure | Severe to critical conditions |

### Transition Probabilities P(s' | s, a)

Base transitions reflect clinical assumptions about how each action affects disease progression. Example for the **Critical** state:

| Action | → Severe | → Recovered | → Deceased | → Stay Critical |
|---|---|---|---|---|
| No Treatment | — | — | 0.30 | 0.70 |
| Medication | 0.30 | — | 0.20 | 0.50 |
| Surgery | 0.40 | **0.40** | 0.20 | — |

These are then **personalised** based on the patient's age and comorbidity — see [Patient Personalisation](#patient-personalisation).

### Discount Factor (γ)
```
γ = 0.95
```

A value of 0.95 ensures the agent values long-term health outcomes strongly while preventing disproportionate inflation of terminal state values. It was raised from 0.90 after finding the lower value caused Recovered's discounted value to dominate policy decisions unrealistically.

---

## 💰 Reward Function

The reward `R(s, a, profile)` is:
```
R(s, a) = health_reward(s) + treatment_cost(a) + risk_penalty(a) + surgery_adjustment(profile)
```

> **Important:** For **terminal states**, only `health_reward(s)` is returned — no treatment costs are deducted since no action is being taken.

### Health Reward

| State | Reward |
|---|---|
| Healthy | +8 |
| Mild | +5 |
| Moderate | +2 |
| Severe | −3 |
| Critical | −8 |
| Recovered | **+15** |
| Deceased | **−20** |

### Treatment Cost

| Action | Cost |
|---|---|
| No Treatment | 0 |
| Medication | −2 |
| Surgery | −6 |

### Risk Penalty

| Action | Penalty | Rationale |
|---|---|---|
| No Treatment | **0** | Disease progression risk already captured in transition probabilities — not double-counted |
| Medication | −2 | Side-effect and complication risk |
| Surgery | −4 | Base procedural risk |

### Surgery Risk Adjustment

| Factor | Level | Adjustment |
|---|---|---|
| Age | Young | 0 |
| Age | Adult | −1 |
| Age | Elderly | −3 |
| Comorbidity | None | 0 |
| Comorbidity | Moderate | −1 |
| Comorbidity | Severe | −3 |

---

## 👤 Patient Personalisation

The system adjusts transition probabilities based on two patient-specific factors. The key design principle: probability mass is shifted **away from the best outcome** (identified by `STATE_VALUE`) and **toward the next-worse state**.

### State Value Ordering
```python
STATE_VALUE = {
    'Recovered': 7,   # best possible outcome
    'Healthy':   6,
    'Mild':      5,
    'Moderate':  4,
    'Severe':    3,
    'Critical':  2,
    'Deceased':  0    # worst possible outcome
}
```

This ordering is critical. For a Critical patient undergoing Surgery where the distribution includes `{Severe: 0.4, Recovered: 0.4, Deceased: 0.2}`, the penalty correctly reduces the probability of **Recovered** (best outcome, value=7) — not Severe. Earlier versions incorrectly excluded Recovered from consideration, making surgery appear risk-free for high-risk patients.

### Penalty Magnitudes

| Factor | Level | Penalty |
|---|---|---|
| Age | Young | 0.00 |
| Age | Adult | 0.05 |
| Age | Elderly | 0.12 |
| Comorbidity | None | 0.00 |
| Comorbidity | Moderate | 0.06 |
| Comorbidity | Severe | 0.14 |

**Maximum combined penalty** (Elderly + Severe comorbidity): `0.12 + 0.14 = 0.26`

The shift is capped at `min(total_penalty, best_prob × 0.8)` to prevent negative probabilities. All distributions are renormalised after adjustment.

---

## 🔄 Algorithm: Policy Iteration

Policy Iteration finds the optimal policy π* maximising expected discounted cumulative reward.
```
┌─────────────────────────────────────────┐
│  Initialise: π(s) = No_Treatment ∀s     │  ← Deterministic start
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Policy Evaluation               │
│                                         │
│  V(terminal) = R(t) / (1 - γ)          │  ← Analytical
│                                         │
│  For non-terminal s, iterate until      │
│  δ < 1×10⁻⁸ (max 2000 iterations):     │
│                                         │
│  V(s) = Σ P(s'|s,π(s)) ×               │
│             [R(s,π(s)) + γ·V(s')]       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Policy Improvement              │
│                                         │
│  π'(s) = argmax_a Σ P(s'|s,a) ×        │
│                       [R(s,a) + γ·V(s')]│
└──────────────────┬──────────────────────┘
                   │
          ┌────────▼────────┐
          │   π' == π?      │
          └────┬──────┬─────┘
              Yes      No
               │        └──────► repeat (max 500 outer iters)
               ▼
        Return π*, V*
```

**Convergence guarantee:** Policy iteration always converges to the globally optimal policy for finite MDPs in a finite number of steps. Terminal states are excluded from the improvement loop since their policy is irrelevant.

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Graphviz **system binary** (separate from the Python package)

### Step 1 — Install Graphviz system binary

**macOS:**
```bash
brew install graphviz
```

**Ubuntu / Debian:**
```bash
sudo apt-get install graphviz
```

**Windows:** Download from [graphviz.org/download](https://graphviz.org/download) and check **"Add Graphviz to PATH"** during installation.

**Verify:**
```bash
dot -V
# Expected: dot - graphviz version X.X.X
```

### Step 2 — Install Python dependencies
```bash
pip install streamlit graphviz pillow reportlab
```

Or using `requirements.txt`:
```
streamlit>=1.28.0
graphviz>=0.20.0
Pillow>=10.0.0
reportlab>=4.0.0
```
```bash
pip install -r requirements.txt
```

### Streamlit Cloud Deployment

Create `packages.txt` in your repo root:
```
graphviz
```
Streamlit Cloud reads this file and installs the system binary before launching the app.

---

## 🚀 Usage

### Run locally
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`.

### Step-by-step

1. **Select patient profile** — current health condition, age group, comorbidity level
2. **Set simulation seed** — any integer for reproducible trajectory rollout
3. **Click "Generate Optimal Treatment Plan"**
4. Review the recommended action, expected value (V*), and explanation
5. Expand **"Full Optimal Policy"** to see recommendations for all states
6. Expand **"Simulated Patient Trajectory"** for a step-by-step state rollout
7. View the colour-coded **MDP Flow Diagram**
8. **Download the 4-page PDF Executive Report**

---

## 📁 Project Structure
```
healthcare-dss/
│
├── app.py              # All application logic and UI (single-file)
├── requirements.txt    # Python package dependencies
├── packages.txt        # System packages for Streamlit Cloud (graphviz)
└── README.md           # This file
```

---

## ⚠️ Known Limitations

| Limitation | Details |
|---|---|
| **Synthetic transitions** | Probabilities are illustrative — not derived from real clinical data |
| **Discrete actions** | Real treatment involves far more nuanced intervention options |
| **Infinite horizon** | Does not model treatment duration, recovery timelines, or episode length |
| **Static comorbidities** | Comorbidity level is fixed; in reality it evolves over time |
| **Abstract cost model** | Reward penalties are relative values, not real monetary figures |
| **Deterministic rewards** | No uncertainty in the reward signal given state and action |
| **No diagnostic uncertainty** | Assumes the true health state is perfectly observable (full observability) |

---

## 🔮 Future Improvements

- [ ] **POMDP formulation** — model diagnostic uncertainty where the true state is unobserved
- [ ] **Real clinical data** — calibrate transition probabilities from EHR / clinical trial datasets
- [ ] **Multi-objective optimisation** — explicitly balance health outcomes vs financial cost
- [ ] **Time-dependent transitions** — model how disease progression rate changes over time
- [ ] **Continuous state space** — replace discrete states with continuous biomarker values (e.g. lab results)
- [ ] **Explainability layer** — SHAP-style attribution of why a particular action was chosen
- [ ] **Sensitivity analysis** — show how policy changes across a range of γ values or reward weights
- [ ] **Multi-agent extension** — model interactions between patient, physician, and healthcare system

---

## 🧠 Key Design Decisions & Bug Fixes

The following critical issues were identified and corrected during development:

| # | Issue | Fix |
|---|---|---|
| 1 | Patient age/comorbidity collected but ignored | `build_transitions()` personalises P(s'\|s,a) per profile |
| 2 | Policy computed once at module load | `policy_iteration()` runs inside button handler, per patient |
| 3 | Terminal states charged treatment costs in reward | `reward()` returns `health_reward` only for terminal states |
| 4 | No-Treatment risk double-counted transitions | `risk_penalty['No_Treatment'] = 0` |
| 5 | Penalty targeted wrong state (Recovered excluded) | `STATE_VALUE` dict ensures Recovered is penalised correctly |
| 6 | Non-deterministic policy initialisation | Fixed start: `π(s) = No_Treatment ∀s` |
| 7 | Terminal states iterated in evaluation loop | Computed analytically: `V(t) = R(t) / (1 − γ)` |
| 8 | γ=0.90 inflated terminal values ~10× | Raised to γ=0.95 with analytical terminal value seeding |
| 9 | Zero-sum distribution silent failure | `_normalize()` raises `ValueError` on invalid distributions |
| 10 | Black PDF background on pages 2–4 | `_new_page()` helper paints white background on every page |
| 11 | Fill color bleeding across PDF elements | Explicit `setFillColor(black)` reset after every colored element |
| 12 | Lambda parameter shadowing outer variable `a` | Renamed to `act` in policy improvement step |

---

## 📚 References

- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming.* Wiley.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Sonnenberg, F. A., & Beck, J. R. (1993). Markov models in medical decision making. *Medical Decision Making, 13*(4), 322–338.

---

<div align="center">

**Built with** Streamlit · ReportLab · Graphviz · Policy Iteration

*Academic simulation — not for clinical use*

</div>
