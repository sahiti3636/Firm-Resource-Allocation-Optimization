#  Firm Resource Allocation Optimization

##  Project Overview

This project provides a **Convex Optimization–based decision system** that helps firms distribute their capital across four key channels efficiently:

- **Labour** (linear productivity)  
- **Materials** (linear productivity)  
- **Energy** (linear productivity)  
- **Marketing** (diminishing returns via √M)

The optimization is solved using **Second-Order Cone Programming (SOCP)** with the **ECOS** solver (Interior-Point Method), ensuring **high precision**, especially for the √marketing term where first-order solvers tend to leave slack.

---

##  Team Cute Force

| Member            | ID        |
|-------------------|-----------|
| **Potini Sahiti** | BT2024163 |
| **Navya Sharma**  | BT2024237 |
| **D. Sreeteja**   | BT2024160 |

---

##  Features

###  Interactive Firm Quiz
Converts 8 qualitative business responses into quantitative optimization parameters (productivity, costs, curvature).

###  ECOS-Forced Optimization
Solver explicitly fixed to **ECOS** for stable SOCP execution.

###  Automatic KKT Condition Validation
Checks:
- Stationarity  
- Complementary slackness  
- Primal & dual feasibility

###  Sensitivity Analysis
Analyzes how the firm's total output responds to different budget levels.

---

##  Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

##  Running the Project

```bash
python main.py
```

Choose one of:
- **Option 1 — Interactive Quiz**  
- **Option 2 — Demo Run**

---

##  Visualization Flow

The optimization produces **three sequential plots**.  
Close each to proceed to the next.

1. **Allocation Pie Chart + Bar Chart**  
2. **KKT Residual Plot**  
3. **Sensitivity Analysis Curve**

---

##  Generated Output Files

| File                           | Description                                |
|--------------------------------|--------------------------------------------|
| `allocation_results_ecos.png`  | Optimal resource allocation visualization   |
| `kkt_residuals_check.png`      | Validation of KKT optimality conditions     |
| `sensitivity_analysis_ecos.png`| Output–vs–budget relationship               |

---

##  Tech Stack

- Python 3.x  
- `cvxpy`  
- `numpy`  
- `ecos`  
- `matplotlib`

---
