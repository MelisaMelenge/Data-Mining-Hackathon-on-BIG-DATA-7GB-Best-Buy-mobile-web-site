# Best Buy Kaggle System Simulation (Workshop 4)

This folder contains the simulation code and report for **Workshop 4** of the course *Systems Analysis & Design*.  
The workshop implements two simulation approaches—one data-driven and one event-driven—based on the **Best Buy Mobile Web Site**.

---

## Summary

The goal of this workshop is to validate the system architecture developed earlier in the semester by simulating how it behaves under real data and event-driven environments.  
Two simulations were implemented:

1. **Data-Driven Simulation (Classic ML):**  
   A Random Forest model is used to approximate product-category prediction using a sampled portion of the clickstream dataset.

2. **Event-Driven Simulation (Cellular Automata):**  
   A cellular automata model simulates user session states (search, click, abandonment, purchase) to observe emergent patterns and dynamic behaviors.


---

## Simulation Approach

### **1. Data Preparation**
- Used the same Kaggle competition as Workshops 1 and 2:  
  **ACM SF Chapter Hackathon — Best Buy Mobile Web Site**
- Loaded `train.csv` and `test.csv` (≈7GB combined).
- Reduced dataset to ~200,000 interactions to maintain feasible runtime.
- Cleaned and normalized:
  - Converted timestamps to datetime.
  - Lowercased and sanitized queries.
  - Selected only relevant columns (user, sku, category, query, click_time, query_time).
- From `product_data.tar.gz` (~805MB), extracted **one representative keyword** per review using simple keyword selection (e.g., most frequent non-stopword).

### **2. Scenario 1 — Data-Driven Simulation (ML)**
Simulates the system’s learning/prediction workflow.

- TF-IDF encoding of search queries  
- Timestamp and categorical feature engineering  
- Random Forest classifier  
- Evaluation using:
  - MAP@5  
  - Recall@K  
  - Runtime  
- Assessed robustness under noisy or incomplete queries.

### **3. Scenario 2 — Event-Driven Simulation (Cellular Automata)**
Simulates behavior and emergent patterns in user sessions.

- Each cell represents a session state:
  - Query  
  - Click  
  - Exploration  
  - Abandonment  
  - Purchase  
- Transition rules based on:
  - Number of clicks  
  - Dwell time  
  - Query type  
  - Session patterns  
- Evaluated:
  - Stability under perturbations  
  - Convergence or decay patterns  
  - Clusters of user drop-offs  

### **4. System Workflow Validation**
Running both simulations validates:

- Whether modules behave correctly under real data loads  
- Whether performance degrades under noise or chaos-like conditions  
- Whether preprocessing improves prediction stability  

---

## How to Run
1. Clone the repository and navigate to the `Workshop 4` folder.
2. Install required Python packages:  
   `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`

3. Run the simulation scripts:

   - **Data-Driven Simulation:**  
     ```bash
     python simulation_ML.py
     ```

   - **Event-Driven Simulation:**  
     ```bash
     python simulation_CA.py
     ```


---

## Results
Data-Driven ML Simulation

Performance Overview
The ML simulation validated the recommendation pipeline architecture. The Random Forest model processed synthetic clickstream data efficiently, achieving moderate predictive accuracy with sub-second training times.

Key Findings
TF-IDF combined with temporal features captured meaningful user behavior patterns
A notable gap between Top-5 accuracy and MAP@5 indicated correct predictions often ranked lower than optimal
Training efficiency confirmed scalability potential on standard hardware
Feature engineering emerged as the primary performance bottleneck

Limitations
Ambiguous queries and rare categories proved challenging
Class imbalance biased predictions toward popular categories
Current feature set reached an accuracy ceiling requiring semantic understanding

Event-Driven CA Simulation

System Behavior
The cellular automata simulation revealed how user sessions evolved through behavioral states, influenced by query quality and neighbor interactions (social proof effects).

Temporal Dynamics
Three distinct phases emerged:
1. Rapid Reorganization: Browsing sessions quickly resolved to engagement or abandonment
2. Conversion Growth: Clicked sessions transitioned primarily to purchases
3. Equilibrium:  System converged to binary outcomes (abandoned or converted)

Emergent Patterns
Transient states collapsed rapidly, leaving only terminal states at equilibrium
High-quality perturbations created conversion "hotspots" that influenced neighboring cells

Comparative Insights

The simulations provided complementary perspectives:
ML Simulation validated prediction accuracy and computational efficiency
CA Simulation explored collective behavior and emergent dynamics

Next Steps

Immediate priorities:
Implement semantic embeddings for better query understanding
Calibrate CA parameters using real behavioral data
Develop ranking-optimized models for MAP@5

Future enhancements:
- Build hybrid simulation combining ML predictions with CA dynamics
- Create A/B testing framework to evaluate improvements before deployment


---

## Report
- **Full Workshop 4 Simulation Report (PDF):**  
  [`Workshop4_Report.pdf`](./docs/Workshop4_Report.pdf)

---

##  Authors
- **Juan Diego Martínez Beltrán**  
- **Luis Felipe Suárez Sánchez**  
- **Jean Pierre Mora Cepeda**  
- **Melisa Maldonado Melenge**

Universidad Distrital Francisco José de Caldas  
**Systems Analysis & Design – 2025**

