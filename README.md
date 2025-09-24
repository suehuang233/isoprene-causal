# isoprene-causal
Public code release accompanying the paper Causal Inference of Urban Isoprene Variability: Quantifying Environmental Drivers and Future Projections under Warming Climate.
# Causal Inference of Urban Isoprene Variability

This repository contains the public release of code accompanying the paper:

> *Causal Inference of Urban Isoprene Variability: Quantifying Environmental Drivers and Future Projections under Warming Climate*  
> (Author: <Your Name>, Year: <YYYY>)

The code provides a minimal and transparent pipeline for estimating **Average Treatment Effect (ATE)** and **Individual Treatment Effect (ITE)** of environmental drivers on **isoprene concentrations** using [DoWhy](https://github.com/py-why/dowhy) and [EconML](https://github.com/microsoft/EconML).  
It leverages **Double Machine Learning (CausalForestDML)** with LightGBM base learners and includes hyperparameter tuning.

---

## ✨ Features
- **Minimal causal inference pipeline** with DoWhy + EconML.  
- **User-editable DAG**: adapt the causal graph to your own study design.  
- **LightGBM hyperparameter tuning** with `GridSearchCV`.  
- Outputs both **ATE** and **per-sample ITE**.  
- Includes a **toy dataset** for demonstration.  

---

## 📂 Repository Structure
isoprene-causal/
├─ src/
│ └─ causal_min_with_tuning.py # main script
├─ examples/
│ ├─ toy_data_full.csv # toy dataset (12 rows, monthly data)
│ └─ sample_dag.md # guide on how to edit the DAG
├─ outputs/ # created after running (empty by default)
├─ requirements.txt
├─ README.md
└─ LICENSE

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/isoprene-causal.git
cd isoprene-causal
### 2. Install dependencies
```bash
pip install -r requirements.txt
### 3. Prepare input data

Your CSV should include:

Treatment variable (e.g., Temp)

Outcome variable (e.g., Isoprene)

Covariates (e.g., RH, Radiation, WS, WD, Oxides, Toluene, Month)

You can start with the toy dataset provided in examples/toy_data_full.csv.
