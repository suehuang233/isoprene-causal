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
```
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
```
---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/isoprene-causal.git
cd isoprene-causal
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Prepare input data

Your CSV should include:

- **Treatment variable** (e.g., ```Temp```)

- **Outcome variable** (e.g., ```Isoprene```)

- **Covariates** (e.g., ```RH```, ```Radiation```, ```WS```, ```WD```, ```Oxides```, ```Toluene```, ```Month```)

You can start with the example dataset provided in ```examples/example_data.csv```.

### 4. Define your DAG

Open ```src/causal_min_with_tuning.py``` and edit the function ```user_dag_gml()```.

- ```nodes_label```: must match your CSV columns.

- ```nodes_alias```: short codes (A, B, C, …).

- ```edges```: causal edges (e.g., "AY" means A → Y).

- Node ```"U" ```can represent unobserved confounding and should not appear in your dataset.


### 5. Run the pipeline
```bash
python src/causal_min_with_tuning.py \
  --csv ./examples/toy_data_full.csv \
  --treatment Temp \
  --outcome Isoprene \
  --outdir ./outputs
```

### 6. Outputs

The script will generate:

- ```causal_results_<timestamp>.txt```
Contains the causal estimand, estimated ATE, and tuned hyperparameters.

- ```ite_<timestamp>.csv```
Contains ITE values for each sample.

### 📊 Example Toy Dataset

The file examples/toy_data_full.csv contains 12 rows of synthetic monthly data with the following columns:

- RH: Relative Humidity

- Temp: Air Temperature

- Radiation: Incoming Radiation

- WS: Wind Speed

- WD: Wind Direction

- Oxides: NO2+Ozone

- Toluene: VOC proxy

- Month: Month index

- Isoprene: Synthetic isoprene concentration (outcome)

This dataset is for demonstration purposes only.

### 📖 Citation

If you use this code, please cite:

Causal Inference of Urban Isoprene Variability: Quantifying Environmental Drivers and Future Projections under Warming Climate.

### 📜 License

This project is released under the MIT License
.
You are free to use, modify, and distribute this code with attribution.
