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
- **Causal Inference**: Estimation of ATE and ITE using DoWhy and EconML.
- **LightGBM Hyperparameter Tuning**: Automatic tuning of outcome and treatment models with `GridSearchCV`.
- **Refutation Tests**: Perform data subset refutations for robustness checks.
- **Data Standardization**: Option to standardize the dataset using **StandardScaler**.
- **Causal Graph Visualization**: Visualizes and saves the causal DAG as a PNG image.
- **Easy-to-use Command Line Interface (CLI)**. 

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Parameters](#parameters)
4. [Outputs](#outputs)
5. [Causal Graph](#causal-graph)
6. [License](#license)

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/isoprene-causal.git
cd isoprene-causal
```
### 2. Install dependencies
Create a Python virtual environment and install the necessary packages:
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
  --csv ./examples/your_data.csv \
  --treatment Temp \
  --outcome Isoprene \
  --outdir ./outputs \
  --standardize
```
#### Parameters:

--csv: Path to the input CSV file containing the dataset.

--treatment: The treatment variable (e.g., Temp).

--outcome: The outcome variable (e.g., Isoprene).

--outdir: The output directory to save the results (default is ./outputs).

--standardize: (Optional) If set, the dataset will be standardized using StandardScaler.
#### Parameters Explained

Treatment Variable: The variable whose effect you want to estimate (e.g., Temperature).

Outcome Variable: The response variable (e.g., Isoprene concentration).

Covariates: Other environmental and meteorological factors (e.g., Radiation, Wind Speed, Humidity) used to control for confounders in the causal inference.

### 6. Outputs

The script will generate:

- ```causal_results_<timestamp>.txt```
Contains the causal estimand, estimated ATE, tuned hyperparameters, and refutation results.

- ```ite_<timestamp>.csv```
Contains ITE values for each sample.

- ```causal_graph_<timestamp>.png```
Contains the causal DAG.

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
### Causal Graph
The **Causal Directed Acyclic Graph(DAG)** will be automatically created based on the user-defined DAG in the function ```user_dag_gml()```. The DAG is visualized using **NetworkX** and saved as a PNG image. The graph depicts the causal relationships between variables like ```Temperature```, ```Radiation```, ```Isoprene```, and other meteorological factors.
### 📖 Citation

If you use this code, please cite:

Causal Inference of Urban Isoprene Variability: Quantifying Environmental Drivers and Future Projections under Warming Climate.

### 📜 License

This project is released under the MIT License
.
You are free to use, modify, and distribute this code with attribution.
