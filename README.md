# isoprene-causal
Code and data for Causal Inference of Urban Isoprene Variability: Quantifying Environmental Drivers and Future Projections under Warming Climate.
# Causal Inference of Urban Isoprene Variability

This repository contains the research code and data for the paper:

> *Causal Inference of Urban Isoprene Variability: Quantifying Environmental Drivers and Future Projections under Warming Climate*  
> (Author: Shu HUANG et al., Year: 2026)

The code provides a minimal and transparent pipeline for estimating **Average Treatment Effect (ATE)** and **Individual Treatment Effect (ITE)** of environmental drivers on **isoprene concentrations** using [DoWhy](https://github.com/py-why/dowhy) and [EconML](https://github.com/microsoft/EconML).  
It leverages **Double Machine Learning (CausalForestDML)** with LGBMRegressor base learners and includes hyperparameter tuning.

---
### Environment
This project was developed and tested with:
 - Python 3.11
Install the required packages with:

```bash
pip install -r requirements.txt
```



### 📖 Citation

If you use this code, please cite:

Causal Inference of Urban Isoprene Variability: Quantifying Environmental Drivers and Future Projections under Warming Climate.

### 📜 License

This project is released under the MIT License
.
You are free to use, modify, and distribute this code with attribution.
