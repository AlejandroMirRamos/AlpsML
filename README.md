# AlpsML

ML analysis (XGBoost + SHAP + MCMC) on likelihoods of axion-like particles
(ALPs) in non-universal UV models with `alpaca`. A χ² surrogate trained on
rare meson decays (K⁺ → a π⁺, B⁺ → K⁺ a, …) and posterior exploration
with `emcee`.

## Notebook

**[notebooks/AlpsML.ipynb](https://github.com/AlejandroMirRamos/AlpsML/blob/main/notebooks/AlpsML.ipynb)** — full pipeline: dataset → XGBoost → SHAP → MCMC

## Repository structure

```
AlpsML/
├── notebooks/
│   └── AlpsML.ipynb       # full pipeline: dataset → XGBoost → SHAP → MCMC
├── outputs/               # generated (gitignored)
│   ├── datasets/          # dataset_alps_uv.csv, posterior_samples_uv.csv, …
│   ├── figures/           # corner_plot_uv.png, SHAP*.png, CurvaAprendizaje.png
│   └── models/            # modelo_alps_uv.json (XGBoost), best_params_uv.json
└── requirements.txt
```

## Pipeline overview

- **(1) Dataset generation**: χ² sampled via Latin Hypercube over 6 dimensions —
  `log f_a` and the Peccei–Quinn couplings `pq_qL, pq_lL, pq_uR, pq_dR, pq_eR` —
  using `alpaca`.
- **(2) XGBoost surrogate**: trained with Optuna hyperparameter search and
  weighted samples.
- **(3) SHAP interpretability**: feature importance and dependence plots.
- **(4) MCMC**: posterior sampling with `emcee` over the surrogate χ².

All steps run as independent cells in `notebooks/AlpsML.ipynb`.
`outputs/` is fully regenerable from the notebook and is gitignored.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/AlpsML.ipynb
```

## Notes

- SHAP figures and the learning curve are not saved automatically by the
  notebook (only `plt.show()`). The versions committed under `outputs/figures/`
  were exported manually from Jupyter.
- `alpaca-alps` is the public package for UV models; see
  [github.com/alpaca-physics/alpaca](https://github.com/alpaca-physics/alpaca).
