# AlpsML

Análisis ML (XGBoost + SHAP + MCMC) sobre likelihoods de partículas tipo axión
(ALPs) en modelos UV no universales con `alpaca`. Surrogate de χ² entrenado
sobre transiciones raras de mesones (K+ → a π+, B+ → K+ a, …) y exploración del
posterior con `emcee`.

## Estructura

```
AlpsML/
├── notebooks/
│   └── AlpsML.ipynb       # pipeline completo: dataset → XGBoost → SHAP → MCMC
├── outputs/               # generado (gitignored)
│   ├── datasets/          # dataset_alps_uv.csv, posterior_samples_uv.csv, …
│   ├── figures/           # corner_plot_uv.png, SHAP*.png, CurvaAprendizaje.png
│   └── models/            # modelo_alps_uv.json (XGBoost), best_params_uv.json
└── requirements.txt
```

## Filosofía

- **`notebooks/AlpsML.ipynb`**: pipeline en celdas independientes
  (1) generación del dataset χ² con `alpaca` (Latin Hypercube en 6 dimensiones
      sobre `log f_a` y los acoples Peccei–Quinn `pq_qL, pq_lL, pq_uR, pq_dR, pq_eR`),
  (2) entrenamiento XGBoost con búsqueda Optuna y muestras ponderadas,
  (3) interpretabilidad SHAP, (4) MCMC con `emcee` sobre el surrogate.
- **`outputs/`**: todo regenerable desde el notebook. Se ignora en git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/AlpsML.ipynb
```

## Notas

- Las figuras SHAP y la curva de aprendizaje no se guardan automáticamente en el
  notebook actual (solo `plt.show()`). Las versiones que están commit-adas en
  `outputs/figures/` se guardaron manualmente desde Jupyter.
- `alpaca-alps` es el paquete público para los modelos UV; ver
  [github.com/alpaca-physics/alpaca](https://github.com/alpaca-physics/alpaca).
