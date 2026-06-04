# AlpsML

ML analysis (XGBoost + SHAP + MCMC) on likelihoods of axion-like particles
(ALPs) in non-universal UV models with `alpaca`. A χ² surrogate trained on
rare meson decays (K⁺ → a π⁺, B⁺ → K⁺ a, …) and posterior exploration
with `emcee`.

**[→ Open notebook: notebooks/AlpsML.ipynb](https://github.com/AlejandroMirRamos/AlpsML/blob/main/notebooks/AlpsML.ipynb)**

## Index

- [Physics](#physics)
- [Notebook](https://github.com/AlejandroMirRamos/AlpsML/blob/main/notebooks/AlpsML.ipynb) ↗
- [Repository structure](#repository-structure)
- [Pipeline overview](#pipeline-overview)
- [Setup](#setup)
- [Notes](#notes)

## Physics

**Axion-like particles (ALPs)** emerge from the spontaneous breaking of a
global U(1)<sub>PQ</sub> symmetry at scale f<sub>a</sub>. Their couplings to
SM particles are not free parameters: in any UV-complete model they are
entirely determined by the **Peccei–Quinn (PQ) charges** assigned to the SM
fermion representations.

### From PQ charges to ALP couplings

Two rules connect the UV input to the observable IR couplings:

1. **Couplings to fermions** equal the PQ charge of that fermion (up to a sign
   from the Yukawa structure).
2. **Couplings to gauge bosons** (gluons, photons, W/Z) are linear combinations
   of the PQ charges, fixed by the **chiral anomaly coefficients** of the
   PQ current with each gauge group:

$$
g_{aGG} \propto \sum_f T(R_f)\,Q_f^{\rm PQ}, \qquad
g_{a\gamma\gamma} \propto \sum_f Q_f^{\rm em,2}\,Q_f^{\rm PQ}.
$$

This is the physical advantage over scanning over independent IR couplings:
the five PQ charges plus f<sub>a</sub> fully determine the ALP phenomenology,
respecting the UV structure of the theory.

### Non-universal model

The charges are *generation-dependent* for the SU(2)<sub>L</sub> doublets —
only the **third generation** of q<sub>L</sub> and l<sub>L</sub> carries a
non-zero PQ charge — while the right-handed singlets are universal across
generations. In `alpaca`:

```python
from alpaca.uvmodels import PQChargedModel
import numpy as np

model = PQChargedModel('non-universal model', {
    'qL': [0, 0, pq_qL],   # left-handed quark doublet: charge only in 3rd gen
    'lL': [0, 0, pq_lL],   # left-handed lepton doublet: charge only in 3rd gen
    'uR': pq_uR,            # right-handed up quarks: universal
    'dR': pq_dR,            # right-handed down quarks: universal
    'eR': pq_eR,            # right-handed charged leptons: universal
})
couplings = model.get_couplings({}, 4 * np.pi * fa)
```

`get_couplings` propagates the five PQ charges through the anomaly equations
and returns the full set of IR couplings used by `alpaca` to compute the χ² over
its **full observable sector** — the rare meson decays
(K⁺ → a π⁺, K⁰<sub>L</sub> → a π⁰, B⁺ → K⁺ a, B⁰ → K⁰ a, B⁺ → a π⁺) together with
visible channels (e.g. B → K μμ), meson mixing and radiative/leptonic decays.

The scan is therefore over seven physically meaningful parameters — the six UV
inputs plus the ALP mass:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `log_fa`  | log₁₀ of the PQ scale f<sub>a</sub> (GeV) | [6, 8] |
| `pq_qL`   | PQ charge of the 3rd-gen left-handed quark doublet | [−1, 1] |
| `pq_lL`   | PQ charge of the 3rd-gen left-handed lepton doublet | [−1, 1] |
| `pq_uR`   | PQ charge of right-handed up quarks | [−1, 1] |
| `pq_dR`   | PQ charge of right-handed down quarks | [−1, 1] |
| `pq_eR`   | PQ charge of right-handed charged leptons | [−1, 1] |
| `ma`      | ALP mass (GeV) | [1.5, 2.5] |

## Notebook

**[notebooks/AlpsML.ipynb](https://github.com/AlejandroMirRamos/AlpsML/blob/main/notebooks/AlpsML.ipynb)** — full pipeline: dataset → XGBoost → SHAP → MCMC

## Repository structure

```
AlpsML/
├── notebooks/
│   └── AlpsML.ipynb       # full pipeline: dataset → XGBoost → SHAP → MCMC
├── outputs/paper/         # generated (gitignored)
│   ├── datasets/          # dataset_alps_uv_v2.csv, posterior_samples_uv_v2.csv, …
│   ├── figures/           # corner_plot_uv_v2*.png, SHAP*.png, learning_curve_uv_v2.png
│   └── models/            # modelo_alps_{clf,reg}_v2.json (XGBoost), best_params_{clf,reg}_v2.json
└── requirements.txt
```

## Pipeline overview

- **(1) Dataset generation**: χ² sampled via a 7-D Latin Hypercube —
  `log_fa`, the five PQ charges `pq_qL, pq_lL, pq_uR, pq_dR, pq_eR`, and the ALP
  mass `ma` — evaluated with `alpaca` over its full observable sector.
- **(2) XGBoost surrogates**: two models tuned with Optuna — a classifier of the
  allowed/excluded boundary and a regressor of the raw χ² in the allowed region.
- **(3) SHAP interpretability**: feature importance and dependence plots.
- **(4) MCMC**: posterior sampling with `emcee` — the χ² regressor provides the
  likelihood and the classifier acts as a soft wall.

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
