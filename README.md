# AlpsML

ML analysis (XGBoost + SHAP + MCMC) on likelihoods of axion-like particles
(ALPs) in non-universal UV models with `alpaca`. A χ² surrogate trained on
rare meson decays (K⁺ → a π⁺, B⁺ → K⁺ a, …) and posterior exploration
with `emcee`.

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
and returns the full set of IR couplings used by `alpaca` to compute χ² over
the five meson transitions (K⁺ → a π⁺, K⁰<sub>L</sub> → a π⁰,
B⁺ → K⁺ a, B⁰ → K⁰ a, B⁺ → a π⁺).

The scan is therefore over six physically meaningful UV parameters:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `log_fa`  | log₁₀ of the PQ scale f<sub>a</sub> (GeV) | [6, 8] |
| `pq_qL`   | PQ charge of the 3rd-gen left-handed quark doublet | [−10, 10] |
| `pq_lL`   | PQ charge of the 3rd-gen left-handed lepton doublet | [−10, 10] |
| `pq_uR`   | PQ charge of right-handed up quarks | [−10, 10] |
| `pq_dR`   | PQ charge of right-handed down quarks | [−10, 10] |
| `pq_eR`   | PQ charge of right-handed charged leptons | [−10, 10] |

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
