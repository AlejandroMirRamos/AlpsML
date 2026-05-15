"""AlpsML — pipeline completo para GPU server.

Equivalente al notebook AlpsML.ipynb, adaptado para correr headless.
Outputs en AlpsML/outputs/paper/{datasets,models,figures}/

Uso:
    cd /home/aleja/PhysicsML/AlpsML/notebooks
    /path/to/AlpsML/.venv/bin/python run_alps.py

Flags al inicio del archivo:
    FORCE_REGENERATE  — regenerar dataset aunque exista el CSV
    FORCE_RETRAIN     — reentrenar XGBoost aunque exista el modelo
    FORCE_MCMC        — re-correr MCMC aunque exista el posterior CSV
"""

# ── Backend no-interactivo ANTES de cualquier import de matplotlib ──
import matplotlib
matplotlib.use("Agg")

import os
# BLAS: 1 thread por proceso (Pool reparte la carga)
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import warnings
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
import sklearn.metrics
import time
import xgboost as xgb
import optuna
import shap
import corner

# ── GPU forzada ────────────────────────────────────────────────────────────────
_DEVICE = "cuda"   # script de producción — siempre GPU

# ── Paths ──────────────────────────────────────────────────────────────────────
PAPER_DIR    = Path(__file__).parent.parent / "outputs" / "paper"
DATASETS_DIR = PAPER_DIR / "datasets"
MODELS_DIR   = PAPER_DIR / "models"
FIGURES_DIR  = PAPER_DIR / "figures"
for _d in (DATASETS_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_CSV    = DATASETS_DIR / "dataset_alps_uv.csv"
DATASET_BACKUP = DATASETS_DIR / "backup_dataset_alps_uv.csv"
MODEL_PATH     = MODELS_DIR   / "modelo_alps_uv.json"
BEST_PARAMS    = MODELS_DIR   / "best_params_uv.json"
POSTERIOR_CSV  = DATASETS_DIR / "posterior_samples_uv.csv"
CORNER_PNG     = FIGURES_DIR  / "corner_plot_uv.png"

# ── Flags ──────────────────────────────────────────────────────────────────────
FORCE_REGENERATE = False
FORCE_RETRAIN    = False
FORCE_MCMC       = False

# ── Configuración física ───────────────────────────────────────────────────────
N_PUNTOS         = 50_000
CUTOFF           = 10.0
SIGMOID_WIDTH    = 0.4
MA_FIXED         = 2.0
L_BOUNDS_GEN     = [6.0, -10.0, -10.0, -10.0, -10.0, -10.0]
U_BOUNDS_GEN     = [8.0,  10.0,  10.0,  10.0,  10.0,  10.0]
TRANSICIONES_TARGET = [
    "K+ -> a pi+", "K0L -> a pi0", "B+ -> K+ a", "B0 -> K0 a", "B+ -> a pi+"
]
FEATURES = ["log_fa", "pq_qL", "pq_lL", "pq_uR", "pq_dR", "pq_eR"]

# ── Funciones de χ² (nivel de módulo — picklables por mp.Pool) ────────────────
def obtener_chi2_uv(p):
    import alpaca
    from alpaca.uvmodels import PQChargedModel
    from alpaca.statistics import get_chi2, ChiSquaredList

    logfa, pq_qL, pq_lL, pq_uR, pq_dR, pq_eR = p
    fa = 10 ** logfa
    fa_scale = 4 * np.pi * fa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uv_model = PQChargedModel("non-universal model", {
            "qL": [0, 0, pq_qL],
            "lL": [0, 0, pq_lL],
            "uR": pq_uR,
            "dR": pq_dR,
            "eR": pq_eR,
        })
        c = uv_model.get_couplings({}, fa_scale)
        res_list = get_chi2(
            transitions=TRANSICIONES_TARGET, ma=MA_FIXED, couplings=c, fa=fa
        )
        lista_formal = ChiSquaredList(res_list)
        chi2_comb = lista_formal.combine("GlobalUV", r"\text{Global UV}")
        valores = list(chi2_comb.chi2_dict.values())
        return float(np.nansum(valores))


def procesar_punto_paralelo(args):
    i, p = args
    try:
        chi_val = obtener_chi2_uv(p)
        return {
            "ma": MA_FIXED, "log_fa": p[0], "pq_qL": p[1], "pq_lL": p[2],
            "pq_uR": p[3], "pq_dR": p[4], "pq_eR": p[5], "chi2": chi_val,
        }
    except Exception as e:
        return {"error": f"Punto {i}: {e}"}


# ──────────────────────────────────────────────────────────────────────────────
# MCMC paralelo con GPU (mismo motor que CosmoML, likelihood adaptada)
# ──────────────────────────────────────────────────────────────────────────────

def _sokal_tau(post_chain: np.ndarray, n_chains_sample: int = 128) -> float:
    """Tiempo de autocorrelación integrado máximo (ventana de Sokal)."""
    n_post, n_chains, ndim = post_chain.shape
    tau_max = 1.0
    for d in range(ndim):
        for c in range(min(n_chains, n_chains_sample)):
            x = post_chain[:, c, d].astype(float)
            x -= x.mean()
            var = np.dot(x, x) / n_post
            if var < 1e-30:
                continue
            acf = np.correlate(x, x, mode="full")[n_post - 1:] / (var * n_post)
            tau = 1.0
            for k in range(1, n_post // 2):
                tau += 2.0 * acf[k]
                if k > 5 * tau:
                    break
            tau_max = max(tau_max, tau)
    return tau_max


def _parallel_mcmc_alps(
    log_p_fn,
    lows: np.ndarray,
    highs: np.ndarray,
    n_chains: int = 1024,
    n_steps: int = 10_000,
    burn_in: int = 500,
    seed: int = 42,
    ess_target: int = 10_000,
    progress_every: int = 200,
) -> tuple[np.ndarray, int, float, float]:
    """n_chains cadenas RWMH independientes. log_p_fn(arr) -> log-prob array.

    Fases: diagonal adaptativa → covarianza empírica (Cholesky) tras burn_in.
    Para automáticamente cuando ESS ≥ ess_target.
    Retorna (flat_samples, n_steps_actual, tau_max, ess_final).
    """
    rng = np.random.default_rng(seed)
    ndim = len(lows)

    # Posiciones iniciales
    sigma_init = 0.02 * (highs - lows)
    center = (lows + highs) / 2.0
    pos = np.clip(
        center + rng.normal(0, 1, (n_chains, ndim)) * sigma_init,
        lows + 1e-10, highs - 1e-10,
    )
    log_p = log_p_fn(pos)

    chain = np.empty((n_steps, n_chains, ndim), dtype=np.float32)
    chain[0] = pos

    step_scale = 0.05 * (highs - lows)
    L = None
    n_accepted = 0
    t0 = time.time()
    stopped_at = n_steps

    for i in range(1, n_steps):
        if L is None:
            proposal = pos + rng.normal(0, step_scale, (n_chains, ndim))
        else:
            z = rng.standard_normal((n_chains, ndim))
            proposal = pos + (z @ L.T) * (2.38 / np.sqrt(ndim))

        in_box = np.all((proposal >= lows) & (proposal <= highs), axis=1)
        log_p_new = np.full(n_chains, -np.inf)
        if in_box.any():
            log_p_new[in_box] = log_p_fn(proposal[in_box])

        accept = np.log(rng.uniform(size=n_chains)) < (log_p_new - log_p)
        pos    = np.where(accept[:, None], proposal, pos)
        log_p  = np.where(accept, log_p_new, log_p)
        n_accepted += int(accept.sum())
        chain[i] = pos

        if i < 200 and i % 50 == 0 and L is None:
            acc = n_accepted / (i * n_chains)
            step_scale *= 0.7 if acc < 0.15 else (1.4 if acc > 0.40 else 1.0)

        if i == burn_in:
            past = chain[:burn_in].reshape(-1, ndim).astype(float)
            cov = np.cov(past.T) + 1e-8 * np.eye(ndim)
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                pass

        if progress_every and i % progress_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            acc_so_far = n_accepted / (i * n_chains)
            phase = "diagonal" if L is None else "multivar"
            if i > burn_in:
                tau = _sokal_tau(chain[burn_in:i + 1])
                n_post = i - burn_in
                ess = n_post * n_chains / tau
                eta = max(0.0, (ess_target - ess) * tau / (n_chains * rate))
                print(f"  step {i:>5}/{n_steps}  |  {rate:.0f} it/s  |  "
                      f"ETA ≤{eta:.0f}s  |  acc {acc_so_far:.2f}  |  "
                      f"τ={tau:.1f}  ESS={ess:.0f}/{ess_target}  |  {phase}")
                if ess >= ess_target:
                    stopped_at = i + 1
                    break
            else:
                print(f"  step {i:>5}/{n_steps}  |  {rate:.0f} it/s  |  "
                      f"acc {acc_so_far:.2f}  |  {phase}")

    acc_rate = n_accepted / (stopped_at * n_chains)
    elapsed = time.time() - t0
    print(f"  done: {elapsed:.1f}s  |  {stopped_at/elapsed:.0f} it/s  |  "
          f"acceptance {acc_rate:.3f}  |  "
          f"{'diagonal' if L is None else 'multivariate'} proposal")

    post_chain = chain[burn_in:stopped_at]
    samples = post_chain.reshape(-1, ndim).astype(float)
    tau_final = _sokal_tau(post_chain)
    ess_final = (stopped_at - burn_in) * n_chains / tau_final
    print(f"  flat chain: {len(samples):,} muestras  |  "
          f"τ_max={tau_final:.1f}  |  ESS={ess_final:.0f}")
    return samples, stopped_at, tau_final, ess_final


def _ml_log_p_fn(model: xgb.XGBRegressor):
    """Construye log_p_fn GPU para el emulador (sigmoid output)."""
    booster = model.get_booster().copy()
    booster.set_param({"device": "cuda"})
    def fn(arr: np.ndarray) -> np.ndarray:
        prob = booster.inplace_predict(arr.astype(np.float32), iteration_range=(0, 0))
        return np.log(np.clip(prob, 1e-10, 1.0))
    return fn


def _theory_log_p_factory(pool: mp.Pool):
    """Construye log_p_fn teórica usando mp.Pool abierto (CPU paralela).

    Aplica el mismo transform sigmoidal que el dataset de entrenamiento para
    que ambos MCMC (ML y teórico) exploren el mismo posterior.
    """
    def fn(arr: np.ndarray) -> np.ndarray:
        points = [tuple(arr[i]) for i in range(len(arr))]
        chi2s = np.array(pool.map(obtener_chi2_uv, points))
        target = 1.0 / (1.0 + np.exp((chi2s - CUTOFF) / SIGMOID_WIDTH))
        return np.log(np.clip(target, 1e-10, 1.0))
    return fn


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("AlpsML — GPU server run")
    print(f"  PAPER_DIR : {PAPER_DIR}")
    print(f"  FORCE_REGENERATE={FORCE_REGENERATE}  FORCE_RETRAIN={FORCE_RETRAIN}  FORCE_MCMC={FORCE_MCMC}")
    print("=" * 60)

    # ── §1. Dataset χ² ────────────────────────────────────────────────────────
    print("\n=== §1. Dataset χ² ===")
    if not FORCE_REGENERATE and DATASET_CSV.exists():
        print(f"  Usando caché: {DATASET_CSV} ({DATASET_CSV.stat().st_size/1e6:.1f} MB)")
    else:
        sampler = qmc.LatinHypercube(d=6)
        puntos  = qmc.scale(sampler.random(n=N_PUNTOS), L_BOUNDS_GEN, U_BOUNDS_GEN)
        tareas  = list(enumerate(puntos))
        cores   = max(1, mp.cpu_count() - 1)
        print(f"  Calculando {N_PUNTOS} puntos con {cores} cores...")

        dataset = []
        with mp.Pool(processes=cores) as pool:
            for resultado in pool.imap_unordered(
                procesar_punto_paralelo, tareas, chunksize=10
            ):
                if resultado is None:
                    continue
                if "error" in resultado:
                    print(f"  [!] {resultado['error']}")
                else:
                    dataset.append(resultado)
                if len(dataset) > 0 and len(dataset) % 100 == 0:
                    pd.DataFrame(dataset).to_csv(DATASET_BACKUP, index=False)
                    print(f"  Backup: {len(dataset)} puntos", end="\r")

        if dataset:
            df_gen = pd.DataFrame(dataset)
            df_gen["target"] = 1 / (1 + np.exp((df_gen["chi2"] - CUTOFF) / SIGMOID_WIDTH))
            df_gen.to_csv(DATASET_CSV, index=False)
            print(f"\n  Guardado: {DATASET_CSV}  ({len(df_gen)} filas)")

    # ── §2. Inspección del dataset ────────────────────────────────────────────
    print("\n=== §2. Inspección del dataset ===")
    df = pd.read_csv(DATASET_CSV)
    print(df["chi2"].describe().to_string())
    permitidos = len(df[df["target"] > 0.8])
    excluidos  = len(df[df["target"] < 0.2])
    frontera   = len(df[(df["target"] >= 0.2) & (df["target"] <= 0.8)])
    print(f"\n  Permitidos (target~1): {permitidos}")
    print(f"  Excluidos  (target~0): {excluidos}")
    print(f"  Frontera:              {frontera}")

    # ── §3. Entrenamiento XGBoost ─────────────────────────────────────────────
    print("\n=== §3. Entrenamiento XGBoost ===")
    X = df[FEATURES]
    y = df["target"]
    pesos = np.where(y <= 0.8, 5.0, 1.0)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, pesos, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train.values, label=y_train, weight=w_train, feature_names=FEATURES)
    dval   = xgb.DMatrix(X_val.values,   label=y_val,   weight=w_val,   feature_names=FEATURES)
    evals  = [(dtrain, "train"), (dval, "eval")]

    if not FORCE_RETRAIN and MODEL_PATH.exists():
        print(f"  Usando caché: {MODEL_PATH}")
        model_final = xgb.Booster()
        model_final.load_model(str(MODEL_PATH))
    else:
        # Optuna
        def optuna_obj(trial):
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": _DEVICE,
                "max_depth":         trial.suggest_int("max_depth", 2, 6),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1),
                "gamma":             trial.suggest_float("gamma", 0.1, 5.0),
                "min_child_weight":  trial.suggest_int("min_child_weight", 5, 20),
                "subsample":         trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "eval_metric": "mae",
                "nthread": -1,
                "base_score": 0.5,
                "reg_alpha":  trial.suggest_float("reg_alpha",  0.1, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
            }
            m = xgb.train(
                params, dtrain, num_boost_round=3000, evals=evals,
                early_stopping_rounds=50, verbose_eval=False,
            )
            return sklearn.metrics.mean_absolute_error(
                y_val, m.predict(dval), sample_weight=w_val
            )

        print("  Optuna (300 trials, 20 min timeout)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_obj, n_trials=300, timeout=1200, show_progress_bar=True)
        with open(BEST_PARAMS, "w") as f:
            json.dump(study.best_params, f)

        with open(BEST_PARAMS) as f:
            best_params = json.load(f)

        final_config = best_params | {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": _DEVICE,
            "eval_metric": ["mae"],
            "nthread": -1,
            "base_score": 0.5,
        }
        eval_results = {}
        print("  Entrenamiento final...")
        model_final = xgb.train(
            final_config, dtrain, num_boost_round=5000, evals=evals,
            early_stopping_rounds=50, verbose_eval=False, evals_result=eval_results,
        )

        # Curva de aprendizaje
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eval_results["train"]["mae"], label="Train MAE")
        ax.plot(eval_results["eval"]["mae"],  label="Val MAE")
        ax.set_yscale("log")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("MAE")
        ax.set_title("Curva de aprendizaje (con pesos)")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        lc_path = FIGURES_DIR / "learning_curve_uv.png"
        fig.savefig(lc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {lc_path}")

        model_final.save_model(str(MODEL_PATH))
        print(f"  Modelo guardado: {MODEL_PATH}")

    # ── §4. SHAP ──────────────────────────────────────────────────────────────
    print("\n=== §4. SHAP ===")
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(str(MODEL_PATH))

    explainer   = shap.Explainer(model_reg.predict, X)
    shap_values = explainer(X)

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    bs_path = FIGURES_DIR / "shap_beeswarm_uv.png"
    plt.savefig(bs_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {bs_path}")

    # Bar
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    bar_path = FIGURES_DIR / "shap_bar_uv.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {bar_path}")

    # Waterfall
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    wf_path = FIGURES_DIR / "shap_waterfall_uv.png"
    plt.savefig(wf_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {wf_path}")

    # Dependence por feature
    for col in FEATURES:
        shap.plots.scatter(shap_values[:, col], color=shap_values, show=False)
        plt.tight_layout()
        sc_path = FIGURES_DIR / f"shap_scatter_{col}.png"
        plt.savefig(sc_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  Dependence plots guardados en {FIGURES_DIR}/")

    # ── §5. MCMC + corner plot ────────────────────────────────────────────────
    print("\n=== §5. MCMC + corner plot ===")
    L_BOUNDS_MCMC = np.array([6.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    U_BOUNDS_MCMC = np.array([8.0,  5.0,  5.0,  5.0,  5.0,  5.0])
    MCMC_KWARGS   = dict(lows=L_BOUNDS_MCMC, highs=U_BOUNDS_MCMC,
                         n_chains=1024, n_steps=10_000, burn_in=500,
                         seed=42, ess_target=10_000)

    timing_ml = timing_th = None

    # ── §5a. ML MCMC (GPU) ───────────────────────────────────────────────────
    POSTERIOR_CSV_THEORY = DATASETS_DIR / "posterior_samples_uv_theory.csv"
    ml_cached = not FORCE_MCMC and POSTERIOR_CSV.exists()
    if ml_cached:
        print(f"  [ML]  Usando caché: {POSTERIOR_CSV}")
        flat_samples = pd.read_csv(POSTERIOR_CSV).values
        ml_meta = {"cached": True, "wall_s": None, "n_steps_actual": None,
                   "ess_final": None, "tau_max": None,
                   "n_samples": len(flat_samples)}
    else:
        model_mcmc = xgb.XGBRegressor()
        model_mcmc.load_model(str(MODEL_PATH))
        log_p_ml = _ml_log_p_fn(model_mcmc)

        print(f"  [ML]  Iniciando RWMH paralelo GPU (1024 cadenas)...")
        t0_ml = time.perf_counter()
        flat_samples, stopped_ml, tau_ml, ess_ml = _parallel_mcmc_alps(
            log_p_ml, **MCMC_KWARGS,
        )
        timing_ml = time.perf_counter() - t0_ml

        pd.DataFrame(flat_samples, columns=FEATURES).to_csv(POSTERIOR_CSV, index=False)
        print(f"  [ML]  Posterior guardado: {POSTERIOR_CSV}")
        ml_meta = {"cached": False, "wall_s": round(timing_ml, 2),
                   "n_steps_actual": int(stopped_ml), "ess_final": round(ess_ml, 1),
                   "tau_max": round(tau_ml, 2), "n_samples": len(flat_samples)}

    # Corner plot ML
    corner_kwargs = {
        "labels": FEATURES, "show_titles": True, "title_fmt": ".2f",
        "quantiles": [0.16, 0.5, 0.84], "color": "royalblue", "smooth": 0.9,
        "levels": (0.68, 0.95), "fill_contours": True, "plot_datapoints": False,
        "label_kwargs": {"fontsize": 12}, "title_kwargs": {"fontsize": 12},
    }
    fig = corner.corner(flat_samples, **corner_kwargs)
    plt.suptitle("Posterior Distribution — UV Model (ma = 2.0 GeV)", fontsize=16, y=1.02)
    fig.savefig(CORNER_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ML]  Corner plot guardado: {CORNER_PNG}")

    # ── §5b. Theory MCMC (CPU paralela) ─────────────────────────────────────
    th_cached = not FORCE_MCMC and POSTERIOR_CSV_THEORY.exists()
    if th_cached:
        print(f"  [TH]  Usando caché: {POSTERIOR_CSV_THEORY}")
        flat_samples_th = pd.read_csv(POSTERIOR_CSV_THEORY).values
        th_meta = {"cached": True, "wall_s": None, "n_steps_actual": None,
                   "ess_final": None, "tau_max": None,
                   "n_samples": len(flat_samples_th)}
    else:
        cores = max(1, mp.cpu_count() - 1)
        print(f"  [TH]  Iniciando RWMH paralelo teórico ({cores} cores CPU)...")
        t0_th = time.perf_counter()
        with mp.Pool(processes=cores) as pool:
            log_p_th = _theory_log_p_factory(pool)
            flat_samples_th, stopped_th, tau_th, ess_th = _parallel_mcmc_alps(
                log_p_th, **MCMC_KWARGS,
            )
        timing_th = time.perf_counter() - t0_th

        pd.DataFrame(flat_samples_th, columns=FEATURES).to_csv(
            POSTERIOR_CSV_THEORY, index=False
        )
        print(f"  [TH]  Posterior guardado: {POSTERIOR_CSV_THEORY}")
        th_meta = {"cached": False, "wall_s": round(timing_th, 2),
                   "n_steps_actual": int(stopped_th), "ess_final": round(ess_th, 1),
                   "tau_max": round(tau_th, 2), "n_samples": len(flat_samples_th)}

    # Corner comparativo ML vs Theory
    CORNER_CMP_PNG = FIGURES_DIR / "corner_plot_uv_compare.png"
    fig_cmp = corner.corner(flat_samples, color="royalblue",
                            labels=FEATURES, smooth=0.9,
                            levels=(0.68, 0.95), fill_contours=True,
                            plot_datapoints=False)
    corner.corner(flat_samples_th, color="tomato", fig=fig_cmp,
                  smooth=0.9, levels=(0.68, 0.95), fill_contours=True,
                  plot_datapoints=False)
    plt.suptitle("Posterior UV Model — ML (blue) vs Theory (red)", fontsize=14, y=1.01)
    fig_cmp.savefig(CORNER_CMP_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig_cmp)
    print(f"  Corner comparativo guardado: {CORNER_CMP_PNG}")

    # ── §5c. Timings JSON + tabla ─────────────────────────────────────────────
    speedup = (
        th_meta["wall_s"] / ml_meta["wall_s"]
        if (ml_meta["wall_s"] and th_meta["wall_s"])
        else None
    )
    timings = {
        "ml_engine":     "RWMH paralelo 1024 chains, GPU booster (sigmoid)",
        "theory_engine": "RWMH paralelo 1024 chains, CPU mp.Pool (sigmoid)",
        "common": {"n_chains": 1024, "n_steps_cap": 10_000,
                   "burn_in": 500, "seed": 42, "ess_target": 10_000},
        "runs": {
            "mcmc_uv": {"ml": ml_meta, "theory": th_meta,
                        "speedup": round(speedup, 2) if speedup else None},
        },
    }
    timings_path = PAPER_DIR / "timings.json"
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"\n  Timings guardados: {timings_path}")

    # Tabla stdout
    def _fmt(v, unit="s"):
        return f"{v:.1f}{unit}" if v is not None else "cached"
    print("\n" + "─" * 65)
    print(f"{'section':<14}{'ml [s]':>10}{'theory [s]':>12}{'speedup':>10}  "
          f"ess(ml/th)")
    print("─" * 65)
    ess_ml_str = str(int(ml_meta["ess_final"])) if ml_meta["ess_final"] else "–"
    ess_th_str = str(int(th_meta["ess_final"])) if th_meta["ess_final"] else "–"
    sp_str = f"{speedup:.1f}×" if speedup else "–"
    print(f"{'mcmc_uv':<14}{_fmt(ml_meta['wall_s']):>10}"
          f"{_fmt(th_meta['wall_s']):>12}{sp_str:>10}  "
          f"{ess_ml_str}/{ess_th_str}")
    print("─" * 65)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
