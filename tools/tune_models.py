from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import (
    CATEGORICAL,
    DATA_PATH,
    NUMERIC,
    TARGET,
    build_preprocessor,
    load_dataset,
    time_split,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "tuning"
DEFAULT_MODEL_PATH = ROOT / "models" / "per_district_final.joblib"

UNTUNED_BASELINE_TEST_MAE = {
    "lightgbm": None,
    "xgboost": None,
    "gbm": 389.88,
    "mlp": 223.59,
    "knn": 413.11,
}

ALL_MODELS = ["lightgbm", "xgboost", "gbm", "mlp", "knn"]
DEFAULT_MODELS = ["lightgbm", "xgboost", "gbm"]


@dataclass
class TuningResult:
    model: str
    best_cv_mae: float
    cv_mae_std: float
    best_params: dict[str, Any]
    test_mae: float
    test_rmse: float
    test_mape: float
    test_r2: float
    tuning_time_s: float
    n_trials: int
    untuned_test_mae: float | None = None
    improvement_pct: float | None = field(default=None)

    def __post_init__(self) -> None:
        if self.untuned_test_mae is not None and self.untuned_test_mae > 0:
            self.improvement_pct = (
                (self.untuned_test_mae - self.test_mae) / self.untuned_test_mae * 100.0
            )


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def lightgbm_objective(seed: int) -> tuple[Callable, Callable]:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    def build(params: dict[str, Any]) -> Any:
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)

    return suggest, build


def xgboost_objective(seed: int) -> tuple[Callable, Callable]:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    def build(params: dict[str, Any]) -> Any:
        return XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **params,
        )

    return suggest, build


def gbm_objective(seed: int) -> tuple[Callable, Callable]:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

    def build(params: dict[str, Any]) -> Any:
        return GradientBoostingRegressor(random_state=seed, **params)

    return suggest, build


MLP_ARCHITECTURES = [
    (64,),
    (128,),
    (256,),
    (512,),
    (64, 32),
    (128, 64),
    (256, 128),
    (128, 64, 32),
    (256, 128, 64),
    (256, 128, 64, 32),
    (512, 256, 128, 64),
]


def mlp_objective(seed: int) -> tuple[Callable, Callable]:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": MLP_ARCHITECTURES[
                trial.suggest_int("arch_idx", 0, len(MLP_ARCHITECTURES) - 1)
            ],
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        }

    def build(params: dict[str, Any]) -> Any:
        return MLPRegressor(
            random_state=seed,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            **params,
        )

    return suggest, build


def knn_objective(seed: int) -> tuple[Callable, Callable]:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }

    def build(params: dict[str, Any]) -> Any:
        return KNeighborsRegressor(n_jobs=-1, **params)

    return suggest, build


OBJECTIVE_FACTORIES: dict[str, Callable[[int], tuple[Callable, Callable]]] = {
    "lightgbm": lightgbm_objective,
    "xgboost": xgboost_objective,
    "gbm": gbm_objective,
    "mlp": mlp_objective,
    "knn": knn_objective,
}

DISPLAY_NAMES = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "gbm": "Gradient Boosting (sklearn)",
    "mlp": "MLP (backprop)",
    "knn": "KNN",
}


def tune_one_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_trials: int,
    cv_splits: int,
    seed: int,
    timeout: int | None = None,
    storage_url: str | None = None,
) -> tuple[TuningResult, Pipeline]:
    print(
        f"\n>>> Tuning {DISPLAY_NAMES[model_key]} ({n_trials} trials, {cv_splits}-fold CV)"
        + (f", timeout={timeout}s" if timeout else "")
        + (f", resumable storage={storage_url}" if storage_url else "")
    )
    suggest_params, build_estimator = OBJECTIVE_FACTORIES[model_key](seed)
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        estimator = build_estimator(params)
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        cv_scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        return float(-cv_scores.mean())

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_value
        marker = " *" if trial.value == best else "  "
        print(
            f"  trial {trial.number + 1:>3}/{n_trials}  CV MAE={trial.value:8.2f}  best={best:8.2f}{marker}"
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    if storage_url:
        study = optuna.create_study(
            study_name=f"tune_{model_key}",
            storage=storage_url,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
        already_done = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        if already_done > 0:
            print(f"  resuming: {already_done} trials already complete")
        remaining_trials = max(n_trials - already_done, 0)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)
        remaining_trials = n_trials

    t0 = time.perf_counter()
    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=timeout,
            callbacks=[progress_callback],
            show_progress_bar=False,
        )
    else:
        print(f"  target n_trials={n_trials} already reached; skipping optimization")
    tuning_time = time.perf_counter() - t0
    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    if completed < n_trials:
        print(f"  (timeout hit after {completed}/{n_trials} trials)")

    best_params = study.best_params
    best_estimator_params = best_params.copy()
    if model_key == "mlp":
        idx = best_estimator_params.pop("arch_idx")
        best_estimator_params["hidden_layer_sizes"] = MLP_ARCHITECTURES[idx]

    cv_mae_std = float(
        np.std(
            [
                t.value
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )
    )

    final_estimator = build_estimator(best_estimator_params)
    final_pipe = Pipeline([("prep", build_preprocessor()), ("model", final_estimator)])
    final_pipe.fit(X_train, y_train)
    y_pred = final_pipe.predict(X_test)

    result = TuningResult(
        model=DISPLAY_NAMES[model_key],
        best_cv_mae=float(study.best_value),
        cv_mae_std=cv_mae_std,
        best_params=best_params,
        test_mae=float(mean_absolute_error(y_test, y_pred)),
        test_rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        test_mape=mape(y_test, y_pred),
        test_r2=float(r2_score(y_test, y_pred)),
        tuning_time_s=tuning_time,
        n_trials=completed,
        untuned_test_mae=UNTUNED_BASELINE_TEST_MAE.get(model_key),
    )
    return result, final_pipe


def print_summary_table(results: list[TuningResult]) -> None:
    print("\n" + "=" * 78)
    print("TUNED MODEL COMPARISON (sorted by test MAE)")
    print("=" * 78)
    sorted_results = sorted(results, key=lambda r: r.test_mae)
    header = f"{'Model':<28} {'Test MAE':>10} {'Test R^2':>9} {'CV MAE':>9} {'Untuned':>9} {'Delta':>8}"
    print(header)
    print("-" * 78)
    for r in sorted_results:
        untuned = (
            f"{r.untuned_test_mae:>9.2f}"
            if r.untuned_test_mae is not None
            else f"{'-':>9}"
        )
        delta = (
            f"{r.improvement_pct:>+7.1f}%"
            if r.improvement_pct is not None
            else f"{'-':>8}"
        )
        print(
            f"{r.model:<28} {r.test_mae:>10.2f} {r.test_r2:>9.3f} {r.best_cv_mae:>9.2f} {untuned} {delta}"
        )
    print("=" * 78)

    print("\nBest hyperparameters per model:")
    for r in sorted_results:
        print(f"\n  {r.model}:")
        for k, v in r.best_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")


def save_results(
    results: list[TuningResult], output_path: Path, config: dict[str, Any]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune supervised regressors with Optuna on Sofia imot.bg data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=DEFAULT_MODELS,
        help="Which models to tune.",
    )
    p.add_argument("--trials", type=int, default=20, help="Optuna trials per model.")
    p.add_argument("--cv-splits", type=int, default=3, help="TimeSeriesSplit folds.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 5 trials, 2 CV splits, lightgbm only.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save results.",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the default tuning_results.json file.",
    )
    p.add_argument(
        "--save-best",
        action="store_true",
        help="Persist the single best-by-test-MAE trained model to disk via joblib.",
    )
    p.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path for the saved best model (only used with --save-best).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-model wall-clock timeout in seconds. Stops the study early if exceeded.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Persist Optuna studies to SQLite so the run can be killed and resumed later. "
        "Re-running with the same --models and --output continues from where it stopped.",
    )
    p.add_argument(
        "--storage",
        type=Path,
        default=None,
        help="Path to the SQLite file backing the resumable studies (only with --resume). "
        "Defaults to <output-dir>/optuna_studies.db.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.models = ["lightgbm"]
        args.trials = 5
        args.cv_splits = 2

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    if not DATA_PATH.exists():
        print(f"ERROR: data file missing at {DATA_PATH}", file=sys.stderr)
        return 2

    print("Loading data...")
    df = load_dataset()
    train, test = time_split(df, test_months=12)
    print(f"  train: {len(train):,} rows, test: {len(test):,} rows")
    print(
        f"  cutoff: {train.snapshot_date.max().date()}  test end: {test.snapshot_date.max().date()}"
    )

    X_train, y_train = train[CATEGORICAL + NUMERIC], train[TARGET].to_numpy()
    X_test, y_test = test[CATEGORICAL + NUMERIC], test[TARGET].to_numpy()

    print(
        f"\nConfig: models={args.models}  trials={args.trials}  cv_splits={args.cv_splits}  seed={args.seed}"
    )
    if args.quick:
        print("        (quick mode active: forced to lightgbm/5 trials/2 splits)")

    storage_url: str | None = None
    if args.resume:
        output_dir = (args.output or DEFAULT_OUTPUT_DIR / "tuning_results.json").parent
        storage_path = args.storage or (output_dir / "optuna_studies.db")
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{storage_path}"
        print(f"        (resume mode: persisting Optuna studies to {storage_path})")

    results: list[TuningResult] = []
    pipelines: dict[str, Pipeline] = {}
    for model_key in args.models:
        try:
            r, fitted_pipe = tune_one_model(
                model_key=model_key,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_trials=args.trials,
                cv_splits=args.cv_splits,
                seed=args.seed,
                timeout=args.timeout,
                storage_url=storage_url,
            )
            results.append(r)
            pipelines[r.model] = fitted_pipe
        except KeyboardInterrupt:
            print("\nInterrupted. Saving partial results...")
            break
        except Exception as exc:
            print(f"  ! {model_key} tuning failed: {exc}")
            continue

    if not results:
        print("\nNo results produced.", file=sys.stderr)
        return 1

    print_summary_table(results)

    if not args.no_save:
        output_path = args.output or (DEFAULT_OUTPUT_DIR / "tuning_results.json")
        config = {
            "models": args.models,
            "trials": args.trials,
            "cv_splits": args.cv_splits,
            "seed": args.seed,
            "timeout": args.timeout,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "quick_mode": args.quick,
        }
        save_results(results, output_path, config)

    if args.save_best:
        winner = min(results, key=lambda r: r.test_mae)
        winner_pipe = pipelines[winner.model]
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(winner_pipe, args.model_output)
        size_mb = args.model_output.stat().st_size / (1024 * 1024)
        print(
            f"\nSaved best model: {winner.model} "
            f"(test MAE {winner.test_mae:.2f} EUR/m^2) -> {args.model_output} ({size_mb:.1f} MB)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
