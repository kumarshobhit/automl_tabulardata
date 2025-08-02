import argparse
from pathlib import Path
import os
import time
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import neps

from sklearn.metrics import r2_score

from automl.automl import AutoML
from automl.data import Dataset

def get_search_space(X):
    n_features = X.shape[1]
    # Build space as in your latest pipeline random/grid search
    return {
        "imputer": neps.Categorical(["simple", "knn"]),
        "scaler": neps.Categorical(["standard", "minmax"]),
        "encoder": neps.Categorical(["onehot", "ordinal"]),
        "feature_selector": neps.Categorical(["none","var_thresh", "select_k_best"]),
        "var_thresh": neps.Float(lower=1e-8, upper=0.001, log=True),
        "select_k": neps.Categorical(
            ["all"] + [k for k in [5, 10, 20] if (isinstance(k, int) and k < n_features)]
        ),
        "n_estimators": neps.Categorical([100, 200, 500, 1000]),
        "max_depth": neps.Categorical(["none", 10, 20, 30]),
        "n_folds": neps.Integer(lower=3, upper=10, is_fidelity=True)
        # You can add other model params or models here
    }

def run_neps_on_dataset(
    task, run_dir, max_evals=15, seed=42, folds=10, optimizer="hyperband"
):
    # Use fold 1 for pipeline space extraction
    X_train = Dataset.load(datadir=Path("data"), task=task, fold=1).X_train

    pipeline_space = get_search_space(X_train)

    def run_pipeline(**config):
        n_folds = int(config.get("n_folds", 10))    # picks up the fidelity value
        # Remove n_folds from config (so not passed to AutoML)
        config = {k: v for k, v in config.items() if k != "n_folds"}

         # --- CONVERT "none" to None ---
        if "max_depth" in config and isinstance(config["max_depth"], str) and config["max_depth"].lower() == "none":
            config["max_depth"] = None

        r2s = []
        for fold in range(1, n_folds + 1):
            dataset = Dataset.load(datadir=Path("data"), task=task, fold=fold)
            automl = AutoML(seed=seed, config=config)
            automl.fit(dataset.X_train, dataset.y_train)
            preds = automl.predict(dataset.X_test)
            if dataset.y_test is not None:
                r2s.append(r2_score(dataset.y_test, preds))
        mean_r2 = np.mean(r2s) if r2s else 0.0
        return {
            "objective_to_minimize": -mean_r2,  # NEPS minimizes! Return -R^2
            "loss": -mean_r2,                   # (optional, for redundancy)
            "info_dict": {"mean_r2": mean_r2, "fold_r2s": list(r2s)}
        }


    neps.run(
        evaluate_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=os.path.join(run_dir, task),
        max_evaluations_total=max_evals,
    )

def plot_neps(
    plt,
    root_directory,
    task_name,
    log_x=False,
    log_y=True,
):
    csv_path = Path(root_directory) / task_name / "summary/full.csv"
    if not csv_path.exists():
        print(f"No NEPS results found at: {csv_path}")
        return plt
    df = pd.read_csv(csv_path, index_col=0)
    df = df.sort_values(by=["time_sampled", "time_end"])
    # New: use 'loss' if present, otherwise use 'objective_to_minimize'
    loss_col = "loss" if "loss" in df.columns else (
    "objective_to_minimize" if "objective_to_minimize" in df.columns else None
    )
    if not loss_col:
        print(f"Neither 'loss' nor 'objective_to_minimize' column found in {csv_path}. Available columns: {df.columns}")
        return plt

    plt.plot(
        df["cost"].cumsum().values if "cost" in df else np.arange(len(df)),
        np.minimum.accumulate(df[loss_col].values),
        label=task_name,
    )

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.xlabel("Cumulative cost" if "cost" in df else "Evaluation")
    plt.ylabel("Best -R² so far")
    plt.legend()
    return plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The seed for the run(s)")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["bike_sharing_demand"], # as a default, but can use --all
                        help="Dataset(s) to run search on")
    parser.add_argument("--all", action="store_true", help="Run NEPS on all 5 datasets")
    parser.add_argument("--optimizer", type=str, default="hyperband",
                        choices=["hyperband", "successive_halving"], help="NePS Optimizer")
    parser.add_argument("--root_directory", type=str, default="./neps_results/", help="Where NEPS output is stored")
    parser.add_argument("--plot_directory", type=str, default="./outputs/", help="Where plots will be saved")
    parser.add_argument("--max_evals", type=int, default=15, help="How many config-evaluations to run")
    parser.add_argument("--only_plot", action="store_true", help="Only plot/load results, don't run NEPS")
    parser.add_argument("--verbose", action="store_true", help="Show logs at INFO level")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    os.makedirs(args.root_directory, exist_ok=True)
    os.makedirs(args.plot_directory, exist_ok=True)

    DATASETS = [
        "bike_sharing_demand",
        "brazilian_houses",
        "superconductivity",
        "wine_quality",
        "yprop_4_1",
        "exam_dataset"  # Added exam_dataset
    ]
    start_time = time.time()
    task_list = DATASETS if args.all else args.tasks

    # Run NEPS
    if not args.only_plot:
        for task in task_list:
            print(f"=== Running NEPS on {task} ===")
            run_neps_on_dataset(
                task=task,
                run_dir=args.root_directory,
                max_evals=args.max_evals,
                seed=args.seed,
                optimizer=args.optimizer,
            )

    # Plotting
    for task in task_list:
        plt.clf()
        plot_neps(
            plt,
            root_directory=args.root_directory,
            task_name=task,
            log_y=False,
        )
        fpath = Path(args.plot_directory) / f"neps_results_{task}.pdf"
        plt.title(f"NEPS Results: {task}")
        plt.savefig(fpath, dpi=300)
        print(f"Saved plot to {fpath}")

    print(f'Duration: {time.time() - start_time:.2f}s')
