import random
from itertools import product
import csv
import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from automl.data import Dataset
from automl.automl import AutoML

# ==== Expand this if you want! ====
SEARCH_SPACE = {
    "imputer": ["simple", "knn"],
    "scaler": ["standard", "minmax"],
    "encoder": ["onehot", "ordinal"],
    "feature_selector": ["none", "var_thresh", "select_k_best"],
    "var_thresh": [0.0, 1e-5, 1e-3],
    "select_k": ["all", 5, 10, 20],
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [None, 10, 20, 30],
}

# ========== Utility Functions ==========

def sample_random_config(search_space):
    # Only use var_thresh if feature_selector == var_thresh
    # Only use select_k if feature_selector == select_k_best
    config = {
        "imputer": random.choice(search_space["imputer"]),
        "scaler": random.choice(search_space["scaler"]),
        "encoder": random.choice(search_space["encoder"]),
        "feature_selector": random.choice(search_space["feature_selector"]),
        "n_estimators": random.choice(search_space["n_estimators"]),
        "max_depth": random.choice(search_space["max_depth"]),
    }
    if config["feature_selector"] == "var_thresh":
        config["var_thresh"] = random.choice(search_space["var_thresh"])
        config["select_k"] = "all"
    elif config["feature_selector"] == "select_k_best":
        config["select_k"] = random.choice(search_space["select_k"])
        config["var_thresh"] = 0.0
    else:
        config["var_thresh"] = 0.0
        config["select_k"] = "all"
    return config


def all_grid_configs(search_space):
    "Exhaustively enumerate all grid search configs."
    keys = list(search_space.keys())
    for values in product(*[search_space[k] for k in keys]):
        yield dict(zip(keys, values))

def get_log_dir(task, cfghash):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path("results") / task / f"config_{cfghash}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def generate_cfg_hash(config):
    "Generate a simple hash for a config dict (ignoring ordering)"
    return hash(frozenset(config.items()))

def ensure_logs_dir():
    Path("logs").mkdir(exist_ok=True)

def log_to_csv(csv_path, header, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# ========== Main Evaluation Function ==========

def evaluate_config_on_folds(config, task, seed, datadir, folds=range(1,11)):
    r2s = []
    fold_scores = []
    cfghash = generate_cfg_hash(config)
    outdir = get_log_dir(task, cfghash)
    for fold in folds:
        dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
        automl = AutoML(seed=seed, config=config)
        automl.fit(dataset.X_train, dataset.y_train)
        preds = automl.predict(dataset.X_test)
        # Always save predictions for reproducibility
        np.save(outdir / f"fold_{fold}_preds.npy", preds)
        score = None
        if dataset.y_test is not None:
            score = r2_score(dataset.y_test, preds)
            r2s.append(score)
            with open(outdir / f"fold_{fold}_metrics.txt", "w") as f:
                f.write(f"R2_test: {score:.6f}\n")
        fold_scores.append(score)
    mean_r2 = np.mean([s for s in fold_scores if s is not None])
    print(f"  Fold R²s: {[round(s,4) for s in fold_scores]}")
    print(f"  Mean R²: {mean_r2:.4f}")
    # Save config and results as a txt file
    with open(outdir / f"config_and_results.txt", "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
        for i, s in enumerate(fold_scores):
            f.write(f"fold_{i+1}_r2: {s}\n")
        f.write(f"mean_r2: {mean_r2}\n")
    return mean_r2, fold_scores, outdir

# ========== Search Functions ==========

def random_search(n_iter, search_space, task, seed, datadir):
    print(f"\nStarting Random Search for {task} ({n_iter} configs)\n")

    # --- DYNAMIC SEARCH SPACE FIX ---
    # Load dataset once to get number of features
    temp_dataset = Dataset.load(datadir=datadir, task=task, fold=1)
    n_features = temp_dataset.X_train.shape[1]
    
    # Create a dynamic search space for select_k
    dynamic_search_space = search_space.copy()
    valid_k_options = [k for k in search_space["select_k"] if isinstance(k, str) or k < n_features]
    if not any(isinstance(k, int) for k in valid_k_options): # Ensure at least one int option if possible
        valid_k_options.append(int(n_features * 0.75)) # e.g., keep 75% of features
    dynamic_search_space["select_k"] = valid_k_options
    print(f"Adjusted 'select_k' options for {n_features} features: {valid_k_options}")
    # --- END FIX ---

    results = []
    best_score = -float("inf")
    best_config = None
    best_fold_scores = None
    best_outdir = None
    ensure_logs_dir()
    for i in range(n_iter):
        config = sample_random_config(dynamic_search_space)
        print(f"\n>> Random search iteration {i+1}/{n_iter}")
        print(f"Config: {config}")
        mean_r2, fold_scores, outdir = evaluate_config_on_folds(
            config, task, seed, datadir
        )
        results.append((config, mean_r2, fold_scores))
        # Save log for each config for reference
        with open(outdir / "search_log.txt", "w") as f:
            f.write(f"Config: {config}\nMean R2: {mean_r2}\nFold R2s: {fold_scores}\n")
        if mean_r2 > best_score:
            best_score = mean_r2
            best_config = config
            best_fold_scores = fold_scores
            best_outdir = outdir
    # Save all results to CSV
    results_dir = Path("results") / task
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"search_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    header = list(SEARCH_SPACE.keys()) + [f"fold_{i+1}_r2" for i in range(10)] + ["mean_r2"]
    rows = [
        [config.get(k) for k in SEARCH_SPACE.keys()]
        + [round(s, 6) if s is not None else "" for s in fold_scores] + [mean_r2]
        for config, mean_r2, fold_scores in results
    ]
    log_to_csv(csv_path, header, rows)
    print(f"\n=== Best Random Search Config ===")
    print(best_config)
    print(f"  Fold R²s: {[round(s,4) for s in best_fold_scores]}")
    print(f"  Mean R²: {best_score:.4f}")
    print(f"\nFull search results saved to {csv_path}")
    print(f"Best run's outputs/logs in: {best_outdir.resolve()}")
    return best_config, best_score

def grid_search(search_space, task, seed, datadir):
    print(f"\nStarting Grid Search for {task}")

     # --- DYNAMIC SEARCH SPACE FIX ---
    # Load dataset once to get number of features
    temp_dataset = Dataset.load(datadir=datadir, task=task, fold=1)
    n_features = temp_dataset.X_train.shape[1]
    
    # Create a dynamic search space for select_k
    dynamic_search_space = search_space.copy()
    valid_k_options = [k for k in search_space["select_k"] if isinstance(k, str) or k < n_features]
    if not any(isinstance(k, int) for k in valid_k_options): # Ensure at least one int option if possible
        valid_k_options.append(int(n_features * 0.75)) # e.g., keep 75% of features
    dynamic_search_space["select_k"] = valid_k_options
    print(f"Adjusted 'select_k' options for {n_features} features: {valid_k_options}")
    # --- END FIX ---

    results = []
    best_score = -float("inf")
    best_config = None
    best_fold_scores = None
    best_outdir = None
    ensure_logs_dir()
    for i, config in enumerate(all_grid_configs(search_space)):
        print(f"\n>> Grid search config {i+1}")
        print(f"Config: {config}")
        mean_r2, fold_scores, outdir = evaluate_config_on_folds(
            config, task, seed, datadir
        )
        results.append((config, mean_r2, fold_scores))
        # Save log for each config for reference
        with open(outdir / "search_log.txt", "w") as f:
            f.write(f"Config: {config}\nMean R2: {mean_r2}\nFold R2s: {fold_scores}\n")
        if mean_r2 > best_score:
            best_score = mean_r2
            best_config = config
            best_fold_scores = fold_scores
            best_outdir = outdir
    # Save all results to CSV
    results_dir = Path("results") / task
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"search_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    header = list(SEARCH_SPACE.keys()) + [f"fold_{i+1}_r2" for i in range(10)] + ["mean_r2"]
    rows = [
        [config.get(k) for k in SEARCH_SPACE.keys()]
        + [round(s, 6) if s is not None else "" for s in fold_scores] + [mean_r2]
        for config, mean_r2, fold_scores in results
    ]
    log_to_csv(csv_path, header, rows)
    print(f"\n=== Best Grid Search Config ===")
    print(best_config)
    print(f"  Fold R²s: {[round(s,4) for s in best_fold_scores]}")
    print(f"  Mean R²: {best_score:.4f}")
    print(f"\nFull search results saved to {csv_path}")
    print(f"Best run's outputs/logs in: {best_outdir.resolve()}")
    return best_config, best_score

# ========== Command Line Interface ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--search", choices=["random", "grid"], required=True)
    parser.add_argument("--iters", type=int, default=10, help="Only for random search.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--all-tasks", action="store_true",help="Run search on all five datasets in sequence.")
    args = parser.parse_args()

    # Validate arguments
    if not args.task and not args.all_tasks:
        parser.error("Either --task or --all-tasks must be specified.")

    # Logging (for debugging/error tracking)
    ensure_logs_dir()
    logfile = Path("logs") / f"run_search_{args.task}_{args.search}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    import logging
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print(f"Logging to {logfile.resolve()}")

    DATASET_LIST = [
    "bike_sharing_demand",
    "brazilian_houses",
    "superconductivity",
    "wine_quality",
    "yprop_4_1",
]
    if args.all_tasks:
        for taskname in DATASET_LIST:
            print("="*50)
            print(f"=== Searching on: {taskname} ===")
            if args.search == "random":
                random_search(
                    n_iter=args.iters,
                    search_space=SEARCH_SPACE,
                    task=taskname,
                    seed=args.seed,
                    datadir=args.datadir,
                )
            else:
                grid_search(
                    search_space=SEARCH_SPACE,
                    task=taskname,
                    seed=args.seed,
                    datadir=args.datadir,
                )
    else:
        if args.search == "random":
            random_search(
                n_iter=args.iters,
                search_space=SEARCH_SPACE,
                task=args.task,
                seed=args.seed,
                datadir=args.datadir,
            )
        else:
            grid_search(
                search_space=SEARCH_SPACE,
                task=args.task,
                seed=args.seed,
                datadir=args.datadir,
            )
