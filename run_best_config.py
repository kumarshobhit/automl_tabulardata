import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from automl.automl import AutoML
from automl.data import Dataset
from sklearn.metrics import r2_score

def evaluate_best_config(task, config, seed=42, datadir="data"):
    fold_scores = []
    print(f"Evaluating config on all 10 folds of task: {task}")
    for fold in range(1, 11):
        dataset = Dataset.load(datadir=Path(datadir), task=task, fold=fold)
        automl = AutoML(seed=seed, config=config)
        automl.fit(dataset.X_train, dataset.y_train)
        preds = automl.predict(dataset.X_test)
        if dataset.y_test is not None:
            score = r2_score(dataset.y_test, preds)
            fold_scores.append(score)
            print(f"Fold {fold}: R² = {score:.4f}")
        else:
            print(f"Fold {fold}: y_test not available (skipping score)")
    if fold_scores:
        mean_r2 = np.mean(fold_scores)
        print(f"\nMean R² across 10 folds: {mean_r2:.4f}")
        return mean_r2, fold_scores
    else:
        print("\nNo y_test available for any fold.")
        return None, []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Dataset/task name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--datadir", type=str, default="data", help="Dataset root dir.")
    parser.add_argument("--config_csv", type=str, required=True, help="CSV file with best config (from NEPS or prior search).")
    args = parser.parse_args()

    df = pd.read_csv(args.config_csv)
    # Use NEPS's objective_to_minimize (which is -mean_r2)
    best_row = df.sort_values(by="objective_to_minimize", ascending=True).iloc[0]
    best_mean_r2 = -best_row["objective_to_minimize"]
    print(f"Best mean R² from NEPS: {best_mean_r2:.4f}")

    # Extract config.<param> columns, remove prefix, assign to config dict
    base_keys = set(AutoML(seed=0).config.keys())
    param_cols = [col for col in best_row.index if col.startswith("config.")]
    config = {}

    for col in param_cols:
        base_name = col.replace("config.", "")
        if base_name in base_keys:
            val = best_row[col]
            # Convert "none" to None for max_depth
            if base_name == "max_depth" and isinstance(val, str) and val.lower() == "none":
                config[base_name] = None
            # select_k as int if possible
            elif base_name == "select_k":
                if isinstance(val, float) and val.is_integer():
                    config[base_name] = int(val)
                else:
                    config[base_name] = val
            else:
                config[base_name] = val

    print(f"Best config being evaluated: {config}")

    evaluate_best_config(args.task, config, seed=args.seed, datadir=args.datadir)
