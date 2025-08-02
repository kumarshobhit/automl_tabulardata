import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.automl.automl import AutoML
from datetime import datetime

DATASETS = [
    "bike_sharing_demand",
    "brazilian_houses",
    "wine_quality",
    "superconductivity",
    "yprop_4_1",
]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_dataset(dataset_name, seed=42):
    print(f"\n📊 Evaluating: {dataset_name}")
    scores = []
    folds = list(range(1, 11))

    for fold in folds:
        base_path = f"data/{dataset_name}/{fold}"
        if not os.path.exists(base_path):
            print(f"⚠️ Fold {fold} for {dataset_name} not found. Skipping.")
            continue

        X_train = pd.read_parquet(f"{base_path}/X_train.parquet")
        y_train = pd.read_parquet(f"{base_path}/y_train.parquet").values.ravel()
        X_test = pd.read_parquet(f"{base_path}/X_test.parquet")
        y_test = pd.read_parquet(f"{base_path}/y_test.parquet").values.ravel()

        model = AutoML(seed=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        scores.append({"fold": fold, "r2_score": r2})
        print(f"✅ Fold {fold}: R² = {r2:.4f}")

    df_scores = pd.DataFrame(scores)
    csv_path = os.path.join(RESULTS_DIR, f"{dataset_name}_fold_scores.csv")
    df_scores.to_csv(csv_path, index=False)

    mean_r2 = df_scores["r2_score"].mean()
    std_r2 = df_scores["r2_score"].std()
    print(f"\n📌 Summary for {dataset_name}:")
    print(f"Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"📁 Saved fold scores to: {csv_path}")
    return dataset_name, mean_r2, std_r2

def evaluate_all_datasets():
    summary = []
    start = datetime.now()

    for dataset in DATASETS:
        result = evaluate_dataset(dataset)
        summary.append(result)

    print("\n🧾 Final Summary Table:")
    print("-" * 40)
    print(f"{'Dataset':25s} {'Mean R²':>10s} {'± Std':>10s}")
    print("-" * 40)
    for name, mean, std in summary:
        print(f"{name:25s} {mean:10.4f} {std:10.4f}")
    print("-" * 40)

    duration = datetime.now() - start
    print(f"\n⏱️ Total time: {duration}")

if __name__ == "__main__":
    evaluate_all_datasets()
