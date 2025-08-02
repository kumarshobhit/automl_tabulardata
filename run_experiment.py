"""
Enhanced runner for AutoML exam project:
- Organizes outputs into folders: ./results/{task}/fold_{fold}/seed_{seed}/
- Logs main checklist stages.
- Saves predictions and (optionally) validation results.
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
from automl.data import Dataset
from automl.automl import AutoML
import argparse
import logging
import datetime

from run_search import SEARCH_SPACE, grid_search, random_search

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
PROJECTDIR = FILE.parent
DATADIR = PROJECTDIR / "data"
RESULTSDIR = PROJECTDIR / "results_new"

def main(
    task: str,
    fold: int,
    output_dir: Path,
    seed: int,
    datadir: Path,
):
    print("# Checklist: Start experiment")
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
    print(" - [x] Loaded dataset:", task, f"(fold {fold})")

    # Initialize and fit AutoML system
    logger.info("Fitting AutoML")
    print(" - [ ] Fitting AutoML (pipeline search and training)")
    automl = AutoML(seed=seed)
    automl.fit(dataset.X_train, dataset.y_train)
    print(" - [x] Fitting AutoML complete")

    # Make predictions
    test_preds: np.ndarray = automl.predict(dataset.X_test)
    print(" - [ ] Predictions for test set generated")

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "test_predictions.npy"
    with pred_path.open("wb") as f:
        np.save(f, test_preds)
    logger.info(f"Predictions written to {pred_path}")
    print(" - [x] Predictions saved at:", pred_path)

    # Evaluate/test if possible
    results = {}
    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test:.4f}")
        print(f" - [x] R^2 score on test set: {r2_test:.4f}")
        results["r2_test"] = r2_test
        # Optionally, save y_test and metrics for report
        np.save(output_dir / "y_test.npy", dataset.y_test)
        with (output_dir / "metrics.txt").open("w") as f:
            f.write(f"R2_test: {r2_test:.6f}\n")
    else:
        print(" - [ ] No test targets provided (exam task submission mode)")
    
    # Checklist summary
    print("\n# Checklist Progress")
    print(" - Load data        ... done")
    print(" - Fit AutoML       ... done")
    print(" - Predict on test  ... done")
    print(" - Save predictions ... done")
    print(" - Log metrics      ...", "done" if 'r2_test' in results else "N/A")
    print("-" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--search", choices=["random", "grid"], required=True)
    parser.add_argument("--iters", type=int, default=10, help="Only for random search.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datadir", type=Path, default=DATADIR)
    parser.add_argument("--resultsdir", type=Path, default=RESULTSDIR)
    args = parser.parse_args()

    # Create logs directory
    LOGSDIR = Path("logs")
    LOGSDIR.mkdir(exist_ok=True)
    import datetime
    logfile = LOGSDIR / f"run_search_{args.task}_{args.search}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    import logging
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print(f"Logging to {logfile.absolute()}")

    if args.search == "random":
        random_search(
            n_iter=args.iters,
            search_space=SEARCH_SPACE,
            task=args.task,
            seed=args.seed,
            data_dir=args.datadir,
            results_dir=args.resultsdir
        )
    else:
        grid_search(
            search_space=SEARCH_SPACE,
            task=args.task,
            seed=args.seed,
            data_dir=args.datadir,
            results_dir=args.resultsdir
        )
    # Run the main experiment with the best config