import os
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew as scipy_skew
from preprocessing.data_loading import load_tabular_data
from preprocessing.eda import plot_target_distribution

def analyze_target(y):
    # Flatten if DataFrame
    if hasattr(y, 'iloc') and y.shape[1] == 1:
        y = y.iloc[:, 0]
    y_arr = np.array(y)
    unique_vals = np.unique(y_arr)
    n_unique = len(unique_vals)
    min_val = np.min(y_arr)
    max_val = np.max(y_arr)
    skewness = scipy_skew(y_arr)

    print(f"Skewness: {skewness:.2f}")
    print(f"Unique values: {n_unique}")
    print(f"Min: {min_val}, Max: {max_val}")

    if n_unique < 10:
        print("Target has few unique values. Consider classification or ordinal regression.")
    elif abs(skewness) > 1:
        if min_val > 0:
            print("Highly skewed and all positive. Consider log or Yeo-Johnson transformation.")
        else:
            print("Highly skewed with non-positive values. Consider Yeo-Johnson or quantile transformation.")
    else:
        print("Target is not highly skewed. No transformation needed.")


def main():
    DATA_DIR = 'data'
    TARGET_EDA_DIR = 'target_eda_yeojohnson'

    for dataset in os.listdir(DATA_DIR):
        dataset_path = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for fold in map(str, range(1, 11)):
            fold_path = os.path.join(dataset_path, fold)
            y_path = os.path.join(fold_path, 'Y_train.parquet')
            if not os.path.exists(y_path):
                print(f"Skipping {dataset} fold {fold}: missing Y_train.parquet.")
                continue
            print(f"\nProcessing {dataset} fold {fold}...")
            output_dir = os.path.join(TARGET_EDA_DIR, dataset, fold)
            os.makedirs(output_dir, exist_ok=True)

            # Load target variable
            y, y_name = load_tabular_data(y_path)
            print(f"Loaded {y_name} with shape {y.shape}")

            # Analyze target variable
            analyze_target(y)

            # Plot and save original target distribution
            target_dist_path = os.path.join(output_dir, 'target_distribution.png')
            _, skewness = plot_target_distribution(y, name=y_name, save_path=target_dist_path)
            print(f"Saved target distribution plot. Skewness: {skewness:.2f}")

            # If highly skewed, plot Yeo-Johnson-transformed target
            if abs(skewness) > 1:
                print("Target is highly skewed. Applying Yeo-Johnson transformation and plotting again.")
                y_arr = y if not hasattr(y, 'iloc') else y.iloc[:, 0]
                pt = PowerTransformer(method='yeo-johnson')
                y_yeojohnson = pt.fit_transform(np.array(y_arr).reshape(-1, 1)).flatten()
                yj_target_dist_path = os.path.join(output_dir, 'target_distribution_yeojohnson.png')
                _, yj_skewness = plot_target_distribution(y_yeojohnson, name=y_name + ' (yeo-johnson)', save_path=yj_target_dist_path)
                print(f"Saved Yeo-Johnson-transformed target distribution plot. New skewness: {yj_skewness:.2f}")

if __name__ == '__main__':
    main() 