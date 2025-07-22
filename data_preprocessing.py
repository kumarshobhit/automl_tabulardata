import os
import sys
import importlib
import pandas as pd
from preprocessing import default_preprocessing as config_module

# Add preprocessing folder to path for imports (if needed)
PREPROCESSING_DIR = os.path.join(os.path.dirname(__file__), "preprocessing")
if PREPROCESSING_DIR not in sys.path:
    sys.path.insert(0, PREPROCESSING_DIR)

# Import modules explicitly according to your structure
eda = importlib.import_module("eda")
feature_eng = importlib.import_module("feature_engineering")
outliers = importlib.import_module("handle_outliers")

DEFAULT_PREPROCESSING_CONFIG = config_module.DEFAULT_PREPROCESSING_CONFIG


def run_eda(df_X, df_y, dataset_name, output_dir, config):
    if not config["eda"]["enabled"]:
        print("EDA disabled in config.")
        return
    os.makedirs(output_dir, exist_ok=True)
    print(f"Running EDA for {dataset_name}...")

    eda.display_data_info(df_X, f"{dataset_name}_X", output_dir)
    eda.display_data_info(df_y, f"{dataset_name}_Y", output_dir)
    eda.plot_correlation_heatmap(df_X, f"{dataset_name}_X", output_dir)
    eda.check_missing_values(df_X, f"{dataset_name}_X", output_dir)
    eda.plot_feature_target_correlation(df_X, df_y, dataset_name, output_dir)
    eda.plot_feature_importances(df_X, df_y, dataset_name, output_dir)
    eda.plot_feature_distributions(df_X, dataset_name, output_dir)
    eda.plot_boxplots(df_X, dataset_name, output_dir)
    if config["eda"].get("pairplot", False):
        eda.plot_pairplot(df_X, dataset_name, output_dir)
    eda.plot_target_distribution(df_y, dataset_name, output_dir)
    eda.plot_outlier_histograms(df_X, dataset_name, output_dir)


def run_imputation(df, config):
    if not config["imputation"]["enabled"]:
        print("Imputation disabled.")
        return df
    print(f"Running imputation using method: {config['imputation']['option']}")
    # Assuming feature_engineering.py has `impute_missing_values(df, method)`
    method = config['imputation'].get("option", "median")
    return feature_eng.impute_missing_values(df, method=method)


def run_encoding(df, config):
    if not config["encoding"]["enabled"]:
        print("Encoding disabled.")
        return df
    option = config["encoding"]["option"]
    print(f"Running encoding using method: {option}")
    if option == "onehot":
        return feature_eng.encode_categorical(df)
    elif option == "label":
        print("Label encoding not implemented, defaulting to onehot.")
        return feature_eng.encode_categorical(df)
    else:
        print(f"Unknown encoding option {option}, skipping encoding.")
        return df


def run_scaling(df, config):
    if not config["scaling"]["enabled"]:
        print("Scaling disabled.")
        return df
    option = config["scaling"]["option"]
    print(f"Running scaling using method: {option}")
    # For now only standard implemented; extend as needed
    return feature_eng.scale_features(df)


def run_outlier_handling(df, config):
    if not config["outlier_handling"]["enabled"]:
        print("Outlier handling disabled.")
        return df
    option = config["outlier_handling"]["option"]
    print(f"Running outlier handling using method: {option}")
    if option == "iqr":
        return outliers.handle_outliers_iqr(df)
    elif option == "zscore":
        return outliers.handle_outliers_zscore(df)
    else:
        print("No outlier handling performed.")
        return df


def preprocess_data(df_X, df_y=None, dataset_name="dataset", output_dir="eda_reports", config=DEFAULT_PREPROCESSING_CONFIG):
    run_eda(df_X, df_y, dataset_name, output_dir, config)

    df_X = run_imputation(df_X, config)
    df_X = run_outlier_handling(df_X, config)
    df_X = run_encoding(df_X, config)
    df_X = run_scaling(df_X, config)

    # Add feature selection/generation steps here if needed

    return df_X, df_y


def main():
    data_dir = 'data'
    eda_dir = 'eda_reports'

    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for fold in map(str, range(1, 11)):
            fold_path = os.path.join(dataset_path, fold)
            x_path = os.path.join(fold_path, 'X_train.parquet')
            y_path = os.path.join(fold_path, 'Y_train.parquet')
            if os.path.exists(x_path) and os.path.exists(y_path):
                output_dir = os.path.join(eda_dir, dataset, fold)
                print(f"Preprocessing {dataset} fold {fold}...")
                X = pd.read_parquet(x_path)
                y = pd.read_parquet(y_path)
                X_processed, y_processed = preprocess_data(X, y, f"{dataset}_fold{fold}", output_dir)
                # Optionally save processed data
                X_processed.to_parquet(os.path.join(fold_path, 'X_train_processed.parquet'))
                y_processed.to_parquet(os.path.join(fold_path, 'Y_train_processed.parquet'))
            else:
                print(f"Skipping {dataset} fold {fold}: missing data.")


if __name__ == "__main__":
    main()
