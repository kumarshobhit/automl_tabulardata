import os
import sys
import importlib
import pandas as pd
from preprocessing import default_preprocessing as config_module
import numpy as np

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
    method = config['imputation'].get("option", "median")
    if method == 'knn':
        import preprocessing.feature_engineering as feature_eng
        df, _ = feature_eng.knn_impute_missing_values(df)
        return df
    else:
        import preprocessing.handle_missing_values as missing
        df, _ = missing.handle_missing_values(df, strategy=method)
        return df


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
    import preprocessing.handle_outliers as outliers
    return outliers.handle_outliers_configurable(df, config["outlier_handling"])[0]


def auto_detect_and_process(df, config):
    """
    Auto-detect datetime and text columns, and apply processing if enabled in config.
    """
    changes = []
    # Datetime features
    if config.get("feature_generation", {}).get("datetime_features", {}).get("enabled", False):
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                changes.append(f"Extracted year/month/day from datetime column {col}")
    # Text features (placeholder, real implementation would use TF-IDF/CountVectorizer)
    if config.get("feature_generation", {}).get("text_features", {}).get("enabled", False):
        for col in df.columns:
            if df[col].dtype == object and df[col].str.len().mean() > 20:
                changes.append(f"Text feature extraction suggested for column {col}")
    return df, changes


def log_transform_skewed_features(df, config):
    """
    Log-transform highly skewed numerical features if enabled in config.
    """
    changes = []
    if config.get("log_transform", {}).get("enabled", False):
        skew_thresh = config["log_transform"].get("skew_thresh", 1.0)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewed = df[numeric_cols].apply(lambda x: x.skew()).abs()
        skewed_cols = skewed[skewed > skew_thresh].index.tolist()
        for col in skewed_cols:
            df[col] = np.log1p(df[col])
            changes.append(f"Log-transformed highly skewed column {col}")
    return df, changes


def preprocess_data(df_X, df_y=None, dataset_name="dataset", output_dir="eda_reports", config=DEFAULT_PREPROCESSING_CONFIG):
    import preprocessing.eda as eda
    import preprocessing.feature_engineering as feature_eng
    changes = []
    # Pre-EDA
    run_eda(df_X, df_y, dataset_name, output_dir, config)
    pre_stats = eda.get_basic_stats(df_X, f"{dataset_name}_X (pre)")
    # Imputation
    df_X = run_imputation(df_X, config)
    # Add missing indicator columns if enabled
    if config.get("missing_indicator", {}).get("enabled", False):
        df_X, indicator_cols = feature_eng.add_missing_indicators(df_X)
        if indicator_cols:
            changes.append(f"Added missing indicator columns: {indicator_cols}")
    # Outlier handling
    df_X = run_outlier_handling(df_X, config)
    # Encoding
    df_X = run_encoding(df_X, config)
    # Scaling
    df_X = run_scaling(df_X, config)
    # Quantile transform (optional)
    if config.get("quantile_transform", {}).get("enabled", False):
        df_X, _ = feature_eng.quantile_transform_features(df_X, output_distribution=config["quantile_transform"].get("output_distribution", "normal"))
        changes.append("Applied quantile transform to numeric features")
    # Power transform (optional)
    if config.get("power_transform", {}).get("enabled", False):
        df_X, _ = feature_eng.power_transform_features(df_X, method=config["power_transform"].get("method", "yeo-johnson"))
        changes.append(f"Applied power transform ({config['power_transform'].get('method', 'yeo-johnson')}) to numeric features")
    # Drop low variance categorical (optional)
    if config.get("feature_generation", {}).get("drop_low_variance_categorical", {}).get("enabled", False):
        df_X, drop_cols = feature_eng.drop_low_variance_categorical_features(df_X, threshold=config["feature_generation"]["drop_low_variance_categorical"].get("threshold", 0.95))
        if drop_cols:
            changes.append(f"Dropped low-variance categorical columns: {drop_cols}")
    # Log-transform skewed features (optional)
    df_X, log_changes = log_transform_skewed_features(df_X, config)
    changes.extend(log_changes)
    # Auto-detect and process datetime/text features (optional)
    df_X, auto_changes = auto_detect_and_process(df_X, config)
    changes.extend(auto_changes)
    # Target transform (optional)
    if config.get("target_transform", {}).get("enabled", False) and df_y is not None:
        option = config["target_transform"].get("option", "none")
        if option != "none":
            df_y = feature_eng.transform_target(df_y, transform_type=option)
            changes.append(f"Applied target transform: {option}")
    # Post-EDA
    eda.run_post_eda(df_X, df_y, dataset_name, output_dir, config)
    post_stats = eda.get_basic_stats(df_X, f"{dataset_name}_X (post)")
    # Save summary report
    eda.save_summary_report(pre_stats, post_stats, changes, output_dir, dataset_name)
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
