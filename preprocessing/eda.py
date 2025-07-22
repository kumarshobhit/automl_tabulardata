import os
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew as scipy_skew

# ---------------------------------------
# General Info
# ---------------------------------------

def get_data_info(df, name=None):
    info_str = f"DataFrame: {name}\n" if name else ""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str += buffer.getvalue() + '\n'
    info_str += str(df.describe()) + "\n\n"
    return info_str

def display_data_info(df, name, output_dir):
    info_str = get_data_info(df, name)
    with open(os.path.join(output_dir, f"{name}_info.txt"), 'w') as f:
        f.write(info_str)

# ---------------------------------------
# Missing Values
# ---------------------------------------

def get_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

def check_missing_values(df, name, output_dir):
    missing = get_missing_values(df)
    with open(os.path.join(output_dir, f"{name}_missing_values.txt"), 'w') as f:
        f.write(f"Missing values in {name}:\n{missing}\n")

# ---------------------------------------
# Correlation
# ---------------------------------------

def plot_correlation_heatmap(df, name=None, save_path=None, show=False):
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(f'Correlation Heatmap{f" for {name}" if name else ""}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    return save_path

def plot_feature_target_correlation(X, y, name=None, save_path=None, show=False):
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    numeric_X = X.select_dtypes(include='number')
    correlations = numeric_X.corrwith(y).abs().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=correlations.values, y=correlations.index, orient='h')
    plt.title(f'Feature Correlation with Target{f" for {name}" if name else ""}')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Feature')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    return save_path, correlations

# ---------------------------------------
# Variance & Importance
# ---------------------------------------

def plot_variance_bar(df, name=None, save_path=None, show=False):
    numeric_df = df.select_dtypes(include='number')
    variances = numeric_df.var().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    variances.plot(kind='bar')
    plt.title(f'Feature Variance{f" for {name}" if name else ""}')
    plt.ylabel('Variance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    return save_path

def plot_feature_importances(X, y, name, output_dir):
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    numeric_X = X.select_dtypes(include='number')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(numeric_X, y)
    importances = pd.Series(model.feature_importances_, index=numeric_X.columns)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances.values, y=importances.index, orient='h')
    plt.title(f'Random Forest Feature Importances for {name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_feature_importances.png"))
    plt.close()

# ---------------------------------------
# Distributions & Boxplots
# ---------------------------------------

def plot_feature_distributions(df, name=None, save_path=None, show=False):
    numeric_df = df.select_dtypes(include='number')
    ax = numeric_df.hist(figsize=(15, 10), bins=30)
    plt.suptitle(f'Distributions{f" for {name}" if name else ""}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    return save_path

def plot_boxplots(df, name=None, save_path=None, show=False):
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=numeric_df, orient='h')
    plt.title(f'Boxplots{f" for {name}" if name else ""}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    return save_path

def plot_outlier_histograms(df, name, output_dir):
    numeric_df = df.select_dtypes(include='number')
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)][col]
        plt.figure(figsize=(8, 4))
        sns.histplot(numeric_df[col], bins=30, color='blue', label='Normal', alpha=0.7)
        if not outliers.empty:
            sns.histplot(outliers, bins=30, color='red', label='Outliers', alpha=0.7)
        plt.title(f'Histogram with Outliers for {col} ({name})')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_outlier_hist_{col}.png"))
        plt.close()

# ---------------------------------------
# Target Distribution
# ---------------------------------------

def plot_target_distribution(y, name=None, save_path=None, show=False):
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    plt.figure(figsize=(10, 5))
    sns.histplot(y, kde=True)
    plt.title(f'Target Distribution{f" for {name}" if name else ""}')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

    if hasattr(y, 'skew'):
        skewness = y.skew()
    else:
        skewness = scipy_skew(y)
    return save_path, skewness

# ---------------------------------------
# Pairplot (Optional - Expensive)
# ---------------------------------------

def plot_pairplot(df, name, output_dir):
    numeric_df = df.select_dtypes(include='number')
    sns.pairplot(numeric_df)
    plt.suptitle(f'Pairplot for {name}')
    plt.savefig(os.path.join(output_dir, f"{name}_pairplot.png"))
    plt.close()

# ---------------------------------------
# Full EDA runner
# ---------------------------------------

def run_eda_for_dataset(dataset_name, x_path, y_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    display_data_info(X, f"{dataset_name}_X", output_dir)
    display_data_info(y, f"{dataset_name}_Y", output_dir)
    plot_correlation_heatmap(X, f"{dataset_name}_X", os.path.join(output_dir, f"{dataset_name}_correlation_heatmap.png"))
    check_missing_values(X, f"{dataset_name}_X", output_dir)
    plot_feature_target_correlation(X, y, dataset_name, os.path.join(output_dir, f"{dataset_name}_feature_target_correlation.png"))
    plot_feature_importances(X, y, dataset_name, output_dir)
    plot_feature_distributions(X, dataset_name, os.path.join(output_dir, f"{dataset_name}_feature_distributions.png"))
    plot_boxplots(X, dataset_name, os.path.join(output_dir, f"{dataset_name}_boxplots.png"))
    plot_target_distribution(y, dataset_name, os.path.join(output_dir, f"{dataset_name}_target_distribution.png"))
    plot_outlier_histograms(X, dataset_name, output_dir)
    # plot_pairplot(X, dataset_name, output_dir)  # Optional, can be slow
