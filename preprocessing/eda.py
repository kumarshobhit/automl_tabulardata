import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import skew as scipy_skew

# Display the first few rows and info

def get_data_info(df, name=None):
    info_str = f"DataFrame: {name}\n" if name else ""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str += buffer.getvalue() + '\n'
    info_str += str(df.describe()) + "\n\n"
    return info_str

# Correlation heatmap

def plot_correlation_heatmap(df, name=None, save_path=None, show=False):
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', square=True)
    plt.title(f'Correlation Heatmap{f" for {name}" if name else ""}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    return save_path

# Missing values

def get_missing_values(df):
    missing_values = df.isnull().sum()
    missing = missing_values[missing_values > 0]
    return missing

# Variance bar plot

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

# Feature distributions

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

# Boxplots

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

# Feature-target correlation

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

# Target distribution

def plot_target_distribution(y, name=None, save_path=None, show=False):
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    plt.figure(figsize=(8, 4))
    sns.histplot(y, kde=True)
    plt.title(f'Target Distribution{f" for {name}" if name else ""}')
    plt.xlabel('Target Value')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    # Handle skewness for both pandas and numpy
    if hasattr(y, 'skew'):
        skewness = y.skew()
    else:
        skewness = scipy_skew(y)
    return save_path, skewness
