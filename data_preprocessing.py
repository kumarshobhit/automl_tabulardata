import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import io

# Display the first few rows and info
def display_data_info(df, name, output_dir):
    info_str = f"DataFrame: {name}\n"
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str += buffer.getvalue() + '\n'
    info_str += str(df.describe()) + "\n\n"
    with open(os.path.join(output_dir, f"{name}_info.txt"), 'w') as f:
        f.write(info_str)

# Correlation heatmap
def plot_correlation_heatmap(df, name, output_dir):
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(f'Correlation Heatmap for {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_correlation_heatmap.png"))
    plt.close()

# Missing values
def check_missing_values(df, name, output_dir):
    missing_values = df.isnull().sum()
    missing = missing_values[missing_values > 0]
    with open(os.path.join(output_dir, f"{name}_missing_values.txt"), 'w') as f:
        f.write(f"Missing values in {name}:\n{missing}\n")

# Feature-target correlation
def plot_feature_target_correlation(X, y, name, output_dir):
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    numeric_X = X.select_dtypes(include='number')
    correlations = numeric_X.corrwith(y).abs().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=correlations.values, y=correlations.index, orient='h')
    plt.title(f'Feature Correlation with Target for {name}')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_feature_target_correlation.png"))
    plt.close()

# Feature importances
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

# Feature distributions
def plot_feature_distributions(df, name, output_dir):
    numeric_df = df.select_dtypes(include='number')
    numeric_df.hist(figsize=(15, 10), bins=30)
    plt.suptitle(f'Distributions for {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_feature_distributions.png"))
    plt.close()

# Boxplots
def plot_boxplots(df, name, output_dir):
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=numeric_df, orient='h')
    plt.title(f'Boxplots for {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_boxplots.png"))
    plt.close()

# Pairplot (optional, can be slow for many features)
def plot_pairplot(df, name, output_dir):
    numeric_df = df.select_dtypes(include='number')
    sns.pairplot(numeric_df)
    plt.suptitle(f'Pairplot for {name}')
    plt.savefig(os.path.join(output_dir, f"{name}_pairplot.png"))
    plt.close()

# Impute missing values
def impute_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

# Encode categorical variables
def encode_categorical(df):
    return pd.get_dummies(df, drop_first=True)

from sklearn.preprocessing import StandardScaler

# Scale features
def scale_features(df):
    numeric_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Target distribution
def plot_target_distribution(y, name, output_dir):
    plt.figure(figsize=(10, 5))
    sns.histplot(y, kde=True)
    plt.title(f'Target Distribution for {name}')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_target_distribution.png"))
    plt.close()

# Outlier histograms
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

# Run EDA for a dataset/fold
def run_eda_for_dataset(dataset_name, x_path, y_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    display_data_info(X, f"{dataset_name}_X", output_dir)
    display_data_info(y, f"{dataset_name}_Y", output_dir)
    plot_correlation_heatmap(X, f"{dataset_name}_X", output_dir)
    check_missing_values(X, f"{dataset_name}_X", output_dir)
    plot_feature_target_correlation(X, y, dataset_name, output_dir)
    plot_feature_importances(X, y, dataset_name, output_dir)
    plot_feature_distributions(X, dataset_name, output_dir)
    plot_boxplots(X, dataset_name, output_dir)
    # plot_pairplot(X, dataset_name, output_dir)  # Uncomment if desired
    plot_target_distribution(y, dataset_name, output_dir)
    plot_outlier_histograms(X, dataset_name, output_dir)
    # Optionally, add more summary/reporting here

# Main loop for all datasets (first fold only)
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
                print(f"Running EDA for {dataset} (fold {fold})...")
                run_eda_for_dataset(f"{dataset}_fold{fold}", x_path, y_path, output_dir)
            else:
                print(f"Skipping {dataset} fold {fold}: missing data.")

if __name__ == "__main__":
    main()