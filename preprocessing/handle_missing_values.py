import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def handle_missing_values(df, strategy='median', n_neighbors=5):
    """
    Handles missing values in a DataFrame:
    - If rows with missing values are <1% of total, drop those rows.
    - Otherwise, impute missing values:
        - Numeric: mean, median, most_frequent, or knn
        - Categorical/boolean: mode
    Returns the processed DataFrame and a summary dict.
    """
    total_rows = len(df)
    rows_with_na = df.isnull().any(axis=1).sum()
    percent_missing = rows_with_na / total_rows
    summary = {'dropped_rows': 0, 'imputed_columns': []}

    if percent_missing < 0.01:
        df = df.dropna()
        summary['dropped_rows'] = rows_with_na
        return df, summary

    if strategy == 'knn':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        summary['imputed_columns'].extend([(col, 'knn', f'n_neighbors={n_neighbors}') for col in numeric_cols if df[col].isnull().sum() == 0])
        # For non-numeric, fallback to mode
        for col in df.columns:
            if col not in numeric_cols and df[col].isnull().any():
                mode = df[col].mode(dropna=True)
                fill_value = mode[0] if not mode.empty else 'missing'
                df[col] = df[col].fillna(fill_value)
                summary['imputed_columns'].append((col, 'mode', fill_value))
        return df, summary

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == 'mean':
                    fill_value = df[col].mean()
                    method = 'mean'
                elif strategy == 'median':
                    fill_value = df[col].median()
                    method = 'median'
                elif strategy == 'most_frequent':
                    fill_value = df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else df[col].median()
                    method = 'most_frequent'
                else:
                    fill_value = df[col].median()
                    method = 'median'
                df[col] = df[col].fillna(fill_value)
                summary['imputed_columns'].append((col, method, fill_value))
            elif pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                mode = df[col].mode(dropna=True)
                fill_value = mode[0] if not mode.empty else 'missing'
                df[col] = df[col].fillna(fill_value)
                summary['imputed_columns'].append((col, 'mode', fill_value))
            else:
                # For other types, just fill with a placeholder
                df[col] = df[col].fillna('missing')
                summary['imputed_columns'].append((col, 'placeholder', 'missing'))
    return df, summary
