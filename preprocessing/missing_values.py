import pandas as pd
import numpy as np

def handle_missing_values(df):
    """
    Handles missing values in a DataFrame:
    - If rows with missing values are <1% of total, drop those rows.
    - Otherwise, impute missing values:
        - Numeric: median
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

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                df[col] = df[col].fillna(median)
                summary['imputed_columns'].append((col, 'median', median))
            elif pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    fill_value = mode[0]
                else:
                    fill_value = np.nan
                df[col] = df[col].fillna(fill_value)
                summary['imputed_columns'].append((col, 'mode', fill_value))
            else:
                # For other types, just fill with a placeholder
                df[col] = df[col].fillna('missing')
                summary['imputed_columns'].append((col, 'placeholder', 'missing'))
    return df, summary 