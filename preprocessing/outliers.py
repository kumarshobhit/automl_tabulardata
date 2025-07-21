import pandas as pd
import numpy as np

def handle_outliers(df, method='cap', lower_quantile=0.01, upper_quantile=0.99):
    """
    Handles outliers in numerical columns.
    method: 'cap' (default), 'remove', 'flag', or 'none'
    - 'cap': Cap outliers at the given quantiles (Winsorization)
    - 'remove': Remove rows containing outliers
    - 'flag': Add boolean columns indicating outliers
    - 'none': Do nothing
    Returns the processed DataFrame and a summary dict.
    """
    df = df.copy()
    summary = {'method': method, 'columns': []}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'cap':
        for col in numeric_cols:
            lower = df[col].quantile(lower_quantile)
            upper = df[col].quantile(upper_quantile)
            before = df[col].copy()
            df[col] = np.clip(df[col], lower, upper)
            n_capped = ((before < lower) | (before > upper)).sum()
            if n_capped > 0:
                summary['columns'].append({
                    'column': col,
                    'lower_threshold': lower,
                    'upper_threshold': upper,
                    'n_capped': int(n_capped)
                })
        return df, summary
    elif method == 'remove':
        mask = pd.Series([False] * len(df))
        for col in numeric_cols:
            lower = df[col].quantile(lower_quantile)
            upper = df[col].quantile(upper_quantile)
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            mask = mask | outlier_mask
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                summary['columns'].append({
                    'column': col,
                    'lower_threshold': lower,
                    'upper_threshold': upper,
                    'n_removed': int(n_outliers)
                })
        n_rows_before = len(df)
        df = df[~mask].reset_index(drop=True)
        summary['n_rows_removed'] = int(mask.sum())
        summary['n_rows_before'] = n_rows_before
        summary['n_rows_after'] = len(df)
        return df, summary
    elif method == 'flag':
        for col in numeric_cols:
            lower = df[col].quantile(lower_quantile)
            upper = df[col].quantile(upper_quantile)
            flag_col = f'{col}_is_outlier'
            df[flag_col] = (df[col] < lower) | (df[col] > upper)
            n_flagged = df[flag_col].sum()
            summary['columns'].append({
                'column': col,
                'lower_threshold': lower,
                'upper_threshold': upper,
                'n_flagged': int(n_flagged),
                'flag_column': flag_col
            })
        return df, summary
    elif method == 'none':
        summary['columns'] = []
        return df, summary
    else:
        raise ValueError(f"Unknown method: {method}") 