import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold


def one_hot_encode(df, drop_first=True):
    """
    One-hot encode categorical columns. Returns new DataFrame and list of encoded columns.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df_encoded, cat_cols


def standard_scale(df):
    """
    Standard scale numeric columns. Returns new DataFrame and list of scaled columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, numeric_cols


def log_transform_skewed(df, skew_thresh=1.0):
    """
    Log-transform numeric columns with skewness above threshold. Returns new DataFrame and list of transformed columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed = df[numeric_cols].apply(lambda x: x.skew()).abs()
    skewed_cols = skewed[skewed > skew_thresh].index.tolist()
    df_log = df.copy()
    for col in skewed_cols:
        # Add 1 to avoid log(0)
        df_log[col] = np.log1p(df_log[col])
    return df_log, skewed_cols


def add_missing_indicators(df):
    """
    Add boolean columns indicating missing values for each original column.
    Returns new DataFrame and list of indicator columns.
    """
    missing_cols = [col for col in df.columns if df[col].isnull().any()]
    df_ind = df.copy()
    indicator_cols = []
    for col in missing_cols:
        ind_col = f'{col}_was_missing'
        df_ind[ind_col] = df_ind[col].isnull()
        indicator_cols.append(ind_col)
    return df_ind, indicator_cols


def variance_threshold_selector(df, threshold=0.0):
    """
    Remove features with variance below the threshold.
    Returns reduced DataFrame and list of selected features.
    """
    selector = VarianceThreshold(threshold=threshold)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df, []
    selected = selector.fit_transform(df[numeric_cols])
    selected_cols = numeric_cols[selector.get_support(indices=True)].tolist()
    df_reduced = df[selected_cols].copy()
    return df_reduced, selected_cols


def correlation_threshold_selector(df, threshold=0.95):
    """
    Remove one of each pair of features with correlation above the threshold.
    Returns reduced DataFrame and list of selected features.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_cols = [col for col in df.columns if col not in to_drop]
    df_reduced = df[selected_cols].copy()
    return df_reduced, selected_cols


def model_based_selector(df, y, task='auto', top_n=20, random_state=42):
    """
    Select top_n features based on feature importances from a RandomForest model.
    task: 'auto', 'classification', or 'regression'
    Returns reduced DataFrame and list of selected features.
    """
    X = df.select_dtypes(include=[np.number])
    if task == 'auto':
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            task = 'regression'
        else:
            task = 'classification'
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected_cols = importances.sort_values(ascending=False).head(top_n).index.tolist()
    df_reduced = df[selected_cols].copy()
    return df_reduced, selected_cols


def select_scaler(df):
    """
    Selects RobustScaler if any numeric feature has |skewness| > 1, else StandardScaler.
    Returns the scaler instance.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numeric_cols].skew().abs()
    if (skewness > 1).any():
        print("Using RobustScaler due to high skewness or outliers.")
        return RobustScaler()
    else:
        print("Using StandardScaler.")
        return StandardScaler()

def apply_selected_scaler(df):
    """
    Selects and applies the appropriate scaler to numeric columns.
    Returns the scaled DataFrame and the scaler used.
    """
    scaler = select_scaler(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler

def scale_features_with_option(df, scaler_option='auto'):
    """
    Scales numeric columns using the specified scaler option:
    - 'auto': RobustScaler if any |skewness| > 1, else StandardScaler
    - 'standard': StandardScaler
    - 'robust': RobustScaler
    - 'minmax': MinMaxScaler
    Returns the scaled DataFrame and the scaler used.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if scaler_option == 'auto':
        scaler = select_scaler(df)
    elif scaler_option == 'standard':
        scaler = StandardScaler()
    elif scaler_option == 'robust':
        scaler = RobustScaler()
    elif scaler_option == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler option: {scaler_option}")
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler 