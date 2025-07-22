import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    QuantileTransformer, PowerTransformer, PolynomialFeatures
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from scipy import stats


def one_hot_encode(df, drop_first=True):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df_encoded, cat_cols


def standard_scale(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, numeric_cols


def log_transform_skewed(df, skew_thresh=1.0):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed = df[numeric_cols].apply(lambda x: x.skew()).abs()
    skewed_cols = skewed[skewed > skew_thresh].index.tolist()
    df_log = df.copy()
    for col in skewed_cols:
        df_log[col] = np.log1p(df_log[col])
    return df_log, skewed_cols


def add_missing_indicators(df):
    missing_cols = [col for col in df.columns if df[col].isnull().any()]
    df_ind = df.copy()
    indicator_cols = []
    for col in missing_cols:
        ind_col = f'{col}_was_missing'
        df_ind[ind_col] = df_ind[col].isnull()
        indicator_cols.append(ind_col)
    return df_ind, indicator_cols


def variance_threshold_selector(df, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df, []
    selected = selector.fit_transform(df[numeric_cols])
    selected_cols = numeric_cols[selector.get_support(indices=True)].tolist()
    df_reduced = df[selected_cols].copy()
    return df_reduced, selected_cols


def correlation_threshold_selector(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_cols = [col for col in df.columns if col not in to_drop]
    df_reduced = df[selected_cols].copy()
    return df_reduced, selected_cols


def model_based_selector(df, y, task='auto', top_n=20, random_state=42):
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numeric_cols].skew().abs()
    if (skewness > 1).any():
        print("Using RobustScaler due to high skewness or outliers.")
        return RobustScaler()
    else:
        print("Using StandardScaler.")
        return StandardScaler()


def apply_selected_scaler(df):
    scaler = select_scaler(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler


def scale_features_with_option(df, scaler_option='auto'):
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


def add_polynomial_features(df, degree=2, include_bias=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_features = poly.fit_transform(df[numeric_cols])
    feature_names = poly.get_feature_names_out(numeric_cols)
    df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    return df_poly, feature_names.tolist()


def transform_target(y, transform_type='log'):
    if transform_type == 'log':
        return np.log1p(y)
    elif transform_type == 'sqrt':
        return np.sqrt(y)
    elif transform_type == 'boxcox':
        return stats.boxcox(y + 1e-3)[0]
    else:
        raise ValueError(f"Unsupported target transform: {transform_type}")


def apply_quantile_transform(df, output_distribution='normal'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    transformer = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    df_trans = df.copy()
    df_trans[numeric_cols] = transformer.fit_transform(df[numeric_cols])
    return df_trans, transformer


def apply_power_transform(df, method='yeo-johnson'):
    if method not in ['yeo-johnson', 'box-cox']:
        raise ValueError("Only 'yeo-johnson' and 'box-cox' supported")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    pt = PowerTransformer(method=method)
    df_trans = df.copy()
    df_trans[numeric_cols] = pt.fit_transform(df[numeric_cols])
    return df_trans, pt


def drop_low_variance_categorical(df, threshold=0.95):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    drop_cols = [col for col in cat_cols if df[col].value_counts(normalize=True).max() > threshold]
    df_reduced = df.drop(columns=drop_cols)
    return df_reduced, drop_cols


def impute_missing_values(df, strategy='mean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy not in ['mean', 'median', 'most_frequent']:
        raise ValueError("Unsupported imputation strategy")
    imputer = SimpleImputer(strategy=strategy)
    df_copy = df.copy()
    df_copy[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df_copy, imputer
