# default_config.py

DEFAULT_PREPROCESSING_CONFIG = {
    "eda": {
        "enabled": True,
        "pairplot": False  # If True, generates pairplots (can be slow for many features)
    },
    "imputation": {
        "enabled": True,
        "option": "median",  # Options: "mean", "median", "most_frequent" (mode), "knn" (not implemented)
        # - "mean": fill numeric cols with mean
        # - "median": fill numeric cols with median (default, robust to outliers)
        # - "most_frequent": fill with mode (for numeric/categorical)
        # - "knn": use KNN imputer (requires extra dependency, not implemented)
    },
    "encoding": {
        "enabled": True,
        "option": "onehot",  # Options: "onehot", "label"
        # - "onehot": one-hot encoding (creates dummy variables)
        # - "label": label encoding (integer encoding)
    },
    "scaling": {
        "enabled": True,
        "option": "standard",  # Options: "standard", "minmax", "robust", "auto"
        # - "standard": StandardScaler (mean=0, std=1)
        # - "minmax": MinMaxScaler (scale to [0,1])
        # - "robust": RobustScaler (robust to outliers)
        # - "auto": Chooses scaler based on skewness/outliers
    },
    "outlier_handling": {
        "enabled": True,
        "option": "iqr",  # Options: "cap", "remove", "flag", "iqr", "zscore", "none"
        # - "cap": Cap outliers at quantiles (Winsorization)
        # - "remove": Remove rows containing outliers
        # - "flag": Add boolean columns indicating outliers
        # - "iqr": Interquartile Range method (default, uses "cap")
        # - "zscore": Z-score method (uses "cap")
        # - "none": do not handle outliers
        "lower_quantile": 0.01,  # Quantile for lower outlier threshold
        "upper_quantile": 0.99   # Quantile for upper outlier threshold
    },
    "log_transform": {
        "enabled": False,  # Log-transform highly skewed features if True
        "skew_thresh": 1.0  # Threshold for skewness to apply log-transform
    },
    "quantile_transform": {
        "enabled": False,  # Apply quantile transformation to numeric features
        "output_distribution": "normal"  # "normal" or "uniform"
    },
    "power_transform": {
        "enabled": False,  # Apply power transformation (yeo-johnson, box-cox)
        "method": "yeo-johnson"  # "yeo-johnson" or "box-cox"
    },
    "missing_indicator": {
        "enabled": False  # Add missing indicator columns if True
    },
    "feature_selection": {
        "enabled": False,
        "method": "correlation",  # Options: "variance", "correlation", "model_importance", "none"
        # - "variance": VarianceThreshold
        # - "correlation": select top features by correlation with target
        # - "model_importance": select features by importance from e.g. RandomForest
        # - "none": no feature selection
        "top_k": 20  # Number of features to keep if feature selection is enabled
    },
    "feature_generation": {
        "enabled": False,
        "polynomial_features": {
            "enabled": False,
            "degree": 2,  # Degree of polynomial features
            "interaction_only": False,  # If True, only interaction terms (no powers)
            "include_bias": False  # Include bias (constant term)
        },
        "datetime_features": {
            "enabled": False  # Extract datetime parts like year, month, day if enabled
        },
        "text_features": {
            "enabled": False,
            "method": "tfidf"  # Options: "tfidf", "count" (not implemented)
            # - "tfidf": TF-IDF vectorization (not implemented)
            # - "count": Count vectorization (not implemented)
        },
        "drop_low_variance_categorical": {
            "enabled": False,
            "threshold": 0.95  # Drop categorical columns with >95% same value
        }
    },
    "target_transform": {
        "enabled": False,
        "option": "none"  # Options: "none", "log", "sqrt", "boxcox"
        # - "none": leave target as-is
        # - "log": log1p transform
        # - "sqrt": square root transform
        # - "boxcox": box-cox transform
    },
    "custom_steps": {
        "enabled": False,
        "scripts": []  # List of custom preprocessing scripts (strings of script names)
        # Example: ["normalize_data", "remove_leaky_features"]
        # (Not implemented: placeholder for user custom code)
    }
}
