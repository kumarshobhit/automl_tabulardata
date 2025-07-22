# default_config.py

DEFAULT_PREPROCESSING_CONFIG = {
    "eda": {
        "enabled": True,
        "pairplot": False  # If True, generates pairplots (can be slow for many features)
    },
    "imputation": {
        "enabled": True,
        "option": "median"  
        # Options for missing value imputation:
        # - "mean": fill numeric cols with mean
        # - "median": fill numeric cols with median (default, robust to outliers)
        # - "mode": fill categorical cols with mode
        # - "knn": use KNN imputer (requires extra dependency)
    },
    "encoding": {
        "enabled": True,
        "option": "onehot"  
        # Encoding categorical features:
        # - "onehot": one-hot encoding (creates dummy variables)
        # - "label": label encoding (integer encoding)
    },
    "scaling": {
        "enabled": True,
        "option": "standard"  
        # Feature scaling options:
        # - "standard": StandardScaler (mean=0, std=1)
        # - "minmax": MinMaxScaler (scale to [0,1])
        # - "robust": RobustScaler (robust to outliers)
    },
    "outlier_handling": {
        "enabled": True,
        "option": "iqr"  
        # Outlier detection/removal methods:
        # - "iqr": Interquartile Range method (default)
        # - "zscore": Z-score method
        # - "none": do not handle outliers
    },
    "feature_selection": {
        "enabled": False,
        "method": "correlation",  
        # Methods for feature selection:
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
            "method": "tfidf"  
            # Text feature extraction methods:
            # - "tfidf": TF-IDF vectorization
            # - "count": Count vectorization
        }
    },
    "target_encoding": {
        "enabled": False,
        "option": "none"  
        # Encoding targets:
        # - "none": leave target as-is
        # - "label": label encode target if categorical
        # - "binarize": binarize target (for binary classification)
    },
    "custom_steps": {
        "enabled": False,
        "scripts": []  
        # List of custom preprocessing scripts (strings of script names)
        # Example: ["normalize_data", "remove_leaky_features"]
    }
}
