from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AutoML:
    def __init__(self, seed: int, metric: str = "r2", config: dict = None):
        self.seed = seed
        self.metric = r2_score
        self.config = config or {
            "imputer": "simple",
            "scaler": "standard",
            "encoder": "onehot",
            "feature_selector": "none",
            "var_thresh": 0.0,
            "select_k": "all",
            "n_estimators": 100,
            "max_depth": None,
        }
        self._pipeline = None

    def _build_pipeline(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Imputation
        num_imputer = KNNImputer() if self.config["imputer"] == "knn" else SimpleImputer(strategy="mean")
        num_scaler = MinMaxScaler() if self.config["scaler"] == "minmax" else StandardScaler()
        num_pipeline = Pipeline([
            ("imputer", num_imputer),
            ("scaler", num_scaler),
        ])

        cat_imputer = SimpleImputer(strategy="most_frequent")
        if self.config["encoder"] == "ordinal":
            cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_pipeline = Pipeline([
            ("imputer", cat_imputer),
            ("encoder", cat_encoder),
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ], remainder="drop")

        # Feature selection step
        fs_type = self.config.get("feature_selector", "none")
        if fs_type == "var_thresh":
            feature_selector = VarianceThreshold(threshold=self.config.get("var_thresh", 0.0))
        elif fs_type == "select_k_best":
            feature_selector = SelectKBest(score_func=f_regression, k=self.config.get("select_k", "all"))
        elif fs_type == "model_rf":
            feature_selector = SelectFromModel(
                RandomForestRegressor(n_estimators=self.config.get("n_estimators", 100), random_state=self.seed),
                threshold="mean"
            )
        else:
            feature_selector = "passthrough"

        # --- Begin robust max_depth fixing block ---
        max_depth_val = self.config.get("max_depth", None)
        if isinstance(max_depth_val, str) and max_depth_val.lower() == "none":
            max_depth = None
        else:
            max_depth = max_depth_val if max_depth_val is None else int(max_depth_val)
        # --- End robust block ---

        model = RandomForestRegressor(
            n_estimators=self.config.get("n_estimators", 100),
            max_depth=max_depth,
            random_state=self.seed
        )

        self._pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("feature_selector", feature_selector),
            ("regressor", model),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, random_state=self.seed, test_size=0.2
        )
        self._build_pipeline(X_train)
        self._pipeline.fit(X_train, y_train)
        val_preds = self._pipeline.predict(X_val)
        val_score = self.metric(y_val, val_preds)
        logger.info(f"Validation score: {val_score:.4f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise ValueError("Pipeline not fitted")
        return self._pipeline.predict(X)
