from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

from . import feat_engg as fe


def make_features(X, selected_columns):
    result = fe.add_features_v1(X)
    result = fe.add_features_v2(result, X)

    return result.reindex(columns=selected_columns, fill_value=0)


def tune_forest(X_train_raw, y_train, selected_columns, param_grid=None):
    numeric_columns = X_train_raw.select_dtypes(include="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "ocean",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    feature_name_combiner=lambda feature, category: str(category),
                ),
                ["ocean_proximity"],
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "features",
                FunctionTransformer(
                    make_features,
                    kw_args={"selected_columns": list(selected_columns)},
                ),
            ),
            (
                "forest",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    if param_grid is None:
        param_grid = {
            "forest__max_depth": [10, 20, None],
            "forest__min_samples_leaf": [1, 5],
            "forest__max_features": [0.7, 1.0],
        }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        },
        refit="RMSE",
        n_jobs=-1,
        pre_dispatch="n_jobs",
        verbose=2,
        error_score="raise",
    )

    search.fit(X_train_raw, y_train)

    return search
