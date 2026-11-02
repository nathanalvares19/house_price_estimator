from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd


# to find impact of each feature on baseline model performance (FOR A LINEAR REGRESSOR)
def feat_analyse(X_train_encoded, y_train, X_valid_encoded, y_valid):
    results = []

    for feature in [None, *X_train_encoded.columns]:
        columns = X_train_encoded.columns.tolist()

        if feature is not None:
            columns.remove(feature)

        lin_reg = LinearRegression()
        lin_reg.fit(X_train_encoded[columns], y_train)

        predictions = lin_reg.predict(X_valid_encoded[columns])

        results.append(
            {
                "dropped": feature if feature is not None else "None",
                "MAE": mean_absolute_error(y_valid, predictions),
                "RMSE": root_mean_squared_error(y_valid, predictions),
                "R2": r2_score(y_valid, predictions),
            }
        )

    results_df = pd.DataFrame(results).sort_values("RMSE")

    return results_df


# to find impact of each feature on baseline model performance (FOR A RANDOM FOREST)
def feat_analyse_forest(X_train_mod, y_train, X_valid_mod, y_valid):
    results = []

    for feature in [None, *X_train_mod.columns]:
        columns = X_train_mod.columns.tolist()

        if feature is not None:
            columns.remove(feature)

        forest = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        forest.fit(X_train_mod[columns], y_train)

        predictions = forest.predict(X_valid_mod[columns])

        results.append(
            {
                "dropped": feature if feature is not None else "None",
                "MAE": mean_absolute_error(y_valid, predictions),
                "RMSE": root_mean_squared_error(y_valid, predictions),
                "R2": r2_score(y_valid, predictions),
            }
        )

    results_df = pd.DataFrame(results).sort_values("RMSE")

    return results_df
