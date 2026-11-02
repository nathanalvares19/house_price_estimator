# drop features
def drop_features(columns, X_train_fe, X_valid_fe, X_test_fe):
    X_train_fe = X_train_fe.drop(columns=columns)
    X_valid_fe = X_valid_fe.drop(columns=columns)
    X_test_fe = X_test_fe.drop(columns=columns)

    return X_train_fe, X_valid_fe, X_test_fe


# add features
def add_features_v1(X):
    X_mod = X.copy()

    X_mod["rooms_per_household"] = X_mod["total_rooms"] / X_mod["households"]
    X_mod["bedrooms_per_room"] = X_mod["total_bedrooms"] / X_mod["total_rooms"]
    X_mod["people_per_household"] = X_mod["population"] / X_mod["households"]

    return X_mod


# add features
def add_features_v2(X, X_og):
    X_mod = X.copy()

    X_mod["bedrooms_per_household"] = X_og["total_bedrooms"] / X_mod["households"]
    X_mod["rooms_per_person"] = X_mod["total_rooms"] / X_mod["population"]
    X_mod["income_squared"] = X_mod["median_income"] ** 2

    return X_mod
