from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# create encoder for area column
def encode_string(X_train, X_valid, X_test):
    encoder = ColumnTransformer(
        transformers=[
            (
                "ocean_proximity",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    feature_name_combiner=lambda feature, category: str(category),
                ),
                ["ocean_proximity"],
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    encoder.set_output(transform="pandas")

    X_train_encoded = encoder.fit_transform(X_train)
    X_valid_encoded = encoder.transform(X_valid)
    X_test_encoded = encoder.transform(X_test)

    return X_train_encoded, X_valid_encoded, X_test_encoded
