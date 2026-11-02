from sklearn.model_selection import train_test_split


# split data
def split(data):
    X = data.drop(columns=["median_house_value"])
    y = data["median_house_value"]

    # training and test data
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # test and validation data
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.50, random_state=42
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
