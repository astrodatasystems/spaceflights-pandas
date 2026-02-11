import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets."""
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> RandomForestRegressor:
    """Trains the random forest regression model."""
    regressor = RandomForestRegressor(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        random_state=parameters["random_state"],
    )
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Calculates and logs the coefficient of determination."""
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}