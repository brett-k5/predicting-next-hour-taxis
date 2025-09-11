# Third-party imports
import pandas as pd

def lag_adjustments(forecast_length: int, X: pd.DataFrame) -> pd.DataFrame:
    """
    Removes lag features from the input DataFrame to prevent data leakage 
    during forecasting.

    Lag features are removed based on the specified forecast horizon. 
    For example, if the model is forecasting 24 hours ahead, it should 
    not have access to lag features that correspond to hours within that 
    future window.

    Parameters:
        forecast_length (int): Number of hours ahead to forecast.
        X (pd.DataFrame): Input features, including lag features.

    Returns:
        pd.DataFrame: Input features with appropriate lag columns removed.
    """
    if forecast_length == 12 or forecast_length == 24:
        X = X.drop('lag_1', axis=1)
    elif forecast_length == 72:
        X = X.drop(columns=['lag_1', 'lag_24'])
    elif forecast_length == 168:
        X = X.drop(columns=['lag_1', 'lag_24', 'lag_72'])
    return X