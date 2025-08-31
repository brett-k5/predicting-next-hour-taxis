# Third-party imports
import pandas as pd

def lag_adjustments(forecast_length: int, X: pd.DataFrame) -> pd.DataFrame:
    # We can't pass our linear regression models lag features for times when the lag value would not occur until after the 
    # the model is supposed to be making its forecast - that would be data leakage.
    # Therefore, we need the below conditionals to drop lag features depending on the forecast length. 
    if forecast_length == 12 or forecast_length == 24:
        X = X.drop('lag_1', axis=1)
    elif forecast_length == 72:
        X = X.drop(columns=['lag_1', 'lag_24'])
    elif forecast_length == 168:
        X = X.drop(columns=['lag_1', 'lag_24', 'lag_72'])
    return X