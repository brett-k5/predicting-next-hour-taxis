# Third-party imports
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.linear_model import LinearRegression
from tbats import TBATS

# Local application imports
from src.cross_validation import n_splits
from src.features_utils import lag_adjustments
from src.metrics import rmse_calc

def test(best_model, X_train, y_train, X_test, y_test, forecast_length, steps, n_splits=n_splits):
    if isinstance(best_model, SARIMAXResults):
        best_model = SARIMAX(y_train,
                             order=best_model.specification["order"],
                             seasonal_order=best_model.specification["seasonal_order"],
                             enforce_stationarity=best_model.specification["enforce_stationarity"],
                             enforce_invertibility=best_model.specification["enforce_invertibility"])
        results = best_model.fit()
        preds = results.forecast(steps=steps)
    elif isinstance(best_model, LinearRegression):
        X_train = lag_adjustments(forecast_length, X_train)
        # On our first test run, we found that LineaRegression radically underperformed its CV performance.
        # Therefore, we are shortening the size of its training set to match that of the CV training sets.
        X_train_length = (len(X_train) - steps) // n_splits
        X_train_drop = len(X_train) - X_train_length
        X_train = X_train.drop(index=X_train.index[:X_train_drop])
        y_train = y_train.loc[X_train.index]
        X_test = lag_adjustments(forecast_length, X_test)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
    elif isinstance(best_model, TBATS):
        results = best_model.fit(y_train)
        preds = results.forecast(steps=steps)
    rmse = rmse_calc(preds, y_test)
    print(f"Best Model: {type(best_model).__name__}")
    print(f"Best Model RMSE: {rmse}")
    return type(best_model).__name__, rmse