from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tbats import TBATS


model_lin = LinearRegression()

def sarima(y_train):
    model_sarima = SARIMAX(
    y_train,
    order=(7, 1, 0),
    seasonal_order=(1, 0, 0, 24),
    trend='n',
    enforce_stationarity=True,
    enforce_invertibility=True)
    return model_sarima 

# Define TBATS model with default settings (you can customize seasonal periods)
model_tbats = TBATS(
    seasonal_periods=[24, 48, 168],  # e.g., daily, 2-day, weekly seasonality in hourly data
    use_box_cox=True,                # apply Box-Cox transformation automatically
    use_trend=True,                  # model trend component
)