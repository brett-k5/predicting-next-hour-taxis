import pandas as pd 
import numpy as np


from src.features_utils import lag_adjustments
from sklearn.model_selection import train_test_split


taxi_orders_df = pd.read_csv('taxi.csv', index_col=[0], parse_dates=[0])

hourly_orders = taxi_orders_df.resample('1h').sum()

# Adding lag features to capture seasonality
hourly_orders['lag_1'] = hourly_orders['num_orders'].shift(1)
hourly_orders['lag_24'] = hourly_orders['num_orders'].shift(24)
hourly_orders['lag_72'] = hourly_orders['num_orders'].shift(72)
hourly_orders['lag_168'] = hourly_orders['num_orders'].shift(168)

# Adding day of week and hour features
hourly_orders['hour'] = hourly_orders.index.hour
hourly_orders['day_of_week'] = hourly_orders.index.dayofweek
hourly_orders['day_of_month'] = hourly_orders.index.day

hourly_orders['hour_sin'] = np.sin(2 * np.pi * hourly_orders['hour'] / 24)
hourly_orders['hour_cos'] = np.cos(2 * np.pi * hourly_orders['hour'] / 24)

hourly_orders['dow_sin'] = np.sin(2 * np.pi * hourly_orders['day_of_week'] / 7)
hourly_orders['dow_cos'] = np.cos(2 * np.pi * hourly_orders['day_of_week'] / 7)

hourly_orders['dom_sin'] = np.sin(2 * np.pi * hourly_orders['day_of_month'] / 30.44)
hourly_orders['dom_cos'] = np.cos(2 * np.pi * hourly_orders['day_of_month'] / 30.44)


# Adding day of week and hour features
hourly_orders['hour'] = hourly_orders['hour'].astype(str)
hourly_orders['day_of_week'] = hourly_orders['day_of_week'].astype(str)
hourly_orders['day_of_month'] = hourly_orders['day_of_month'].astype(str)

hourly_orders = hourly_orders.dropna()

# Split data set into test set and training set for different forecast lengths 
train_hour, test_hour = train_test_split(hourly_orders, shuffle=False, test_size=24)
train_12_hours, test_12_hours = train_test_split(hourly_orders, shuffle=False, test_size=24)
train_day, test_day = train_test_split(hourly_orders, shuffle=False, test_size=28)
train_3_days, test_3_days = train_test_split(hourly_orders, shuffle=False, test_size=76)
train_week, test_week = train_test_split(hourly_orders, shuffle=False, test_size=172)

def features_target(train: pd.DataFrame, test: pd.DataFrame, forecast_length: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = train.drop('num_orders', axis=1)
    X_train = lag_adjustments(forecast_length, X_train)
    y_train = train['num_orders']
    X_test = test.drop('num_orders', axis=1)
    X_test = lag_adjustments(forecast_length, X_test)
    y_test = test['num_orders']
    return X_train, y_train, X_test, y_test 

X_train_hour, y_train_hour, X_test_hour, y_test_hour = features_target(train_hour, test_hour, 1)
X_train_12_hours, y_train_12_hours, X_test_12_hours, y_test_12_hours = features_target(train_12_hours, test_12_hours, 12)
X_train_day, y_train_day, X_test_day, y_test_day = features_target(train_day, test_day, 24)
X_train_3_days, y_train_3_days, X_test_3_days, y_test_3_days = features_target(train_3_days, test_3_days, 72)
X_train_week, y_train_week, X_test_week, y_test_week = features_target(train_week, test_week, 168)
