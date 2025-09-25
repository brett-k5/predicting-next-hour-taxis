# Standard library imports
import random
import warnings

# Suppress FutureWarnings 
warnings.filterwarnings("ignore", category=FutureWarning)

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from tbats import TBATS

# Local application imports
from src.test import naive_models, save_df, test
from src.model_io import rmse_comp
import src.pre_processing as pre



# Ensure all UserWarnings are always shown during execution (including repeated warnings)
warnings.simplefilter('always', UserWarning)

# Create a list of file paths for rmse all folds rmse scores for all forecast lengths and
# cv methods to be added to all_folds_rmse_dfs dictionary as values
paths_all_folds_rmse = ['cv_rmse_scores/cv_all_folds_rmse_scores/blocked_all_folds_rmse_1h.csv', 'cv_rmse_scores/cv_all_folds_rmse_scores/expanded_w_all_folds_rmse_1h.csv',
                       'cv_rmse_scores/cv_all_folds_rmse_scores/blocked_all_folds_rmse_12h.csv', 'cv_rmse_scores/cv_all_folds_rmse_scores/expanded_w_all_folds_rmse_12h.csv',
                       'cv_rmse_scores/cv_all_folds_rmse_scores/blocked_all_folds_rmse_24h.csv', 'cv_rmse_scores/cv_all_folds_rmse_scores/expanded_w_all_folds_rmse_24h.csv',
                       'cv_rmse_scores/cv_all_folds_rmse_scores/blocked_all_folds_rmse_72h.csv', 'cv_rmse_scores/cv_all_folds_rmse_scores/expanded_w_all_folds_rmse_72h.csv',
                       'cv_rmse_scores/cv_all_folds_rmse_scores/blocked_all_folds_rmse_168h.csv', 'cv_rmse_scores/cv_all_folds_rmse_scores/expanded_w_all_folds_rmse_168h.csv']
    
# Create a list of DataFrame names to be added to all_folds_rmse_dfs dictionary as keys
all_folds_rmse_names = ['df_hour_blocked', 'df_hour_exp_w',
                        'df_12_hours_blocked', 'df_12_hours_exp_w',
                        'df_one_day_blocked', 'df_one_day_exp_w',
                        'df_72_hours_blocked', 'df_72_hours_exp_w',
                        'df_one_week_blocked', 'df_one_week_exp_w']

# Initialize empty all_folds_rmse_dfs dict  
all_folds_rmse_dfs = {}

# Populate empty dict by assigning each df name in all_folds_rmse_names 
# its corresponding file path from the paths_all_folds_rmse list
for file_path, name in zip(paths_all_folds_rmse, all_folds_rmse_names):
    all_folds_rmse_dfs[name] = pd.read_csv(file_path)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    np.random.seed(12345)
    random.seed(12345)



    # ================================
    # HOUR LONG FORECAST
    # ================================

    # Return best model for hour long forecast using custom rmse_comp() function
    best_model_hour = rmse_comp(all_folds_rmse_dfs['df_hour_blocked'], all_folds_rmse_dfs['df_hour_exp_w'], 'hour', 1)

    # Return performance metrics and model type for the hour long forecast's best performing model
    best_model_hour_type, rmse_hour, normalized_rmse_hour, r2_score_hour = test(best_model_hour,
                                                                                pre.X_train_hour, 
                                                                                pre.y_train_hour, 
                                                                                pre.X_test_hour, 
                                                                                pre.y_test_hour,
                                                                                'one_hour',
                                                                                1)
    ( # Return all lag predictors forecasts for 1 hour forecast with custom naive_models() function
        lag_1_rmse_1h_set, lag_1_nrmse_1h_set, lag_1_r2_1h_set,
        lag_24_rmse_1h_set, lag_24_nrmse_1h_set, lag_24_r2_1h_set,
        lag_72_rmse_1h_set, lag_72_nrmse_1h_set, lag_72_r2_1h_set,
        lag_168_rmse_1h_set, lag_168_nrmse_1h_set, lag_168_r2_1h_set
    ) = naive_models(pre.y_train_hour, pre.y_test_hour)

    # Save DataFrame containing all lag predictor metrics and best model metrics for 1 hour forecast
    save_df(best_model_hour_type, 'hour',
    rmse_hour, normalized_rmse_hour, r2_score_hour,
    lag_168_rmse_1h_set, lag_168_nrmse_1h_set, lag_168_r2_1h_set,
    lag_72_rmse_1h_set, lag_72_nrmse_1h_set, lag_72_r2_1h_set,
    lag_24_rmse_1h_set, lag_24_nrmse_1h_set, lag_24_r2_1h_set,
    lag_1_rmse_1h_set, lag_1_nrmse_1h_set, lag_1_r2_1h_set)




    # ================================
    # 12 HOUR LONG FORECAST
    # ================================

    # Return best model for 12 hour long forecast using custom rmse_comp() function
    best_model_12_hours = rmse_comp(all_folds_rmse_dfs['df_12_hours_blocked'], all_folds_rmse_dfs['df_12_hours_exp_w'], '12_hours', 12)

    # Return performance metrics and model type for the 12 hour long forecast's best performing model
    best_model_12_hours_type, rmse_12_hours, normalized_rmse_12_hours, r2_score_12_hours = test(best_model_12_hours,
                                                                                                pre.X_train_12_hours,
                                                                                                pre.y_train_12_hours,
                                                                                                pre.X_test_12_hours,
                                                                                                pre.y_test_12_hours,
                                                                                                '12_hours',
                                                                                                12)
    ( # Return lag predictors' forecasts for all lag predictors predicting utilizing lags of 12 hours or greater  
        _, _, _, # You cannot have lag predictors for a 12 hour forecast that have a lag shorter than 12 hours. Therefore, 
        lag_24_rmse_12h_set, lag_24_nrmse_12h_set, lag_24_r2_12h_set, # we are assigning the lag_1 predictor's rmse scores to placeholder variables
        lag_72_rmse_12h_set, lag_72_nrmse_12h_set, lag_72_r2_12h_set,
        lag_168_rmse_12h_set, lag_168_nrmse_12h_set, lag_168_r2_12h_set
    ) = naive_models(pre.y_train_12_hours, pre.y_test_12_hours)
    
    # Save DataFrame containing metrics for all lag predictors utilizing lags 
    # greater than 12 hours and best model for 12 hour forecast
    save_df(best_model_12_hours_type, '12_hours',
    rmse_12_hours, normalized_rmse_12_hours, r2_score_12_hours,
    lag_168_rmse_12h_set, lag_168_nrmse_12h_set, lag_168_r2_12h_set,
    lag_72_rmse_12h_set, lag_72_nrmse_12h_set, lag_72_r2_12h_set,
    lag_24_rmse_12h_set, lag_24_nrmse_12h_set, lag_24_r2_12h_set)




    # ================================
    # DAY LONG FORECAST
    # ================================

    # Return best model for day long forecast using custom rmse_comp() function
    best_model_day = rmse_comp(all_folds_rmse_dfs['df_one_day_blocked'], all_folds_rmse_dfs['df_one_day_exp_w'], 'day', 24, override=True, override_model=LinearRegression)

    # Return performance metrics and model type for the day long forecast's best performing model
    best_model_day_type, rmse_day, normalized_rmse_day, r2_score_day = test(best_model_day,
                                                                            pre.X_train_day,
                                                                            pre.y_train_day,
                                                                            pre.X_test_day,
                                                                            pre.y_test_day,
                                                                            'one_day',
                                                                            24)
    ( # Return forecasts for all lag predictors utilizing lags of 24 hour or greater
        _, _, _, # You cannot have lag predictors for a 24 hour forecast that have a lag shorter than 24 hours. Therefore,
        lag_24_rmse_24h_set, lag_24_nrmse_24h_set, lag_24_r2_24h_set, # we are assigning the lag_1 predictor's rmse scores to placeholder variables
        lag_72_rmse_24h_set, lag_72_nrmse_24h_set, lag_72_r2_24h_set,
        lag_168_rmse_24h_set, lag_168_nrmse_24h_set, lag_168_r2_24h_set
    ) = naive_models(pre.y_train_day, pre.y_test_day)
    
    # Save DataFrame containing metrics for all lag predictors utilizing lags of 24 hour or greater
    # and best_model for one day forecast 
    save_df(best_model_day_type, 'day',
    rmse_day, normalized_rmse_day, r2_score_day,
    lag_168_rmse_24h_set, lag_168_nrmse_24h_set, lag_168_r2_24h_set,
    lag_72_rmse_24h_set, lag_72_nrmse_24h_set, lag_72_r2_24h_set,
    lag_24_rmse_24h_set, lag_24_nrmse_24h_set, lag_24_r2_24h_set)




    # ================================
    # 3 DAY FORECAST 
    # ================================

    # Return best model for 72 hour long forecast using custom rmse_comp() function
    best_model_3_days = rmse_comp(all_folds_rmse_dfs['df_72_hours_blocked'], all_folds_rmse_dfs['df_72_hours_exp_w'], '3_days', 72)

    # Return performance metrics and model type for the 72 hour long forecast's best performing model
    best_model_3_days_type, rmse_3_days, normalized_rmse_3_days, r2_score_3_days = test(best_model_3_days,
                                                                                        pre.X_train_3_days,
                                                                                        pre.y_train_3_days,
                                                                                        pre.X_test_3_days,
                                                                                        pre.y_test_3_days,
                                                                                        '3_days',
                                                                                        72)
    (# Return forecasts for all lag predictors utilizing lags of 72 hours or greater 
        _, _, _, # You cannot have lag predictors for a 72 hour forecast that have a lag
        _, _, _, # shorter than 72 hours. Therefore, we are assigning the rmse scores for these shorter lags to placeholder variables
        lag_72_rmse_72h_set, lag_72_nrmse_72h_set, lag_72_r2_72h_set,
        lag_168_rmse_72h_set, lag_168_nrmse_72h_set, lag_168_r2_72h_set
    ) = naive_models(pre.y_train_3_days, pre.y_test_3_days)

    # Save DataFrame containing metrics for all lag predictors utilizing lags of 72 hours or greater
    # and best_model for 72 hour forecast 
    save_df(best_model_3_days_type, '3_days',
    rmse_3_days, normalized_rmse_3_days, r2_score_3_days,
    lag_168_rmse_72h_set, lag_168_nrmse_72h_set, lag_168_r2_72h_set,
    lag_72_rmse_72h_set, lag_72_nrmse_72h_set, lag_72_r2_72h_set)


    

    # ================================
    # WEEK LONG FORECAST
    # ================================

    # Return best model for week long forecast using custom rmse_comp() function
    best_model_week = rmse_comp(all_folds_rmse_dfs['df_one_week_blocked'], all_folds_rmse_dfs['df_one_week_exp_w'], 'week', 168)

    # Return performance metrics and model type for the week long forecast's best performing model
    best_model_week_type, rmse_week, normalized_rmse_week, r2_score_week = test(best_model_week,
                                                                                pre.X_train_week,
                                                                                pre.y_train_week,
                                                                                pre.X_test_week,
                                                                                pre.y_test_week,
                                                                                'one_week',
                                                                                168)
    ( # Return forecasts for lag predictors utilizing lags of 168 hour or greater
        _, _, _, # You cannot have lag predictors for a 168 hour forecast that have a lag
        _, _, _, # shorter than 168 hours. Therefore, we are assigning the rmse scores 
        _, _, _, # for shorter lag outputs to these placeholder variables, since we do not intend to use them
        lag_168_rmse_168h_set, lag_168_nrmse_168h_set, lag_168_r2_168h_set
    ) = naive_models(pre.y_train_week, pre.y_test_week)

    # Save DataFrame containing metrics for all lag predictors utilizing lags of 168 hours or greater
    # and best_model for week long forecast 
    save_df(best_model_week_type, 'week',
    rmse_week, normalized_rmse_week, r2_score_week,
    lag_168_rmse_168h_set, lag_168_nrmse_168h_set, lag_168_r2_168h_set)



    
    
        
