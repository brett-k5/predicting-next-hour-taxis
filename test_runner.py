import numpy as np
import pandas as pd
import random

# We need all three model types imported so that in case override=True in rmse comp, 
# we can pass any model we like. 
from sklearn.linear_model import LinearRegression 
from tbats import TBATS
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from src.test import test
from src.model_io import rmse_comp
import src.pre_processing as pre 


paths_avg_rmse = ['cv_rmse_scores/cv_avg_rmse_scores/blocked_avg_rmse_1h.csv', 'cv_rmse_scores/cv_avg_rmse_scores/expanded_w_avg_rmse_1h.csv',
                'cv_rmse_scores/cv_avg_rmse_scores/blocked_avg_rmse_12h.csv', 'cv_rmse_scores/cv_avg_rmse_scores/expanded_w_avg_rmse_12h.csv',
                'cv_rmse_scores/cv_avg_rmse_scores/blocked_avg_rmse_24h.csv', 'cv_rmse_scores/cv_avg_rmse_scores/expanded_w_avg_rmse_24h.csv',
                'cv_rmse_scores/cv_avg_rmse_scores/blocked_avg_rmse_72h.csv', 'cv_rmse_scores/cv_avg_rmse_scores/expanded_w_avg_rmse_72h.csv',
                'cv_rmse_scores/cv_avg_rmse_scores/blocked_avg_rmse_168h.csv', 'cv_rmse_scores/cv_avg_rmse_scores/expanded_w_avg_rmse_168h.csv']
    
avg_rmse_names = ['df_hour_blocked', 'df_hour_exp_w',
                'df_12_hours_blocked', 'df_12_hours_exp_w',
                'df_one_day_blocked', 'df_one_day_exp_w',
                'df_72_hours_blocked', 'df_72_hours_exp_w',
                'df_one_week_blocked', 'df_one_week_exp_w']

    
avg_rmse_dfs = {}

for file_path, name in zip(paths_avg_rmse, avg_rmse_names):
    avg_rmse_dfs[name] = pd.read_csv(file_path)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    np.random.seed(12345)
    random.seed(12345)

    
    best_model_hour = rmse_comp(avg_rmse_dfs['df_hour_blocked'], avg_rmse_dfs['df_hour_exp_w'], 'hour', 1)
    best_model_hour_type, rmse_hour, normalized_rmse_hour, r2_score_hour = test(best_model_hour,
                                                                                pre.X_train_hour, 
                                                                                pre.y_train_hour, 
                                                                                pre.X_test_hour, 
                                                                                pre.y_test_hour,
                                                                                'one_hour',
                                                                                1)


    best_model_12_hours = rmse_comp(avg_rmse_dfs['df_12_hours_blocked'], avg_rmse_dfs['df_12_hours_exp_w'], '12_hours', 12)
    best_model_12_hours_type, rmse_12_hours, normalized_rmse_12_hours, r2_score_12_hours = test(best_model_12_hours,
                                                                                                pre.X_train_12_hours,
                                                                                                pre.y_train_12_hours,
                                                                                                pre.X_test_12_hours,
                                                                                                pre.y_test_12_hours,
                                                                                                '12_hours',
                                                                                                12)

    best_model_day = rmse_comp(avg_rmse_dfs['df_one_day_blocked'], avg_rmse_dfs['df_one_day_exp_w'], 'day', 24)
    best_model_day_type, rmse_day, normalized_rmse_day, r2_score_day = test(best_model_day,
                                                                            pre.X_train_day,
                                                                            pre.y_train_day,
                                                                            pre.X_test_day,
                                                                            pre.y_test_day,
                                                                            'one_day',
                                                                            24)

    best_model_3_days = rmse_comp(avg_rmse_dfs['df_72_hours_blocked'], 
                                  avg_rmse_dfs['df_72_hours_exp_w'], 
                                  '3_days', 
                                  72, 
                                  override=True, 
                                  override_model=TBATS)
    best_model_3_days_type, rmse_3_days, normalized_rmse_3_days, r2_score_3_days = test(best_model_3_days,
                                                                                        pre.X_train_3_days,
                                                                                        pre.y_train_3_days,
                                                                                        pre.X_test_3_days,
                                                                                        pre.y_test_3_days,
                                                                                        '3_days',
                                                                                        72)

    best_model_week = rmse_comp(avg_rmse_dfs['df_one_week_blocked'], avg_rmse_dfs['df_one_week_exp_w'], 'week', 168)
    best_model_week_type, rmse_week, normalized_rmse_week, r2_score_week = test(best_model_week,
                                                                                pre.X_train_week,
                                                                                pre.y_train_week,
                                                                                pre.X_test_week,
                                                                                pre.y_test_week,
                                                                                'one_week',
                                                                                168)

    test_results = pd.DataFrame({
    f"best_hour_model: {best_model_hour_type}": [rmse_hour, normalized_rmse_hour, r2_score_hour],
    f"best_12_hour_model: {best_model_12_hours_type}": [rmse_12_hours, normalized_rmse_12_hours, r2_score_12_hours],
    f"best_day_model: {best_model_day_type}": [rmse_day, normalized_rmse_day, r2_score_day],
    f"best_3_day_model: {best_model_3_days_type}": [rmse_3_days, normalized_rmse_3_days, r2_score_3_days],
    f"best_week_model: {best_model_week_type}": [rmse_week, normalized_rmse_week, r2_score_week]
}, index=["rmse", "nrmse", "r2"])

    test_results.to_csv('test_results.csv', index=False)
     
    
        
