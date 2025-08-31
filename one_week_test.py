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

print(f"length of y_test_week: {len(pre.y_test_week)}")
print(f"length of y_test_3_days: {len(pre.y_test_3_days)}")
print(f"length of y_test_day: {len(pre.y_test_day)}")   
print(f"length of y_test_12_hours: {len(pre.y_test_12_hours)}")
print(f"length of y_test_hour: {len(pre.y_test_hour)}")


best_model_week = rmse_comp(avg_rmse_dfs['df_one_week_blocked'], avg_rmse_dfs['df_one_week_exp_w'], 'week', 168)
best_model_week_type, rmse_week, normalized_rmse_week, r2_score_week = test(best_model_week,
                                                                            pre.X_train_week,
                                                                            pre.y_train_week,
                                                                            pre.X_test_week,
                                                                            pre.y_test_week,
                                                                            'one_week',
                                                                            168)