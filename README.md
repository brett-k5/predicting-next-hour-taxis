# 🚕 Sweet Lift Taxi company has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the amount of taxi orders Sweet Lift will need in future hours


This time series forecasting project aims to predict the number of taxis the Sweet Lift Taxi Company will need for various forecast lengths:

- ⏱️ Next **1 hour**
- 🕛 Next **12 hours**
- 📆 Next **24 hours** (1 day)
- 📅 Next **72 hours** (3 days)
- 📈 Next **168 hours** (1 week)

We developed and compared multiple forecasting models to determine which would perform best for each forecast horizon, emphasizing **realistic retraining** and **robust cross-validation**.

---

## 🔧 Models & Evaluation

We compared the performance of:

- `LinearRegression` (with lag and datetime features)
- `TBATS` 
- `SARIMA`

All models were evaluated using both **Blocked Cross-Validation** (robustness) and **Expanding Window Cross-Validation** (realistic data flow). Performance metrics include:

- **RMSE** (Root Mean Squared Error)
- **NRMSE** (Normalized RMSE)
- **R²** (Coefficient of Determination)

### ✅ Best Performing Models by Forecast Horizon

| Forecast Horizon | Best Model         | RMSE    | NRMSE   | R²      |
|------------------|--------------------|---------|---------|---------|
| 1 hour           | LinearRegression   | 41.07   | 0.6745  | 0.5450  |
| 12 hours         | LinearRegression   | 37.37   | 0.6721  | 0.5482  |
| 1 day            | **LinearRegression*** | 43.12   | 0.6986  | 0.5119  |
| 3 days           | LinearRegression   | 50.82   | 0.8085  | 0.3464  |
| 1 week           | LinearRegression   | 50.03   | 0.7983  | 0.3627  |

> \*Note: The 1-day model was selected using an override explained below.

---

## ⚖️ Model Selection Logic

We used a custom `rmse_comp()` function that selects the best model based on:

1. If **the same model** had the **lowest RMSE** on both **blocked** and **expanding window** CV → ✅ That model is selected.
2. If **different models** performed best on each CV type → 🔁 The model with the **lowest STD across folds** (i.e., more stable) is selected.
3. An **override parameter** allows manual selection in edge cases.

### 🔄 Exception for 1-Day Forecast

TBATS slightly outperformed LinearRegression on **blocked CV**, which is more robust. However:

- LinearRegression outperformed TBATS on **expanding window CV**, where TBATS normally has the advantage (because it benefits disproportionately from increasing training data).
- The difference in performance on blocked CV was **very small**.
- LinearRegression is faster to train and easier to deploy.

➡️ So, even though TBATS was the code's default selection, we used the override parameter to select **LinearRegression** for the 1-day forecast.

---

## 🔁 Forecasting Strategy

To simulate a real-world production environment, we used **rolling horizon forecasting**. After each prediction, the model retrained with the newly available data, and then made the next prediction.

| Forecast Horizon | Rolling Steps | Model Re-training |
|------------------|---------------|--------------------|
| 1 hour           | 24            | 24 retrains        |
| 12 hours         | 12            | 12 retrains        |
| 1 day (24 hrs)   | 4             | 4 retrains         |
| 3 days (72 hrs)  | 4             | 4 retrains         |
| 1 week (168 hrs) | 4             | 4 retrains         |

> 📌 **Why more retrains for shorter horizons?**  
> The 1-hour and 12-hour forecasts produce fewer target values per retrain:
>
> - 1-hour → 1 target value per retrain  
> - 12-hour → 12 target values per retrain
>
> Using only 4 rolling steps would result in:
> - Just **5 total test values** for the 1-hour forecast
> - Only **16 test values** for the 12-hour forecast  
>
> 🔁 We increased the rolling steps to 24 and 12, respectively, to generate increased test data for more reliable evaluation.

---

## 🛠️ Model Features

### ✅ LinearRegression

Our top-performing model (for all forecast lengths) used:

- **Lag features**: `lag_1`, `lag_24`, `lag_72`, `lag_168`
- **Datetime features**:
  - **Categorical**: `hour`, `dayofweek`, `dayofmonth`
  - **Cyclical**: `sin_hour`, `cos_hour`, etc. (to capture periodicity)

### 🧠 TBATS & SARIMA

- Included for benchmarking.
- Did well on longer timeframes but required heavier computation.
- Did not outperform LinearRegression in key settings.

---

## 📊 Results Summary

- LinearRegression was a consistent performer across forecast lengths.
- **Short-term forecasts** (1, 12, 24 hrs) were highly accurate and outperformed naive baselines.
- **Longer-term forecasts** (3 days and 1 week) did not beat naive lag predictors, likely due to:
  - Increased forecast uncertainty
  - Simpler model capacity

---

## 📁 Project Structure
```
project-root/
├── notebooks/ # EDA and results analysis notebooks
│ ├── EDA.ipynb
│ └── results_and_analysis.ipynb
│
├── src/ # Source code for modeling, preprocessing, etc.
│ ├── cross_validation.py
│ ├── features_utils.py
│ ├── metrics.py
│ ├── models.py
│ ├── model_io.py
│ ├── pre_processing.py
│ └── test.py
│
├── models/ # Serialized model files
│ ├── best_model_hour_blocked.pkl
│ ├── best_model_hour_exp_w.pkl
│ └── ... (other forecast lengths)
│
├── cv_rmse_scores/ # RMSE scores from CV (averages and per fold)
│ ├── cv_avg_rmse_scores/
│ └── cv_fold_rmse_scores/
│
├── test_runner.py # Code to evaluate model on test sets
├── taxi.csv # Main dataset
├── README.md # Project documentation
├── Requirements.txt # Project dependencies
├── hour_test_results.csv # Test set results for 1-hour forecast
├── 12_hours_test_results.csv # Test set results for 12-hour forecast
├── day_test_results.csv # Test set results for 24-hour forecast
├── 3_days_test_results.csv # Test set results for 72-hour forecast
└── week_test_results.csv # Test set results for 168-hour forecast
```

## 🧠 Key Takeaways

- **Linear models with engineered features** can outperform more complex models in time series forecasting.
- **Rolling horizon retraining** ensures realistic and robust evaluation.
- Always interpret **cross-validation results** in the context of model behavior, training data size, and deployment needs.

---

## ⚙️ Setup / Running Notebooks

To run the Jupyter notebooks in this project, ensure you have the required packages installed. **Conda is recommended**, especially on Windows, but any Python virtual environment will work.

1. Create and activate your environment:

**Using Conda (recommended on Windows):**
```powershell
conda create --name project_name_env python=3.11
conda activate project_name_env
```

---

## 🧠 Authors

- Developed by Brett Kunkel | [www.linkedin.com/in/brett-kunkel](www.linkedin.com/in/brett-kunkel) | brttkunkel@gmail.com

---

## 📜 License

This project is licensed under the MIT License.