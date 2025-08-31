🚖 **predicting-next-hour-taxi-orders**  

📋 **Project Overview**  
This project aims to forecast the number of taxi orders for the next hour using time series modeling and feature engineering.

The data exhibits clear daily and weekly seasonality, which motivated the choice of lag features. ⏰📅

🔧 **Features Engineered**  
lag(1): The number of orders 1 hour ago ⏳

lag(24): The number of orders 24 hours ago (same hour previous day) 📆

lag(168): The number of orders 168 hours ago (same hour previous week) 📅

These lag features fully explained the improvement in RMSE over a naive baseline model that simply predicts the previous hour’s orders.

We also experimented with additional features such as:

Rolling mean 🔄

Cyclical encodings of the same hour in the previous week using sine and cosine transforms 🌙🌞

However, these additional features either had no effect or worsened model performance.

🤖 **Models Trained**  
Two SARIMA models (seasonal ARIMA) 📈

One Linear Regression model 📉

📊 **Results**  
The Linear Regression model outperformed the SARIMA models, demonstrating that simple lag-based features with a linear approach were more effective for this dataset.

On the test set:

Linear Regression model RMSE: 35.80 ✅

Naive model RMSE: 58.81 ❌

This shows a substantial improvement over the naive baseline that predicts the previous hour’s orders.

📁 **File Structure**
```
predicting_next_hour_taxis/
├── sprint_13_project.ipynb        # Main notebook with analysis and modeling  
├── taxi.csv                      # Dataset with taxi order data  
├── .ipynb_checkpoints/           # Jupyter notebook checkpoints  
├── anaconda_projects/
│   └── db/                      # Anaconda project database files  
├── .gitignore                   # Git ignore template file  
├── Requirements.txt             # Python package dependencies  
├── LICENSE                     # Project license file  
└── README.md                   # Project overview and documentation  
