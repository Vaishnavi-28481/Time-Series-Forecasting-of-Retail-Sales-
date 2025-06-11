# retail_sales_forecasting.py

# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Step 2: Load the dataset
# Replace 'retail_sales.csv' with your actual dataset filename
# Replace 'Date' with your actual date column name
df = pd.read_csv("C:/Users/dell/Downloads/retailsales.csv", parse_dates=['date'], index_col='date')

print("=== Dataset Info ===")
print(df.info())  # Check data types and null values

print("\n=== First 5 rows ===")
print(df.head())  # Preview first few rows

# Step 3: Visualize the original sales data
plt.figure(figsize=(10, 5))
plt.plot(df['value'], label='Original Sales')
plt.title('Retail Sales Over Time')
plt.xlabel('date')
plt.ylabel('value')
plt.legend()
plt.grid()
plt.show()

# Step 4: Decompose the time series to observe trend, seasonality, residuals
decompose_result = seasonal_decompose(df['value'], model='additive', period=12)  # monthly data has period=12
decompose_result.plot()
plt.show()

# Step 5: Check if the time series is stationary using Augmented Dickey-Fuller test
adf_result = adfuller(df['value'])
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] > 0.05:
    print("\nTime series is non-stationary. Differencing is needed.")
    # Step 6: Difference the series to make it stationary
    df_diff = df['value'].diff().dropna()

    adf_result_diff = adfuller(df_diff)
    print(f"ADF Statistic after differencing: {adf_result_diff[0]:.4f}")
    print(f"p-value after differencing: {adf_result_diff[1]:.4f}")

    data_for_model = df_diff
    d = 1  # Differencing order
else:
    print("\nTime series is stationary. No differencing needed.")
    data_for_model = df['value']
    d = 0

# Step 7: Build ARIMA model
# For simplicity, let's choose p=1, d from above, q=1
model = ARIMA(df['value'], order=(1, d, 1))
model_fit = model.fit()

print("\n=== ARIMA Model Summary ===")
print(model_fit.summary())

# Step 8: Forecast future 12 months sales
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)

# Plot actual vs forecasted sales
plt.figure(figsize=(10, 5))
plt.plot(df['value'], label='Actual Sales')
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='M')[1:]
plt.plot(forecast_index, forecast, label='Forecasted Sales', color='red')
plt.title('Retail Sales Forecast')
plt.xlabel('date')
plt.ylabel('value')
plt.legend()
plt.grid()
plt.show()

# Step 9: Evaluate forecast accuracy
# (Only possible if actual future data is available for last 12 months)
# Here we simulate this by splitting the dataset (train-test split)

# Example: Use last 12 months as test set
train = df['value'][:-12]
test = df['value'][-12:]

# Refit model on training data
model_train = ARIMA(train, order=(1, d, 1))
model_train_fit = model_train.fit()

# Forecast for test period
forecast_test = model_train_fit.forecast(steps=12)

# Calculate MAE and MAPE
mae = mean_absolute_error(test, forecast_test)
mape = mean_absolute_percentage_error(test, forecast_test)

print(f"\nForecast Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Step 10: Recommendations for inventory based on forecast
print("\n--- Inventory Recommendations ---")
print("1. Increase inventory during forecasted peak sales months.")
print("2. Reduce inventory during forecasted low sales periods.")
print("3. Use forecast accuracy metrics (MAE, MAPE) to understand confidence level.")
print("4. Continuously update model with new sales data to improve forecasts.")
