# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_volatility(data, window=5):
#     return data.rolling(window=window).std()

# def calculate_var(data, confidence_level=0.95):
#     return np.percentile(data, (1 - confidence_level) * 100)

# def plot_risk(data, volatility, var_95):
#     plt.figure(figsize=(10, 6))
#     plt.plot(data.index, data, label='Price')
#     plt.plot(volatility.index, volatility, label='Rolling Volatility', color='orange')
#     plt.axhline(y=var_95, color='red', linestyle='--', label=f'VaR (95%) = {var_95:.2f}')
#     plt.title('Stock Price with Risk Metrics')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()

# # Main function to load data, calculate risk, and plot results
# def risk_prediction(file_path):
#     # Load data
#     df = pd.read_csv(file_path)
#     data = df.iloc[:, -2].values  # Adjust this for your data column
#     dates = pd.to_datetime(df.iloc[:, 0].values)
#     data_series = pd.Series(data, index=dates)

#     # Calculate rolling volatility
#     window = 5  # You can adjust the window size
#     volatility = calculate_volatility(data_series, window=window)
#     print(f'Rolling Volatility (Last 5 values):\n{volatility[-5:]}')

#     # Calculate VaR (Value at Risk)
#     confidence_level = 0.95  # 95% confidence level
#     var_95 = calculate_var(data_series, confidence_level=confidence_level)
#     print(f'Value at Risk (95% confidence level): {var_95:.2f}')

#     # Plot the risk metrics
#     plot_risk(data_series, volatility, var_95)

#     # Return calculated metrics
#     return volatility, var_95

# # If running the script directly, specify the file path here
# if __name__ == "__main__":
#     file_path = r'C:/Users/anish/Downloads/VIT Downloads/BSTS102P-Quantitative Skills Practice II/B1+TB1-FACE - FACE (APT) - ACAD/XOM.csv'
#     volatility, var_95 = risk_prediction(file_path)import pandas as pdimport os
import pandas as pd
file_path = r'C:/Users/anish/Downloads/VIT Downloads/BSTS102P-Quantitative Skills Practice II/B1+TB1-FACE - FACE (APT) - ACAD/XOM.csv'
df = pd.read_csv(file_path)
print("Columns in the dataset:", df.columns)
df['Close'] = df['Close'].ffill()
if len(df) < 26:
    raise ValueError("Not enough data points for MACD calculation.")
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal
def trading_signal(df):
    rsi = calculate_rsi(df).iloc[-1] if not calculate_rsi(df).empty else None
    macd, signal = calculate_macd(df)
    if rsi is None or macd.empty or signal.empty:
        return "Not enough data to calculate trading signal"
    if rsi < 30 and macd.iloc[-1] > signal.iloc[-1]:
        return "buy"
    elif rsi > 70 and macd.iloc[-1] < signal.iloc[-1]:
        return "sell"
    else:
        return "hold"

signal = trading_signal(df)
print(f"Trading Signal: {signal}")
