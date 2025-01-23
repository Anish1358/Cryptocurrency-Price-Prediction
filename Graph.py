import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the datasets for XOM and AAPL
file_path_xom = r'C:/Users/anish/Downloads/VIT Downloads/BSTS102P-Quantitative Skills Practice II/B1+TB1-FACE - FACE (APT) - ACAD/XOM.csv'
file_path_aapl = r'C:/Users/anish/Downloads/VIT Downloads/BSTS102P-Quantitative Skills Practice II/B1+TB1-FACE - FACE (APT) - ACAD/AAPL.csv'

df_xom = pd.read_csv(file_path_xom)
df_aapl = pd.read_csv(file_path_aapl)

df_xom['Date'] = pd.to_datetime(df_xom['Date'])
df_aapl['Date'] = pd.to_datetime(df_aapl['Date'])

# Keep only the Date, Open, and Close columns
df_xom = df_xom[['Date', 'Open', 'Close']]
df_aapl = df_aapl[['Date', 'Open', 'Close']]

# Step 3: Plot the opening and closing prices for both stocks
plt.figure(figsize=(12, 8))

# Plot XOM opening and closing prices
plt.plot(df_xom['Date'], df_xom['Open'], label='XOM Open', linestyle='--', color='blue')
plt.plot(df_xom['Date'], df_xom['Close'], label='XOM Close', linestyle='-', color='red')

# Plot AAPL opening and closing prices
plt.plot(df_aapl['Date'], df_aapl['Open'], label='AAPL Open', linestyle='--', color='green')
plt.plot(df_aapl['Date'], df_aapl['Close'], label='AAPL Close', linestyle='-', color='orange')

# Step 4: Customize the plot
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Opening and Closing Prices of XOM and AAPL Stocks')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Step 5: Display the plot
plt.show()
