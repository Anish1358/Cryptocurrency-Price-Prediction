import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Convert series to supervised learning format
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # Combine all columns
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # Extract raw values
    raw_values = series
    raw_values = raw_values.reshape(len(raw_values), 1)
    # Transform into supervised learning problem X, y
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    supervised_values = supervised.values
    # Split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test

# Create and train the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# Make forecasts using LSTM model
def make_forecasts(model, train, test, n_lag, n_seq):
    forecasts = []
    # Reshape the train set for LSTM
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Fit the model
    model.fit(X, y, epochs=200, verbose=0)

    # Prepare the test set
    for i in range(len(test)):
        # Reshape the input for LSTM
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        X = X.reshape((1, X.shape[0], 1))
        
        # Make forecast
        forecast = model.predict(X, verbose=0)
        forecasts.append(forecast[0, 0])  # Store single predicted value

    # Create a matrix of forecasts for evaluation
    forecast_matrix = np.zeros((len(test), n_seq))
    for i in range(len(test)):
        for j in range(n_seq):
            forecast_matrix[i, j] = forecasts[i]

    return forecast_matrix

# Evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = test[:, (n_lag + i)]
        predicted = forecasts[:, i]  # Extract predictions for the i-th forecast step
        
        # Calculate RMSE
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        # Calculate MAE
        mae = mean_absolute_error(actual, predicted)
        # Calculate MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100  # MAPE as a percentage

        print(f't+{i + 1} RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%')
# Plot forecasts against the actual series
def plot_forecasts(series, forecasts, n_test):
    # Plot the entire dataset in blue
    plt.plot(series, color='blue', label='Actual')
    
    # Calculate the starting index for the forecasted points
    start_index = len(series) - n_test
    
    # Plot the forecasts in red
    for i in range(forecasts.shape[1]):  # Iterate over the forecast time steps
        xaxis = range(start_index + i, start_index + i + len(forecasts))  # x values for predictions
        yaxis = forecasts[:, i]  # y values for the i-th forecast step
        plt.plot(xaxis, yaxis, color='red', marker='o', label=f'Forecast t+{i+1}' if i == 0 else "")
    
    plt.legend()
    plt.show()
    
def returnArray():
    n_lag = 1
    n_seq = 3
    n_test = 3
    forecast_array = []
    filenames = np.array(['XOM'])
    
    # Load data once, including the date column
    df = pd.read_csv(r'C:/Users/anish/Downloads/VIT Downloads/BSTS102P-Quantitative Skills Practice II/B1+TB1-FACE - FACE (APT) - ACAD/XOM.csv')
    data = df.iloc[:, -2].values
    dates = pd.to_datetime(df.iloc[:, 0].values)  # Assuming the first column is the date

    # Prepare data
    train, test = prepare_data(data, n_test, n_lag, n_seq)

    # Create LSTM model
    model = create_lstm_model((n_lag, 1))  # Ensure the input shape is correct

    # Make forecasts
    forecasts = make_forecasts(model, train, test, n_lag, n_seq)  # Pass the model here

    # Store the forecast values
    for i in range(n_seq):
        forecast_value = forecasts[-1][i]  # Forecast for the last time step
        forecast_array.append(forecast_value)

    # Evaluate forecasts
    evaluate_forecasts(test, forecasts, n_lag, n_seq)

    # Plot forecasts
    plot_forecasts(data, forecasts, n_test)

    # Prepare dates for predictions
    last_date = dates[-n_test]  # Get the last date from the training set
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_seq + 1)]

    # Create a DataFrame for the predicted values with corresponding dates
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted': forecast_array
    })

    # Check if forecast_array is not empty before calculating min and max
    if forecast_array:
        mini = min(forecast_array)
        maxi = max(forecast_array)

        if maxi == mini:
            forecast_array = [5] * len(forecast_array)
        else:
            for i in range(len(forecast_array)):
                forecast_array[i] = ((forecast_array[i] + abs(mini)) / (maxi - mini))
                forecast_array[i] = round(forecast_array[i] * 10, 2)
    else:
        print("No forecast values were generated.")

    return forecast_df  # Return the DataFrame with dates and predicted values

print(returnArray())
