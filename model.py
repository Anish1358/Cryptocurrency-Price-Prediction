# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Convert series to supervised learning format
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg

# # Prepare data for supervised learning
# def prepare_data(series, n_test, n_lag, n_seq):
#     raw_values = series.reshape(len(series), 1)
#     supervised = series_to_supervised(raw_values, n_lag, n_seq)
#     supervised_values = supervised.values
#     train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
#     return train, test

# # Create and train the LSTM model
# def create_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Make forecasts using LSTM model
# def make_forecasts(model, train, test, n_lag, n_seq):
#     forecasts = []
#     X, y = train[:, 0:n_lag], train[:, n_lag:]
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     model.fit(X, y, epochs=200, verbose=0)

#     for i in range(len(test)):
#         X, y = test[i, 0:n_lag], test[i, n_lag:]
#         X = X.reshape((1, X.shape[0], 1))
#         forecast = model.predict(X, verbose=0)
#         forecasts.append(forecast[0, 0])

#     forecast_matrix = np.zeros((len(test), n_seq))
#     for i in range(len(test)):
#         for j in range(n_seq):
#             forecast_matrix[i, j] = forecasts[i]

#     return forecast_matrix

# # Evaluate the RMSE for each forecast time step
# def evaluate_forecasts(test, forecasts, n_lag, n_seq):
#     for i in range(n_seq):
#         actual = test[:, (n_lag + i)]
#         predicted = forecasts[:, i]
#         rmse = math.sqrt(mean_squared_error(actual, predicted))
#         mae = mean_absolute_error(actual, predicted)
#         mape = np.mean(np.abs((actual - predicted) / actual)) * 100
#         print(f't+{i + 1} RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%')

# # Calculate accuracy
# def calculate_accuracy(y_true, y_pred):
#     TP = np.sum((y_true == 1) & (y_pred == 1))
#     TN = np.sum((y_true == 0) & (y_pred == 0))
#     FP = np.sum((y_true == 0) & (y_pred == 1))
#     FN = np.sum((y_true == 1) & (y_pred == 0))
    
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     return accuracy

# # Plot forecasts against the actual series
# def plot_forecasts(series, forecasts, n_test):
#     plt.plot(series, color='blue', label='Actual')
#     start_index = len(series) - n_test
#     for i in range(forecasts.shape[1]):
#         xaxis = range(start_index + i, start_index + i + len(forecasts))
#         yaxis = forecasts[:, i]
#         plt.plot(xaxis, yaxis, color='red', marker='o', label=f'Forecast t+{i+1}' if i == 0 else "")
#     plt.legend()
#     plt.show()

# def returnArray():
#     n_lag = 1
#     n_seq = 1
#     n_test = 1
#     forecast_array = []
    
#     # Load data
#     df = pd.read_csv(r'C:/Users/anish/Downloads/VIT Downloads/BSTS102P-Quantitative Skills Practice II/B1+TB1-FACE - FACE (APT) - ACAD/XOM.csv')
#     data = df.iloc[:, -2].values
#     dates = pd.to_datetime(df.iloc[:, 0].values)

#     # Prepare data
#     train, test = prepare_data(data, n_test, n_lag, n_seq)

#     # Create LSTM model
#     model = create_lstm_model((n_lag, 1))

#     # Make forecasts
#     forecasts = make_forecasts(model, train, test, n_lag, n_seq)

#     # Store the forecast values
#     for i in range(n_seq):
#         forecast_value = forecasts[-1][i]
#         forecast_array.append(forecast_value)

#     # Evaluate forecasts
#     evaluate_forecasts(test, forecasts, n_lag, n_seq)

#     # Calculate accuracy (dummy example for y_true and y_pred)
#     y_true = np.array([0, 1, 1, 0])  # Replace with actual values
#     y_pred = np.array([0, 1, 0, 0])  # Replace with predicted values
#     accuracy = calculate_accuracy(y_true, y_pred)
#     print(f'Accuracy: {accuracy:.2f}')

#     # Plot forecasts
#     plot_forecasts(data, forecasts, n_test)

#     # Prepare dates for predictions
#     last_date = dates[-n_test]
#     forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_seq + 1)]

#     # Create a DataFrame for the predicted values with corresponding dates
#     forecast_df = pd.DataFrame({
#         'Date': forecast_dates,
#         'Predicted': forecast_array
#     })

#     return forecast_df

# print(returnArray())import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def prepare_data(series, n_test, n_lag, n_seq):
    raw_values = series.reshape(len(series), 1)
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    supervised_values = supervised.values
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))  
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
def make_forecasts(model, train, test, n_lag, n_seq):
    forecasts = []
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model.fit(X, y, epochs=200, verbose=0)
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        X = X.reshape((1, X.shape[0], 1))
        forecast = model.predict(X, verbose=0)
        forecasts.append(forecast[0, 0])
    forecast_matrix = np.zeros((len(test), n_seq))
    for i in range(len(test)):
        for j in range(n_seq):
            forecast_matrix[i, j] = forecasts[i]
    return forecast_matrix
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = test[:, (n_lag + i)]
        predicted = forecasts[:, i]
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        print(f'Timestep {i+1} RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%')
def plot_forecasts(data, forecasts, n_test):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(data)), data, label='Actual Data')
    for i in range(forecasts.shape[1]):
        offset = len(data) - n_test + i
        plt.plot(range(offset, offset + len(forecasts)), forecasts[:, i], label=f'Forecast t+{i+1}')
    plt.legend()
    plt.show()
def returnArray():
    n_lag = 1
    n_seq = 2 
    n_test = 2 
    forecast_array = []
    df = pd.read_csv(r'C:/Users/anish/Downloads/VIT Downloads/BSTS102P-Quantitative Skills Practice II/B1+TB1-FACE - FACE (APT) - ACAD/XOM.csv')
    data = df.iloc[:, -2].values
    dates = pd.to_datetime(df.iloc[:, 0].values)
    train, test = prepare_data(data, n_test, n_lag, n_seq)
    model = create_lstm_model((n_lag, 1))
    forecasts = make_forecasts(model, train, test, n_lag, n_seq)
    evaluate_forecasts(test, forecasts, n_lag, n_seq)
    plot_forecasts(data, forecasts, n_test)
    last_date = dates[-n_test]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_seq + 1)]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted': forecasts[-1] 
    })
    return forecast_df
predictions = returnArray()
print(predictions)
