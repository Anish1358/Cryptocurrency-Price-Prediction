
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import timedelta
import matplotlib.pyplot as plt
import base64
from io import BytesIO
# import mysql.connector as ms
from flask import Flask, render_template, request, redirect, session, url_for, jsonify

import pymysql

# Connect to the database using pymysql
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="ANish@25",
    database="stock"
)

# Check if the connection is successful
if conn.open:
    print("Successfully connected to the database.")
else:
    print("Failed to connect to the database.")

# Close the connection after use
conn.close()


# Close the connection after use

# if conn.open():
#     print("MySQL connection established.")
# else:
#     print("Failed to connect to MySQL.")
# mc = conn.cursor()
# # Connect to the database using the new user 'Anish'
# conn = ms.connect(
#     host="localhost",  # Host name (use 'localhost' if MySQL is installed locally)
#     user="Anish",      # The MySQL username you created
#     password="Anish@1234",  # The password for the new user
#     database="stocks"   # The database name you want to connect to
# )

# # Check if the connection is successful
# if conn.is_connected():
#     print("Successfully connected to the database.")
# else:
#     print("Failed to connect to the database.")

# # Close the connection after use
# conn.close()

# if conn.is_connected():
#     print("MySQL connection established.")
# else:
#     print("Failed to connect to MySQL.")
# mc = conn.cursor()

# Flask setup
app = Flask(__name__)
app.secret_key = os.urandom(24)

stock_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']

start_date = '2020-01-01'
end_date = '2024-11-10'

folder_path = r'C:\Users\anish\Downloads\VIT Downloads\BSTS102P-Quantitative Skills Practice II\B1+TB1-FACE - FACE (APT) - ACAD\stock_data'
os.makedirs(folder_path, exist_ok=True)

def download_stock_data():
    """Downloads stock data for the given symbols from Yahoo Finance."""
    for stock_symbol in stock_symbols:
        print(f"Downloading data for {stock_symbol}...")
        try:
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
            stock_data.reset_index(inplace=True)
            csv_file_path = os.path.join(folder_path, f'{stock_symbol}.csv')
            stock_data.to_csv(csv_file_path, index=False)
            print(f'Data for {stock_symbol} saved to {csv_file_path}')
        except Exception as e:
            print(f"Error downloading {stock_symbol}: {e}")

def process_and_train():
    """Process the data, train LSTM models, and save the models."""
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"\nProcessing {file_name}...\n")
            try:
                stock_df = pd.read_csv(file_path, encoding='utf-8')

                print(stock_df.head())
                print(stock_df.dtypes)

                if 'Date' in stock_df.columns:
                    stock_df['date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
                else:
                    raise ValueError(f"'Date' column not found in {file_name}")

                necessary_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in necessary_columns:
                    stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')

                stock_df = stock_df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])

                
                stock_df['MA5'] = stock_df['Close'].rolling(window=5).mean()
                stock_df['MA10'] = stock_df['Close'].rolling(window=10).mean()
                stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()

                stock_df['Next_Close'] = stock_df['Close'].shift(-1)
                stock_df = stock_df.dropna()  
                stock_features = stock_df[['Open', 'Low', 'High', 'Close', 'MA5', 'MA10', 'MA20']]
                stock_target = stock_df['Next_Close']

                
                feature_scaler = MinMaxScaler()
                stock_features_scaled = feature_scaler.fit_transform(stock_features)

                target_scaler = MinMaxScaler()
                stock_target_scaled = target_scaler.fit_transform(stock_target.values.reshape(-1, 1))

                def create_sequences(features, target, n_steps):
                    X, y = [], []
                    for i in range(len(features) - n_steps):
                        X.append(features[i:i + n_steps])
                        y.append(target[i + n_steps])
                    return np.array(X), np.array(y)

                n_steps = 30  
                X, y = create_sequences(stock_features_scaled, stock_target_scaled, n_steps)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create LSTM model
                model = Sequential([
                    LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dropout(0.1),
                    LSTM(units=64),
                    Dropout(0.1),
                    Dense(units=32, activation='relu'),
                    Dense(units=1)
                ])

                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss='mean_squared_error')

                history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

                y_test_pred_scaled = model.predict(X_test)
                y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled)
                y_test_actual = target_scaler.inverse_transform(y_test)

                r2 = r2_score(y_test_actual, y_test_pred)
                print(f"R² Score on test data for {file_name}: {r2:.4f}")

                end_index = len(stock_features_scaled) - 1
                start_index = end_index - n_steps + 1
                sequence = stock_features_scaled[start_index:end_index + 1]
                sequence = np.expand_dims(sequence, axis=0)

                predicted_scaled = model.predict(sequence)
                predicted_scaled_reshaped = predicted_scaled.reshape(-1, 1)
                predicted_price = target_scaler.inverse_transform(predicted_scaled_reshaped)[0][0]

                last_date = stock_df['date'].max()
                next_date = last_date + timedelta(days=1)

                print(f'Predicted closing price for {next_date.strftime("%d-%b-%Y")} in {file_name}: {predicted_price:.2f}')

                # Save the model
                model_filename = os.path.join(folder_path, f'{file_name.replace(".csv", "_model.h5")}')
                model.save(model_filename)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Flask routes
@app.route('/')
def main_page():
    return render_template("landing.html")

@app.route('/signup')
def signup_page():
    return render_template("signup.html")

@app.route('/dashboard', methods=['POST'])
def enter_details():
    if request.method == 'POST':
        uname = request.form.get('username')
        passwd = request.form.get('password')
        email = request.form.get('email')

        if not uname or not passwd or not email:
            err = "All fields are required."
            return render_template("signup.html", err=err)

        mc.execute("SELECT uname FROM users WHERE uname=%s", (uname,))
        result = mc.fetchall()
        conn.commit()

        if result:
            err = 'Username already exists'
            return render_template("get_started.html", err=err)
        else:
            mc.execute("INSERT INTO users (uname, passwd, email) VALUES (%s, %s, %s)", (uname, passwd, email))
            conn.commit()
            return render_template("option.html", result=result)

@app.route('/start/login')
def login_page():
    return render_template("login.html")

@app.route('/start/dashboard', methods=['POST'])
def dashboard_page():
    if request.method == 'POST':
        global uname
        uname = request.form['username']
        passwd = request.form['password']
        mc.execute("SELECT * FROM users WHERE uname=%s AND passwd=%s", (uname, passwd))
        result = mc.fetchall()
        conn.commit()

        if result != []:
            return render_template("option.html", result=result)
        else:
            err = "Invalid username or password!"
            return render_template("login.html", err=err)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    error = ""
    prediction = None

    if request.method == 'POST':
        symbol = request.form.get('symbol')

        if symbol not in stock_symbols:
            error = "Symbol not supported"
        else:
            prediction = predict_next_close(symbol)
            if 'error' in prediction:
                error = prediction['error']
            else:
                prediction = {
                    "symbol": prediction["symbol"],
                    "date": prediction["date"],
                    "predicted_price": prediction["predicted_price"]
                }

    return render_template("predict.html", error=error, prediction=prediction)

@app.route('/view-crypto', methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        pass
    return render_template("chat.html", crypto_symbols=stock_symbols)

def predict_next_close(symbol):
    try:
        model_filename = os.path.join(folder_path, f'{symbol}_model.h5')
        model = load_model(model_filename)
        print(f"Model for {symbol} loaded.")
        
       
        last_data = fetch_latest_data(symbol)  
        predicted_price = model.predict(last_data)
        
        return {"symbol": symbol, "date": last_data.date(), "predicted_price": predicted_price}
    except Exception as e:
        return {"error": str(e)}

def fetch_latest_data(symbol):
    """Fetch the latest data for the given symbol."""
    return np.random.rand(1, 30, 7)  

if __name__ == '__main__':
    app.run(debug=True)



# import pymysql

# # Connect to the database using pymysql
# conn = pymysql.connect(
#     host="localhost",
#     user="root",
#     password="ANish@25",
#     database="stock"
# )

# # Check if the connection is successful
# if conn.open:
#     print("Successfully connected to the database.")
# else:
#     print("Failed to connect to the database.")

# # Close the connection after use
# conn.close()

