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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# List of Indian stock symbols (you can customize these)
stock_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
start_date = '2020-01-01'
end_date = '2024-11-09'

# Folder path for storing data and models
folder_path = r'C:\Users\anish\Downloads\VIT Downloads\BSTS102P-Quantitative Skills Practice II\B1+TB1-FACE - FACE (APT) - ACAD\stock_data'
os.makedirs(folder_path, exist_ok=True)

def download_stock_data():
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

# Process data and train models for each stock
def process_and_train():
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

                # Adding moving averages
                stock_df['MA5'] = stock_df['Close'].rolling(window=5).mean()
                stock_df['MA10'] = stock_df['Close'].rolling(window=10).mean()
                stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()

                stock_df['Next_Close'] = stock_df['Close'].shift(-1)
                stock_df = stock_df.dropna()  # Drop rows with NaN values

                stock_features = stock_df[['Open', 'Low', 'High', 'Close', 'MA5', 'MA10', 'MA20']]
                stock_target = stock_df['Next_Close']

                # Scale the features and target
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

                n_steps = 30  # Number of time steps for LSTM input
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

                # Compile the model
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss='mean_squared_error')

                # Train the model
                history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

                # Evaluate the model on test data
                y_test_pred_scaled = model.predict(X_test)
                y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled)
                y_test_actual = target_scaler.inverse_transform(y_test)

                r2 = r2_score(y_test_actual, y_test_pred)
                print(f"R² Score on test data for {file_name}: {r2:.4f}")

                # Predict the next day's closing price
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
                model_filename = os.path.join(folder_path, f'{file_name}_LSTM_model.h5')
                model.save(model_filename)
                print(f"Model saved as {model_filename}")

                # Save predictions to CSV
                results = pd.DataFrame({
                    'Date': [next_date],
                    'Predicted_Close': [predicted_price],
                    'R2_Score': [r2]
                })
                results_filename = os.path.join(folder_path, f'{file_name}_predictions.csv')
                results.to_csv(results_filename, index=False)
                print(f"Results saved to {results_filename}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Download and process the data
download_stock_data()
process_and_train()



# import pandas as pd
# from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from operator import itemgetter
# from langchain_core.runnables import RunnablePassthrough

# class StockAdvisor:
#     def __init__(self, csv_path):
#         # Load CSV data
#         self.df = pd.read_csv(csv_path)
        
#         # Setup the model
#         self.model = ChatOpenAI(
#             model="meta-llama/llama-3.2-3b-instruct:free",
#             openai_api_key="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",
#             openai_api_base="https://openrouter.ai/api/v1",
#         )
        
#         # Initialize workflow
#         self.workflow = StateGraph(state_schema=MessagesState)
#         self.setup_workflow()
        
#     def get_stock_data(self, query):
#         """Retrieve relevant stock data based on query"""
#         try:
#             # Basic search implementation - can be enhanced based on needs
#             if 'symbol' in query.lower():
#                 symbol = query.split()[-1].upper()
#                 return self.df[self.df['Symbol'] == symbol].to_dict('records')
#             else:
#                 # Return first 5 matches based on simple string matching
#                 return self.df[self.df.astype(str).apply(lambda x: x.str.contains(query, case=False)).any(axis=1)].head().to_dict('records')
#         except Exception as e:
#             return f"Error retrieving data: {str(e)}"

#     def setup_workflow(self):
#         # Create prompt template with context
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are a stock investment advisor who specializes in Indian markets. 
#             Analyze the provided stock data and give recommendations. 
#             Use the data provided in the context for your analysis.
#             If specific data is not available, mention that in your response.
#             End every message with foobar"""),
#             MessagesPlaceholder(variable_name="messages"),
#             ("system", "Context: {context}")
#         ])

#         # Define the model call function
#         def call_model(state):
#             # Get the latest query
#             query = state["messages"][-1].content
            
#             # Retrieve relevant stock data
#             context = self.get_stock_data(query)
            
#             # Generate response using prompt template
#             response = self.prompt.pipe(self.model).invoke({
#                 "messages": state["messages"],
#                 "context": str(context)
#             })
            
#             return {"messages": response}

#         # Set up the workflow graph
#         self.workflow.add_edge(START, "model")
#         self.workflow.add_node("model", call_model)
        
#         # Add memory
#         self.memory = MemorySaver()
#         self.app = self.workflow.compile(checkpointer=self.memory)

#     def run_conversation(self):
#         config = {"configurable": {"thread_id": "abc123"}}
        
#         print("Stock Advisor initialized. Type 'quit' to exit.")
#         print("You can ask about stocks using queries like:")
#         print("- Tell me about symbol AAPL")
#         print("- What are the best performing stocks?")
#         print("- Should I invest in technology sector?")
        
#         while True:
#             query = input("\nHuman: ")
            
#             if query.lower() == 'quit':
#                 break
                
#             input_messages = [HumanMessage(content=query)]
            
#             try:
#                 output = self.app.invoke({"messages": input_messages}, config)
#                 output["messages"][-1].pretty_print()
#             except Exception as e:
#                 print(f"Error: {str(e)}")
#                 print("Please try a different query.")

# # Usage
# if __name__ == "__main__":
#     # Initialize with your CSV file
#     advisor = StockAdvisor('portfolio_data.csv')
#     advisor.run_conversation()