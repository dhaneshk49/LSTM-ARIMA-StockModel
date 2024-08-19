import keras.models
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
import pickle
import os

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

tickers = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JNJ", 
    "JPM", "V", "PG", "NVDA", "UNH", "WMT", "MA", "DIS", "PYPL", "ADBE", 
    "NFLX", "INTC", "AZN", "HSBA", "ULVR", "BP", "GSK", "BATS", "DGE", 
    "VOD", "RIO", "RB", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", 
    "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", 
    "SBIN.NS", "BAJFINANCE.NS","^CNXIT","^CNXAUTO"
)

def format_ticker(symbol):
    return symbol 

class StockApp:
    def __init__(self):
        self.model_choice = None
        self.use_pretrained = None
        self.lstm_model = None
        self.arima_model = None
        self.Symbol = None
        self.info = None
        self.period = None
        self.display_info = False
        self.tickers = tickers
        self.lstm_model_path = "lstm_model.pkl"
        self.arima_model_path = "arima_model.pkl"

    def run(self):
        st.title('Stock Price Predictions Using Machine Learning')
        st.markdown("""
        Welcome to the Stock Prediction app! Here, you can analyze and forecast stock prices using **LSTM** (Long Short-Term Memory) and **ARIMA** (AutoRegressive Integrated Moving Average) models.
        """)
        st.markdown("""
        *Disclaimer:*

        *The information provided on this website, including any data that can be downloaded as CSV files, is intended solely for research purposes. It should **not** be used to make trading or investment decisions. The models, predictions, and data presented here are part of a research project and have not been tested or validated against industry standards. We do not guarantee the accuracy, completeness, or reliability of the information provided. Users are advised to consult with a certified financial advisor before making any financial decisions based on the information presented on this site.*
        """)


        self.Symbol = st.selectbox('Select your stock', options=tuple(self.tickers))
        self.info = yf.Ticker(format_ticker(self.Symbol)).info
        self.custom_sidebar()

        end_date = datetime.now()
        start_date = datetime(end_date.year - int(self.period), end_date.month, end_date.day)
        fin_data = yf.download(format_ticker(self.Symbol), start_date, end_date)

        # Display the line graph and historical data immediately after stock selection
        st.markdown(f"### Stock Closing Price Data for {self.info['longName']}")
        self.plot_close_price(fin_data)
        self.display_historical_data(fin_data)

        # The submit button will control the predictions and analysis
        if st.sidebar.button('Submit'):
            lstm_predictions_df, arima_predictions_df = None, None

            if self.model_choice == "LSTM" or self.model_choice == "Both":
                st.markdown("#### LSTM Prediction")
                self.display_lstm_hyperparameters()
                if self.use_pretrained and os.path.exists(self.lstm_model_path):
                    self.lstm_model = self.load_model(self.lstm_model_path)
                else:
                    self.lstm_model = self.train_lstm(fin_data)
                    self.save_model(self.lstm_model, self.lstm_model_path)
                lstm_predictions_df = self.forecast_lstm(fin_data, self.lstm_model, 30)
                self.download_predictions(lstm_predictions_df, "LSTM")
                self.analyze_predictions(lstm_predictions_df, "LSTM")

            if self.model_choice == "ARIMA" or self.model_choice == "Both":
                st.markdown("#### ARIMA Prediction")
                self.display_arima_hyperparameters()
                if self.use_pretrained and os.path.exists(self.arima_model_path):
                    self.arima_model = self.load_model(self.arima_model_path)
                else:
                    self.arima_model = self.train_arima(fin_data)
                    self.save_model(self.arima_model, self.arima_model_path)
                arima_predictions_df = self.forecast_arima(fin_data, self.arima_model, 30)
                self.download_predictions(arima_predictions_df, "ARIMA")
                self.analyze_predictions(arima_predictions_df, "ARIMA")

            # if self.model_choice == "Both":
            #     self.compare_models(fin_data)

    def custom_sidebar(self):
        sidebar = st.sidebar
        sidebar.subheader('Options')
        sidebar.markdown("Select the time period for historical data and the model you'd like to apply.")
        # period_options = {"6M": 1, "1Y": 2, "2Y": 3, "3Y": 4}
        period_options = {"6M": 1, "1Y": 2, "2Y": 3, "3Y": 4}
        selected_period = sidebar.selectbox(label='Select period:', options=list(period_options.keys()))
        self.period = period_options[selected_period]
        
        self.model_choice = sidebar.selectbox('Choose Model', options=["LSTM", "ARIMA", "Both"])
        self.use_pretrained = sidebar.checkbox('Use Pre-trained Model', False)

    def display_lstm_hyperparameters(self):
        st.markdown("""
        ### LSTM Model Hyperparameters
        
        **Model Architecture:**
        - **First LSTM Layer**: 128 units, `return_sequences=True`, `Dropout` of 0.2
        - **Second LSTM Layer**: 64 units, `return_sequences=True`, `Dropout` of 0.2
        - **Third LSTM Layer**: 32 units, `return_sequences=False`, `Dropout` of 0.2
        - **Dense Layer**: 25 units
        - **Output Layer**: 1 unit
        
        **Compilation:**
        - **Optimizer**: `adam`
        - **Loss Function**: `mean_squared_error`
        
        **Training:**
        - **Batch Size**: 32
        - **Epochs**: 50
        """)

    def display_arima_hyperparameters(self):
        st.markdown("""
        ### ARIMA Model Hyperparameters
        
        **Model Order (p, d, q):**
        - **p**: 2
        - **d**: 0
        - **q**: 2
        - **Seasonality**: None
        """)

    def plot_close_price(self, df):
        fig = make_subplots(rows=1, cols=1, print_grid=True)
        fig.update_layout(title=f"{self.info['longName']} - Historical Closing Prices")
        fig.add_trace(go.Line(x=df.index, y=df['Close'], line=dict(color='green')))
        fig.update_yaxes(range=[0, df['Close'].max()*1.1], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    def display_historical_data(self, df):
        st.markdown("### Historical Stock Data")
        st.dataframe(df)  # Display the DataFrame as a table
        st.download_button(
            label="Download CSV",
            data=df.to_csv().encode('utf-8'),
            file_name=f"{self.Symbol}_historical_data.csv",
            mime='text/csv'
        )

    def train_lstm(self, df: pd.DataFrame):
        st.markdown("Training LSTM model... This may take a few moments.")
        
        data = pd.DataFrame(df['Close'])
        data.fillna(0, inplace=True)
        close = data.values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close)

        train_size = int(0.95 * len(scaled_data))
        train_data = scaled_data[:train_size, :]
        test_data = scaled_data[train_size - 60:, :]

        x_train, x_test = [], []
        y_train, y_test = [], []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])

        y_test = close[train_size:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1)

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        st.write('Root Mean Squared Error (RMSE): ', np.sqrt(mean_squared_error(y_test, predictions)))

        train = data[:train_size]
        valid = data[train_size:]
        valid['Predictions'] = predictions

        fig = plt.figure(figsize=(16, 6))
        plt.title('LSTM Model - Actual vs Predicted Closing Prices')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price in INR', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        
        return model

    def forecast_lstm(self, df: pd.DataFrame, model, days: int):
        st.markdown(f"Forecasting the next {days} days with LSTM...")

        data = pd.DataFrame(df['Close'])
        data.fillna(0, inplace=True)
        close = data.values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close)

        last_60_days = scaled_data[-60:]
        predicted_prices = []

        for _ in range(days):
            last_60_days_input = last_60_days.reshape((1, last_60_days.shape[0], 1))
            predicted_price = model.predict(last_60_days_input)
            predicted_prices.append(predicted_price[0, 0])
            last_60_days = np.append(last_60_days[1:], predicted_price)

        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days+1)]

        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close': predicted_prices.flatten()
        })

        fig = plt.figure(figsize=(16, 6))
        plt.title(f'LSTM Forecast for the Next {days} Days')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price in INR', fontsize=18)
        plt.plot(df.index, df['Close'], label='Historical Close')
        plt.plot(future_dates, predicted_prices, label='LSTM Predictions', linestyle='--', color='red')
        plt.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

        return predictions_df

    def train_arima(self, df: pd.DataFrame):
        st.markdown("Training ARIMA model... Please wait.")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        data = pd.DataFrame(df['Close'])
        data.fillna(0, inplace=True)

        train_size = int(0.95 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        model = ARIMA(train_data, order=(2, 0, 2))
        model_fit = model.fit()

        start_index = len(train_data)
        end_index = len(train_data) + len(test_data) - 1
        predictions = model_fit.predict(start=start_index, end=end_index, dynamic=False)
        
        predictions.index = test_data.index
        
        rmse = np.sqrt(np.mean((predictions - test_data['Close']) ** 2))
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')

        fig = plt.figure(figsize=(16, 6))
        plt.title('ARIMA Model - Actual vs Predicted Closing Prices')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price in INR', fontsize=18)
        plt.plot(train_data['Close'], label='Train')
        plt.plot(test_data['Close'], label='Test')
        plt.plot(predictions, label='Predictions', color='red')
        plt.legend(loc='lower right')
        st.pyplot(fig)
        
        return model_fit

    def forecast_arima(self, df: pd.DataFrame, model_fit, days: int):
        st.markdown(f"Forecasting the next {days} days with ARIMA...")

        future_forecast = model_fit.get_forecast(steps=days)
        forecast_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days)
        forecast_values = future_forecast.predicted_mean

        predictions_df = pd.DataFrame({
            'Date': forecast_index,
            'Predicted Close': forecast_values
        })

        fig = plt.figure(figsize=(16, 6))
        plt.title(f'ARIMA Forecast for the Next {days} Days')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price in INR', fontsize=18)
        plt.plot(df.index, df['Close'], label='Historical Close')
        plt.plot(forecast_index, forecast_values, label='ARIMA Predictions', linestyle='--', color='blue')
        plt.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

        return predictions_df

    def analyze_predictions(self, predictions_df, model_name):
        start_price = predictions_df['Predicted Close'].iloc[0]
        end_price = predictions_df['Predicted Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100

        st.markdown(f"### {model_name} Model Analysis")

        if percent_change > 10:
            st.markdown("**Recommendation:** The stock is predicted to rise significantly. This could be a strong buying opportunity, but ensure you are aware of potential risks.")
        elif 5 < percent_change <= 10:
            st.markdown("**Recommendation:** The stock is predicted to grow steadily. Consider investing but remain cautious of market conditions.")
        elif 0 <= percent_change <= 5:
            st.markdown("**Recommendation:** The stock shows mild growth. It could be a safe investment, but the returns may be moderate.")
        elif -5 <= percent_change < 0:
            st.markdown("**Recommendation:** The stock is expected to decline slightly. It may be wise to hold off on investing or consider selling.")
        else:
            st.markdown("**Recommendation:** The stock is expected to decline significantly. Consider avoiding investment or selling to mitigate potential losses.")

    def compare_models(self, df):
        st.markdown("### Comparative Analysis between LSTM and ARIMA")

        if self.lstm_model and self.arima_model:
            lstm_rmse = self.evaluate_rmse(df, self.lstm_model, 'LSTM')
            arima_rmse = self.evaluate_rmse(df, self.arima_model, 'ARIMA')

            if lstm_rmse < arima_rmse:
                st.markdown("**Overall Recommendation:** The LSTM model has a lower RMSE, indicating it may be more accurate. Consider relying more on LSTM predictions.")
            else:
                st.markdown("**Overall Recommendation:** The ARIMA model has a lower RMSE, indicating it may be more reliable. Consider using ARIMA predictions for decision-making.")

    def evaluate_rmse(self, df, model, model_name):
        train_size = int(0.95 * len(df))
        test_data = df['Close'].values[train_size:]

        if model_name == 'LSTM':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
            x_test = []
            for i in range(60, len(scaled_data[train_size - 60:])):
                x_test.append(scaled_data[train_size - 60 + i: train_size + i, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
        else:
            predictions = model.predict(start=train_size, end=len(df) - 1, dynamic=False)

        rmse = np.sqrt(mean_squared_error(test_data, predictions.flatten()))
        st.write(f'RMSE for {model_name}: {rmse}')
        return rmse

    def download_predictions(self, predictions_df, model_name):
        st.markdown(f"### Download {model_name} Predictions")
        st.dataframe(predictions_df)
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {model_name} Predictions as CSV",
            data=csv,
            file_name=f"{self.Symbol}_{model_name}_predictions.csv",
            mime='text/csv'
        )

    def save_model(self, model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model

if __name__ == '__main__':
    app = StockApp()
    app.run()
