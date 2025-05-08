# Stock Market Analysis and Prediction App

This Streamlit app provides tools for stock market analysis and predictions using various machine learning models. It allows users to visualize stock data, generate predictions, and interact with pre-trained models such as ARIMA and LSTM.


## Website Screenshots

<img width="1511" alt="Landing Page" src="https://github.com/user-attachments/assets/d82e218a-fc02-4465-a4e7-7dbc70693e6a" />

<img width="1512" alt="Market Data" src="https://github.com/user-attachments/assets/3f6e2ea0-9a4f-4eba-9a85-5c1dce1d6eee" />

<img width="1512" alt="Technical Charts" src="https://github.com/user-attachments/assets/a11da2dd-24ae-44c1-8a63-8ca9f136dc94" />



<img width="1512" alt="LSTM predictions" src="https://github.com/user-attachments/assets/43e4b51d-5a89-49f4-b5c4-7c7758309e9b" />


<img width="1512" alt="ARIMA predictions" src="https://github.com/user-attachments/assets/ad74de84-32e1-440b-9692-29f4f5c1c75a" />



## Project Structure

- **Landing.py**: The main entry point of the application.
- **pages/**: Contains the different pages of the app.
  - **ML Predictions.py**: Page for generating stock price predictions using machine learning models.
  - **Stock_Data_and_Charts.py**: Page for visualizing stock data and generating charts.
- **helper.py**: Contains helper functions used across the app.
- **arima_model.pkl**: Pre-trained ARIMA model for stock price prediction.
- **lstm_model.pkl**: Pre-trained LSTM model for stock price prediction.
- **yfinance_extract.ipynb**: Jupyter notebook used for extracting and preprocessing stock data using the `yfinance` library.
- **requirements.txt**: Lists the dependencies required to run the app.
- **README.md**: Project documentation (this file).

## Installation

To run this app locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/username/stock-market-analysis-app.git
    cd stock-market-analysis-app
    ```

2. **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app:**
    ```bash
    streamlit run Landing.py
    ```

**Note:** This app requires Python 3.8 or higher.

## Usage

### Landing Page
- Start the app to load the landing page, which will guide you through the functionalities available in the app.

### Stock Data and Charts
- Navigate to the "Stock Data and Charts" page to visualize stock data over time.
- Upload a CSV file or select a stock symbol to fetch data using the `yfinance` library.

### ML Predictions
- Visit the "ML Predictions" page to use machine learning models like ARIMA and LSTM for stock price predictions.
- The pre-trained models are loaded from the `arima_model.pkl` and `lstm_model.pkl` files.

## Configuration

If the app requires any API keys or specific configuration:

1. Create a `.env` file in the root directory.
2. Add your configuration as environment variables:
    ```env
    API_KEY=your_api_key_here
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Add some feature"
    ```
4. Push to the branch:
    ```bash
    git push origin feature/your-feature-name
    ```

## Acknowledgments

- Thanks to [Streamlit](https://streamlit.io) for providing the framework to build this app.
- Data is fetched using the [yfinance](https://github.com/ranaroussi/yfinance) library.
