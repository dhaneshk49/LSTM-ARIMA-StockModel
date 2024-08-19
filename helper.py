import datetime as dt
import pandas as pd
import streamlit as st
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
from GoogleNews import GoogleNews

googlenews = GoogleNews()
googlenews = GoogleNews(lang='en', region='IN')
googlenews = GoogleNews(encode='utf-8')


# Create function to fetch stock name and id
def get_stocks():
    stock_dict = {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "Amazon Inc.": "AMZN",
        "Alphabet Inc.": "GOOGL",
        "Facebook Inc.": "META",
        "Tesla Inc.": "TSLA",
        "Berkshire Hathaway Inc.": "BRK.B",
        "Johnson & Johnson": "JNJ",
        "JPMorgan Chase & Co.": "JPM",
        "Visa Inc.": "V",
        "Procter & Gamble Co.": "PG",
        "NVIDIA Corporation": "NVDA",
        "UnitedHealth Group Incorporated": "UNH",
        "Walmart Inc.": "WMT",
        "Mastercard Incorporated": "MA",
        "Disney (The Walt Disney Company)": "DIS",
        "PayPal Holdings Inc.": "PYPL",
        "Adobe Inc.": "ADBE",
        "Netflix Inc.": "NFLX",
        "Intel Corporation": "INTC",
        "AstraZeneca plc": "AZN",
        "HSBC Holdings plc": "HSBA",
        "Unilever PLC": "ULVR",
        "BP p.l.c.": "BP",
        "GlaxoSmithKline plc": "GSK",
        "British American Tobacco p.l.c.": "BATS",
        "Diageo plc": "DGE",
        "Vodafone Group Plc": "VOD",
        "Rio Tinto Group": "RIO",
        "Reckitt Benckiser Group plc": "RB",
        "Reliance Industries Limited": "RELIANCE.NS",
        "Tata Consultancy Services Limited": "TCS.NS",
        "HDFC Bank Limited": "HDFCBANK.NS",
        "Infosys Limited": "INFY.NS",
        "Hindustan Unilever Limited": "HINDUNILVR.NS",
        "ICICI Bank Limited": "ICICIBANK.NS",
        "Kotak Mahindra Bank Limited": "KOTAKBANK.NS",
        "Bharti Airtel Limited": "BHARTIARTL.NS",
        "State Bank of India": "SBIN.NS",
        "Bajaj Finance Limited": "BAJFINANCE.NS"
    }

    # Return the dictionary
    return stock_dict


# Create function to fetch periods and intervals
def periods_intervals():
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }

    # Return the dictionary
    return periods


# Function to fetch the stock info
def get_stock_info(stock_ticker):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_info = stock_data.info

    def get_field(data_dict, key):
        return data_dict.get(key, "N/A")

    stock_data_info = {
        "Basic Information": {
            'businessSummary': stock_data_info['longBusinessSummary'],
            "symbol": get_field(stock_data_info, "symbol"),
            "longName": get_field(stock_data_info, "longName"),
            "currency": get_field(stock_data_info, "currency"),
            "exchange": get_field(stock_data_info, "exchange"),
        },
        "Market Data": {
            "symbol": get_field(stock_data_info, "symbol"),
            "currentPrice": get_field(stock_data_info, "currentPrice"),
            "previousClose": get_field(stock_data_info, "previousClose"),
            "open": get_field(stock_data_info, "open"),
            "dayLow": get_field(stock_data_info, "dayLow"),
            "dayHigh": get_field(stock_data_info, "dayHigh"),
            "regularMarketPreviousClose": get_field(
                stock_data_info, "regularMarketPreviousClose"
            ),
            "regularMarketOpen": get_field(stock_data_info, "regularMarketOpen"),
            "regularMarketDayLow": get_field(stock_data_info, "regularMarketDayLow"),
            "regularMarketDayHigh": get_field(stock_data_info, "regularMarketDayHigh"),
            "fiftyTwoWeekLow": get_field(stock_data_info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": get_field(stock_data_info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage": get_field(stock_data_info, "fiftyDayAverage"),
            "twoHundredDayAverage": get_field(stock_data_info, "twoHundredDayAverage"),
        },
        "Volume and Shares": {
            "volume": get_field(stock_data_info, "volume"),
            "regularMarketVolume": get_field(stock_data_info, "regularMarketVolume"),
            "averageVolume": get_field(stock_data_info, "averageVolume"),
            "averageVolume10days": get_field(stock_data_info, "averageVolume10days"),
            "averageDailyVolume10Day": get_field(
                stock_data_info, "averageDailyVolume10Day"
            ),
            "sharesOutstanding": get_field(stock_data_info, "sharesOutstanding"),
            "impliedSharesOutstanding": get_field(
                stock_data_info, "impliedSharesOutstanding"
            ),
            "floatShares": get_field(stock_data_info, "floatShares"),
        },
        "Dividends and Yield": {
            "dividendRate": get_field(stock_data_info, "dividendRate"),
            "dividendYield": get_field(stock_data_info, "dividendYield"),
            "payoutRatio": get_field(stock_data_info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap": get_field(stock_data_info, "marketCap"),
            "enterpriseValue": get_field(stock_data_info, "enterpriseValue"),
            "priceToBook": get_field(stock_data_info, "priceToBook"),
            "debtToEquity": get_field(stock_data_info, "debtToEquity"),
            "grossMargins": get_field(stock_data_info, "grossMargins"),
            "profitMargins": get_field(stock_data_info, "profitMargins"),
        },
        "Financial Performance": {
            "totalRevenue": get_field(stock_data_info, "totalRevenue"),
            "revenuePerShare": get_field(stock_data_info, "revenuePerShare"),
            "totalCash": get_field(stock_data_info, "totalCash"),
            "totalCashPerShare": get_field(stock_data_info, "totalCashPerShare"),
            "totalDebt": get_field(stock_data_info, "totalDebt"),
            "earningsGrowth": get_field(stock_data_info, "earningsGrowth"),
            "revenueGrowth": get_field(stock_data_info, "revenueGrowth"),
            "returnOnAssets": get_field(stock_data_info, "returnOnAssets"),
            "returnOnEquity": get_field(stock_data_info, "returnOnEquity"),
        },
        "Cash Flow": {
            "freeCashflow": get_field(stock_data_info, "freeCashflow"),
            "operatingCashflow": get_field(stock_data_info, "operatingCashflow"),
        },
        "Analyst Targets": {
            "targetHighPrice": get_field(stock_data_info, "targetHighPrice"),
            "targetLowPrice": get_field(stock_data_info, "targetLowPrice"),
            "targetMeanPrice": get_field(stock_data_info, "targetMeanPrice"),
            "targetMedianPrice": get_field(stock_data_info, "targetMedianPrice"),
        },
    }

    # Return the stock data
    return stock_data_info


# Function to fetch the stock history
def stock_history(stock_ticker, period, interval):
    stock_data = yf.Ticker(stock_ticker)

    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    # Return the stock data
    return stock_data_history

def get_news(stock_ticker):
    st.write("***")
    st.subheader("News")
    st.write("")

    googlenews = GoogleNews()
    googlenews.search(stock_ticker)
    results = googlenews.results()
    
    news_data = []

    for news in results:
        # Append each news item as a dictionary to the list
        news_item = {
            "title": news['title'],
            "description": news['desc'],
            "link": news['link']
        }
        news_data.append(news_item)
        
        st.markdown(f"**{news['title']}**")
        st.write(f"{news['desc']}")
        st.write(f"Link: {news['link']}")
        st.write("***")
    return news_data



# Function to generate the stock prediction
def generate_stock_prediction(stock_ticker):
    try:
        stock_data = yf.Ticker(stock_ticker)
        stock_data_hist = stock_data.history(period="2y", interval="1d")
        stock_data_close = stock_data_hist[["Close"]]
        stock_data_close = stock_data_close.asfreq("D", method="ffill")
        stock_data_close = stock_data_close.ffill()
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]  # 90%
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]  # 10%
        model = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")

        predictions = model.predict(
            start=test_df.index[0], end=test_df.index[-1], dynamic=True
        )

        forecast = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=90),
            dynamic=True,
        )
        return train_df, test_df, forecast, predictions

    except:
        return None, None, None, None
