import streamlit as st
import plotly.graph_objects as go
from GoogleNews import GoogleNews

googlenews = GoogleNews()
googlenews = GoogleNews(lang='en', region='UK')
googlenews = GoogleNews(encode='utf-8')

from helper import *

st.set_page_config(
    page_title="Stock Info",
    page_icon="",
)

st.sidebar.markdown("## **User Input Features**")

stock_dict = get_stocks()


st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))
stock_ticker = f"{stock_dict[stock]}"
st.sidebar.text_input(
    label="Ticker code", placeholder=stock_ticker, disabled=True
)

stock_ticker = f"{stock_dict[stock]}"

try:
    stock_data_info = get_stock_info(stock_ticker)
except:
    st.error("Error: Unable to fetch the stock data. Please try again later.")
    st.stop()


                
tab1, tab2, tab3 = st.tabs(["About and News","Fundamentals","Technicals"])

with tab1:    
    st.markdown("## About The Company ##")
    test = (stock_data_info["Basic Information"]["businessSummary"])
    st.markdown(test)

    get_news(stock_data_info["Market Data"]["symbol"])
     
with tab2:
    st.header("Fundamentals")
    # Create 2 columns
    col1, col2 = st.columns(2)

    # Row 1
    col1.dataframe(
        pd.DataFrame({"Issuer Name": [stock_data_info["Basic Information"]["longName"]]}),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame({"Symbol": [stock_ticker]}),
        hide_index=True,
        width=500,
    )

    st.markdown("## **Valuation and Ratios**")

    col1, col2 = st.columns(2)

    col1.dataframe(
        pd.DataFrame(
            {"Market Cap": [stock_data_info["Valuation and Ratios"]["marketCap"]]}
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Enterprise Value": [
                    stock_data_info["Valuation and Ratios"]["enterpriseValue"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )

    col1.dataframe(
        pd.DataFrame(
            {"Price to Book": [stock_data_info["Valuation and Ratios"]["priceToBook"]]}
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {"Debt to Equity": [stock_data_info["Valuation and Ratios"]["debtToEquity"]]}
        ),
        hide_index=True,
        width=500,
    )

    col1.dataframe(
        pd.DataFrame(
            {"Gross Margins": [stock_data_info["Valuation and Ratios"]["grossMargins"]]}
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {"Profit Margins": [stock_data_info["Valuation and Ratios"]["profitMargins"]]}
        ),
        hide_index=True,
        width=500,
    )

    st.markdown("## **Financial Performance**")

    col1, col2 = st.columns(2)
    col1.dataframe(
        pd.DataFrame(
            {"Total Revenue": [stock_data_info["Financial Performance"]["totalRevenue"]]}
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Revenue Per Share": [
                    stock_data_info["Financial Performance"]["revenuePerShare"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )

    col1, col2, col3 = st.columns(3)
    col1.dataframe(
        pd.DataFrame(
            {"Total Cash": [stock_data_info["Financial Performance"]["totalCash"]]}
        ),
        hide_index=True,
        width=300,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Total Cash Per Share": [
                    stock_data_info["Financial Performance"]["totalCashPerShare"]
                ]
            }
        ),
        hide_index=True,
        width=300,
    )
    col3.dataframe(
        pd.DataFrame(
            {"Total Debt": [stock_data_info["Financial Performance"]["totalDebt"]]}
        ),
        hide_index=True,
        width=300,
    )

    col1, col2 = st.columns(2)
    col1.dataframe(
        pd.DataFrame(
            {
                "Earnings Growth": [
                    stock_data_info["Financial Performance"]["earningsGrowth"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {"Revenue Growth": [stock_data_info["Financial Performance"]["revenueGrowth"]]}
        ),
        hide_index=True,
        width=500,
    )
    col1.dataframe(
        pd.DataFrame(
            {
                "Return on Assets": [
                    stock_data_info["Financial Performance"]["returnOnAssets"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Return on Equity": [
                    stock_data_info["Financial Performance"]["returnOnEquity"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )

    st.markdown("## **Cash Flow**")
    col1, col2 = st.columns(2)
    col1.dataframe(
        pd.DataFrame({"Free Cash Flow": [stock_data_info["Cash Flow"]["freeCashflow"]]}),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {"Operating Cash Flow": [stock_data_info["Cash Flow"]["operatingCashflow"]]}
        ),
        hide_index=True,
        width=500,
    )

    st.markdown("## **Analyst Targets**")
    col1, col2 = st.columns(2)

    col1.dataframe(
        pd.DataFrame(
            {"Target High Price": [stock_data_info["Analyst Targets"]["targetHighPrice"]]}
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {"Target Low Price": [stock_data_info["Analyst Targets"]["targetLowPrice"]]}
        ),
        hide_index=True,
        width=500,
    )

    col1.dataframe(
        pd.DataFrame(
            {"Target Mean Price": [stock_data_info["Analyst Targets"]["targetMeanPrice"]]}
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Target Median Price": [
                    stock_data_info["Analyst Targets"]["targetMedianPrice"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )
    
with tab3:
    periods = periods_intervals()
   
    col1, col2 = st.columns(2)

    with col1:
        period = st.selectbox(
            "Choose an Period",
            list(periods.keys())
        )

    with col2:
        interval = st.selectbox(
            "Choose an Interval",
            (periods[period])
        )
    stock_data = stock_history(stock_ticker, period, interval)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=stock_data.index,
                open=stock_data["Open"],
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"],
            )
        ]
    )

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("## **Market Data**")

    col1, col2 = st.columns(2)

    col1.dataframe(
        pd.DataFrame({"Current Price": [stock_data_info["Market Data"]["currentPrice"]]}),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame({"Previous Close": [stock_data_info["Market Data"]["previousClose"]]}),
        hide_index=True,
        width=500,
    )

    col1, col2, col3 = st.columns(3)

    col1.dataframe(
        pd.DataFrame({"Open": [stock_data_info["Market Data"]["open"]]}),
        hide_index=True,
        width=300,
    )
    col2.dataframe(
        pd.DataFrame({"Day Low": [stock_data_info["Market Data"]["dayLow"]]}),
        hide_index=True,
        width=300,
    )
    col3.dataframe(
        pd.DataFrame({"Open": [stock_data_info["Market Data"]["dayHigh"]]}),
        hide_index=True,
        width=300,
    )

    col1, col2 = st.columns(2)
    col1.dataframe(
        pd.DataFrame(
            {
                "Regular Market Previous Close": [
                    stock_data_info["Market Data"]["regularMarketPreviousClose"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {"Regular Market Open": [stock_data_info["Market Data"]["regularMarketOpen"]]}
        ),
        hide_index=True,
        width=500,
    )
    col1.dataframe(
        pd.DataFrame(
            {
                "Regular Market Day Low": [
                    stock_data_info["Market Data"]["regularMarketDayLow"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Regular Market Day High": [
                    stock_data_info["Market Data"]["regularMarketDayHigh"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )

    col1, col2, col3 = st.columns(3)

    col1.dataframe(
        pd.DataFrame(
            {"Fifty-Two Week Low": [stock_data_info["Market Data"]["fiftyTwoWeekLow"]]}
        ),
        hide_index=True,
        width=300,
    )
    col2.dataframe(
        pd.DataFrame(
            {"Fifty-Two Week High": [stock_data_info["Market Data"]["fiftyTwoWeekHigh"]]}
        ),
        hide_index=True,
        width=300,
    )
    col3.dataframe(
        pd.DataFrame(
            {"Fifty-Day Average": [stock_data_info["Market Data"]["fiftyDayAverage"]]}
        ),
        hide_index=True,
        width=300,
    )

    st.markdown("## **Volume and Shares**")

    col1, col2 = st.columns(2)

    col1.dataframe(
        pd.DataFrame({"Volume": [stock_data_info["Volume and Shares"]["volume"]]}),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Regular Market Volume": [
                    stock_data_info["Volume and Shares"]["regularMarketVolume"]
                ]
            }
        ),
        hide_index=True,
        width=500,
    )

    col1, col2, col3 = st.columns(3)

    col1.dataframe(
        pd.DataFrame(
            {"Average Volume": [stock_data_info["Volume and Shares"]["averageVolume"]]}
        ),
        hide_index=True,
        width=300,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Average Volume (10 Days)": [
                    stock_data_info["Volume and Shares"]["averageVolume10days"]
                ]
            }
        ),
        hide_index=True,
        width=300,
    )
    col3.dataframe(
        pd.DataFrame(
            {
                "Average Daily Volume (10 Day)": [
                    stock_data_info["Volume and Shares"]["averageDailyVolume10Day"]
                ]
            }
        ),
        hide_index=True,
        width=300,
    )

    col1.dataframe(
        pd.DataFrame(
            {
                "Shares Outstanding": [
                    stock_data_info["Volume and Shares"]["sharesOutstanding"]
                ]
            }
        ),
        hide_index=True,
        width=300,
    )
    col2.dataframe(
        pd.DataFrame(
            {
                "Implied Shares Outstanding": [
                    stock_data_info["Volume and Shares"]["impliedSharesOutstanding"]
                ]
            }
        ),
        hide_index=True,
        width=300,
    )
    col3.dataframe(
        pd.DataFrame(
            {"Float Shares": [stock_data_info["Volume and Shares"]["floatShares"]]}
        ),
        hide_index=True,
        width=300,
    )