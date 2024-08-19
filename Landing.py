import streamlit as st

# Set page configuration with a title and favicon

st.set_page_config(
    page_title="InvestMate: Your Financial Forecasting Companion",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page Title
st.title("Welcome to InvestMate")

# About Section
st.write("""
## About InvestMate
Thank you for taking the time to explore our platform. **InvestMate** is designed to empower individuals from all walks of life to make informed investment decisions using the power of machine learning.

Whether you're a student just starting to learn about investing, a busy professional managing your portfolio, or a businessperson looking for new opportunities, our platform simplifies the complexities of the stock market.

### What You'll Find Here:
- **Landing Page**: Begin your journey here! This page provides an overview of our project, goals, and features of the platform. We also include a brief survey to gather your valuable feedback.
- **ML Strategies Page**: Learn about the machine learning strategies we employ. We break down complex algorithms into easy-to-understand concepts, showcasing how they can analyze market data and assist in making smarter investment decisions.
- **Stocks Information Page**: Access comprehensive data on individual stocks, including fundamental and technical analysis. Whether you're interested in company financials, market trends, or technical indicators, you'll find all the relevant information here.

Please use the sidebar to navigate through the different sections of the website.
""")

# Disclaimer Section
st.write("""
### Disclaimer
Please note that all tools and strategies provided on this website are tentative and subject to further refinement. The features available are intended solely for educational purposes and to gauge user interest in the concept. They are not yet suitable for making real investment decisions, and we encourage you to use them only as a learning resource.
""")

# Sidebar with navigation and resources
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the links below to explore the website:")

st.sidebar.markdown("### [Survey Form](https://example.com/survey_form) 📋")
st.sidebar.title("Further Reading")
st.sidebar.markdown("Enhance your understanding with these resources:")
st.sidebar.markdown("🔗 [Auto Regressive Integrated Moving Average (ARIMA)](https://medium.com/analytics-vidhya/arima-for-dummies-ba761d59a051)")
st.sidebar.markdown("🔗 [Long Short Term Memory (LSTM)](https://ai.plainenglish.io/recurrent-neural-networks-for-dummies-70991a87e5d7)")


# Optional: Adding a footer or horizontal separator for better visual appeal
st.markdown("<hr>", unsafe_allow_html=True)  # Adds a horizontal line separator

st.write("""
### Stay Informed
Don't forget to fill out the survey form to help us improve this platform. Your feedback is invaluable!
""")

# Add some spacing and a footer message
st.markdown("<br><br>", unsafe_allow_html=True)  # Adds some spacing

st.markdown("""
*InvestMate © 2024 | Your Financial Forecasting Companion.*
""")
