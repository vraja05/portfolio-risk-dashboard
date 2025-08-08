# Portfolio Risk Analytics Dashboard

A real-time portfolio risk management dashboard built with Python and Streamlit, designed to demonstrate quantitative finance and data analytics skills for investment analysis roles.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-risk-dashboard-w7fsqsbyfp8rwtqrtdfaw2.streamlit.app)

## 📊 Features

### Risk Metrics
- **Value at Risk (VaR)**: Historical VaR calculation at customizable confidence levels
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **Volatility Analysis**: Annual and daily volatility calculations

### Portfolio Analytics
- **Performance Tracking**: Real-time portfolio performance vs S&P 500 benchmark
- **Risk-Return Profiling**: Visualization of individual holdings and portfolio positioning
- **Correlation Analysis**: Inter-asset correlation matrix for diversification insights
- **Returns Distribution**: Statistical analysis of daily returns

### Interactive Features
- Customizable portfolio composition (tickers and weights)
- Adjustable analysis periods (1 month to 2 years)
- Dynamic risk-free rate and confidence level inputs
- Real-time data fetching from Yahoo Finance

## 🛠️ Technology Stack

- **Backend**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Data Source**: yfinance (Yahoo Finance API) - Live market data
- **Visualization**: Plotly
- **Web Framework**: Streamlit
- **Deployment**: Streamlit Cloud

## 📁 Project Structure

```
portfolio-risk-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
```

## 🚀 Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

## 📈 Usage

1. **Configure Portfolio**: Enter stock tickers and their respective weights in the sidebar
2. **Select Data Source**: Choose between live Yahoo Finance data or demo mode
3. **Set Parameters**: Adjust analysis period, risk-free rate, and VaR confidence level
4. **Analyze Results**: Explore different tabs for performance, risk metrics, and comparisons
5. **Export Insights**: Use the metrics for investment decision-making

## 📊 Data Sources

- **Live Mode**: Real-time market data from Yahoo Finance API (default)
- **Demo Mode**: Simulated data using Geometric Brownian Motion for testing
- Automatic failover to demo mode if API is unavailable (resilient design)

## 🎯 Key Metrics Explained

- **Sharpe Ratio**: Measures risk-adjusted returns. Values > 1 are considered good, > 2 excellent
- **Value at Risk (95%)**: The maximum expected loss with 95% confidence on a normal trading day
- **Maximum Drawdown**: Largest peak-to-trough decline, indicating worst-case historical scenario
- **Beta**: Systematic risk relative to market (S&P 500)

## 🔄 Future Enhancements

Given more development time, planned features include:
- Monte Carlo simulation for VaR calculation
- Factor model integration (Fama-French)
- Stress testing with historical crisis scenarios
- Options portfolio support
- Real-time alerts for risk threshold breaches
- PDF report generation
- Multi-asset class support (bonds, commodities)
- Machine learning-based risk predictions

## 🏢 Built for BlackRock Analyst Program

This project demonstrates:
- Understanding of portfolio risk management fundamentals
- Proficiency in quantitative finance calculations
- Data engineering and visualization skills
- Ability to create production-ready financial applications
- Knowledge of modern Python stack for finance

## 📝 License

MIT License - Feel free to use this project for educational purposes

## 👤 Author

Varun Raja  
[vraja005@gmail.com]  
[[LinkedIn Profile](https://www.linkedin.com/in/vinay-raja-5aaa0b24b/)]  
[GitHub](https://github.com/vraja05)
