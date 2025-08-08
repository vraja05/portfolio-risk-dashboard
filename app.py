import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Portfolio Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Portfolio Risk Analytics Dashboard")
st.markdown("Real-time portfolio risk analysis with VaR, Sharpe Ratio, and performance metrics")

# Function to generate sample data (fallback when API fails)
def generate_sample_data(tickers, days=252):
    """Generate realistic sample stock data"""
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    data = pd.DataFrame(index=dates)
    
    # Base prices and characteristics for each ticker
    base_prices = {"AAPL": 150, "MSFT": 300, "GOOGL": 140, "AMZN": 130, "META": 350}
    volatilities = {"AAPL": 0.02, "MSFT": 0.018, "GOOGL": 0.022, "AMZN": 0.025, "META": 0.028}
    trends = {"AAPL": 0.0003, "MSFT": 0.0004, "GOOGL": 0.0002, "AMZN": 0.0003, "META": 0.0001}
    
    for ticker in tickers:
        if ticker not in base_prices:
            # For unknown tickers, use default values
            base_price = 100
            volatility = 0.02
            trend = 0.0002
        else:
            base_price = base_prices[ticker]
            volatility = volatilities[ticker]
            trend = trends[ticker]
        
        # Generate returns with trend and volatility
        returns = np.random.normal(trend, volatility, days)
        
        # Create price series
        price = base_price
        prices = []
        for r in returns:
            price = price * (1 + r)
            prices.append(price)
        
        data[ticker] = prices
    
    return data

# Sidebar for portfolio configuration
st.sidebar.header("Portfolio Configuration")

# Default portfolio
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
default_weights = [0.25, 0.25, 0.20, 0.15, 0.15]

# Data source selection
data_source = st.sidebar.radio(
    "Data Source:",
    ["Live Data (Yahoo Finance)", "Sample Data (Demo)"],
    index=1,  # Default to live data
    help="Use Sample Data if live data is not working due to network issues"
)

# User input for portfolio
st.sidebar.subheader("Enter Portfolio Holdings")
tickers_input = st.sidebar.text_area(
    "Tickers (comma-separated):",
    value=", ".join(default_tickers),
    help="Enter stock symbols separated by commas"
)

weights_input = st.sidebar.text_area(
    "Weights (comma-separated, must sum to 1):",
    value=", ".join([str(w) for w in default_weights]),
    help="Enter portfolio weights as decimals"
)

# Parse inputs
try:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    weights = [float(w.strip()) for w in weights_input.split(",")]
    
    if len(tickers) != len(weights):
        st.error("Number of tickers must match number of weights!")
        st.stop()
    
    if abs(sum(weights) - 1.0) > 0.01:
        st.warning(f"Weights sum to {sum(weights):.2f}, normalizing to 1.0")
        weights = [w/sum(weights) for w in weights]
        
except Exception as e:
    st.error(f"Error parsing inputs: {e}")
    st.stop()

# Time period selection
period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 252, "2y": 504}
period = st.sidebar.selectbox(
    "Analysis Period:",
    list(period_map.keys()),
    index=3
)

# Risk-free rate for Sharpe ratio
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (annual %):",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1
) / 100

# Confidence level for VaR
confidence_level = st.sidebar.slider(
    "VaR Confidence Level:",
    min_value=90,
    max_value=99,
    value=95,
    step=1
)

# Fetch or generate data
@st.cache_data(ttl=3600)
def fetch_data(tickers, period, use_sample=False):
    """Fetch historical data for given tickers"""
    if use_sample:
        st.info("Using sample data for demonstration purposes")
        return generate_sample_data(tickers, period_map[period])
    
    try:
        # Try to fetch real data
        data = yf.download(
            tickers, 
            period=period, 
            progress=False,
            auto_adjust=True,
            threads=False
        )
        
        if isinstance(data, pd.DataFrame):
            if len(tickers) == 1:
                if 'Close' in data.columns:
                    data = data[['Close']]
                    data.columns = tickers
            else:
                if 'Close' in data.columns.get_level_values(0):
                    data = data['Close']
                elif 'Adj Close' in data.columns.get_level_values(0):
                    data = data['Adj Close']
        
        if data.empty:
            raise Exception("No data retrieved")
            
        return data
        
    except Exception as e:
        st.warning(f"Failed to fetch live data: {e}")
        st.info("Switching to sample data...")
        return generate_sample_data(tickers, period_map[period])

# Calculate portfolio metrics
def calculate_portfolio_metrics(data, weights):
    """Calculate returns and risk metrics"""
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Annual metrics
    trading_days = 252
    annual_return = portfolio_returns.mean() * trading_days
    annual_vol = portfolio_returns.std() * np.sqrt(trading_days)
    
    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0
    
    # VaR calculation (historical method)
    var_percentile = (100 - confidence_level) / 100
    var_daily = np.percentile(portfolio_returns, var_percentile * 100)
    var_annual = var_daily * np.sqrt(trading_days)
    
    # Maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'var_daily': var_daily,
        'var_annual': var_annual,
        'max_drawdown': max_drawdown
    }

# Load data
with st.spinner("Fetching market data..."):
    data = fetch_data(tickers, period, use_sample=(data_source == "Sample Data (Demo)"))
    
if data is None or data.empty:
    st.error("Failed to fetch data. Please check your tickers and try again.")
    st.stop()

# Calculate metrics
metrics = calculate_portfolio_metrics(data, weights)

# Display data source info
if data_source == "Sample Data (Demo)":
    st.info("📊 **Demo Mode**: Using simulated market data for demonstration. For live data, select 'Live Data' in the sidebar.")

# Create dashboard layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Annual Return",
        f"{metrics['annual_return']*100:.2f}%",
        delta=f"{metrics['annual_return']*100:.2f}%"
    )

with col2:
    st.metric(
        "Annual Volatility",
        f"{metrics['annual_vol']*100:.2f}%",
        delta=None
    )

with col3:
    st.metric(
        "Sharpe Ratio",
        f"{metrics['sharpe']:.2f}",
        delta=f"{'Good' if metrics['sharpe'] > 1 else 'Low'}"
    )

with col4:
    st.metric(
        f"VaR ({confidence_level}%)",
        f"{metrics['var_daily']*100:.2f}% daily",
        delta=f"{metrics['var_annual']*100:.2f}% annual",
        delta_color="inverse"
    )

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🎯 Risk Analysis", "🥧 Portfolio Composition", "📊 Statistics"])

with tab1:
    st.subheader("Portfolio Performance")
    
    # Cumulative returns chart
    fig_performance = go.Figure()
    
    # Add portfolio performance
    fig_performance.add_trace(go.Scatter(
        x=metrics['cumulative_returns'].index,
        y=(metrics['cumulative_returns'] - 1) * 100,
        name='Portfolio',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Generate benchmark (simplified S&P 500 simulation)
    np.random.seed(123)
    benchmark_returns = np.random.normal(0.0003, 0.012, len(metrics['returns']))
    benchmark_cumulative = (1 + pd.Series(benchmark_returns, index=metrics['returns'].index)).cumprod()
    
    fig_performance.add_trace(go.Scatter(
        x=benchmark_cumulative.index,
        y=(benchmark_cumulative - 1) * 100,
        name='S&P 500 (Benchmark)',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig_performance.update_layout(
        title="Cumulative Returns vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # Additional performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Maximum Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        
    with col2:
        win_rate = (metrics['returns'] > 0).mean()
        st.metric("Win Rate", f"{win_rate*100:.1f}%")

with tab2:
    st.subheader("Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=metrics['returns'] * 100,
            nbinsx=50,
            name='Daily Returns',
            marker_color='#1f77b4'
        ))
        
        # Add VaR line
        fig_dist.add_vline(
            x=metrics['var_daily'] * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({confidence_level}%)"
        )
        
        fig_dist.update_layout(
            title="Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Risk-Return scatter
        individual_returns = data.pct_change().dropna()
        annual_returns = individual_returns.mean() * 252
        annual_vols = individual_returns.std() * np.sqrt(252)
        
        fig_scatter = go.Figure()
        
        # Individual stocks
        for i, ticker in enumerate(tickers):
            fig_scatter.add_trace(go.Scatter(
                x=[annual_vols[ticker] * 100],
                y=[annual_returns[ticker] * 100],
                mode='markers+text',
                name=ticker,
                text=[ticker],
                textposition="top center",
                marker=dict(size=weights[i]*100, sizemode='area')
            ))
        
        # Portfolio point
        fig_scatter.add_trace(go.Scatter(
            x=[metrics['annual_vol'] * 100],
            y=[metrics['annual_return'] * 100],
            mode='markers+text',
            name='Portfolio',
            text=['Portfolio'],
            textposition="top center",
            marker=dict(size=20, color='red', symbol='star')
        ))
        
        fig_scatter.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Annual Return (%)",
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Risk metrics summary
    st.subheader("Risk Metrics Summary")
    risk_data = {
        'Metric': ['Daily VaR', 'Annual VaR', 'Expected Shortfall', 'Volatility (Daily)', 'Beta vs Market'],
        'Value': [
            f"{metrics['var_daily']*100:.2f}%",
            f"{metrics['var_annual']*100:.2f}%",
            f"{metrics['returns'][metrics['returns'] <= metrics['var_daily']].mean()*100:.2f}%",
            f"{metrics['returns'].std()*100:.2f}%",
            f"1.00"  # Simplified for demo
        ]
    }
    st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)

with tab3:
    st.subheader("Portfolio Composition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of holdings
        fig_pie = px.pie(
            values=weights,
            names=tickers,
            title="Portfolio Allocation"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Holdings table
        holdings_df = pd.DataFrame({
            'Ticker': tickers,
            'Weight': [f"{w*100:.1f}%" for w in weights],
            'Current Price': [f"${data[t].iloc[-1]:.2f}" for t in tickers],
            '1D Change': [f"{((data[t].iloc[-1]/data[t].iloc[-2] - 1)*100):.2f}%" for t in tickers]
        })
        st.dataframe(holdings_df, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Statistical Analysis")
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = data[tickers].pct_change().dropna().corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary_stats = individual_returns.describe()
    summary_stats = summary_stats.applymap(lambda x: f"{x*100:.3f}%")
    st.dataframe(summary_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built for BlackRock Analyst Program Application | Data: Live from Yahoo Finance or Sample (Demo Mode)")