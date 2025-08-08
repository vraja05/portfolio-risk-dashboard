import yfinance as yf
import ssl
import certifi
import os

# Set SSL certificate
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Disable SSL verification temporarily (just for testing)
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Test 1: Basic download
print("Test 1: Basic download...")
try:
    msft = yf.Ticker("MSFT")
    hist = msft.history(period="5d")
    print(f"✅ Success! Got {len(hist)} days of data")
    print(hist.tail())
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Using download function
print("\nTest 2: Download function...")
try:
    data = yf.download("AAPL", start="2024-12-01", end="2024-12-10", progress=False)
    print(f"✅ Success! Got {len(data)} days")
except Exception as e:
    print(f"❌ Failed: {e}")