import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys

def fetch_and_plot_one_stock(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    if stock_data.empty:
        print("No data found for the given symbol and date range.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], marker='o', linestyle='-')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.grid(True)

    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.title(f"{symbol} Stock Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# fetch_and_plot_one_stock("BA", "2017-01-4", "2017-02-25")  # A shorter range is used for better visibility
# fetch_and_plot_one_stock("AAPL",  "2017-01-4", "2017-02-25")   # A shorter range is used for better visibility
# fetch_and_plot_one_stock("BA", "2017-01-4", "2017-02-25")  # A shorter range is used for better visibility
# fetch_and_plot_one_stock("AAPL",  "2017-01-4", "2017-02-25")   # A shorter range is used for better visibility



def fetch_and_plot_stocks(symbols, start_date, end_date):
    plt.figure(figsize=(12, 6))

    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"No data found for {symbol} in the given date range.")
            continue

        plt.plot(stock_data.index, stock_data['Close'], marker='o', linestyle='-', label=symbol)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.grid(True)

    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.title("Stock Prices")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# fetch_and_plot_stocks(["BA", "AAPL", "TSLA"], "2017-01-4", "2017-02-25")   # Replace with your desired stocks and date range





import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def fetch_and_plot_scaled_stocks(symbols, start_date, end_date):
    plt.figure(figsize=(12, 6))

    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"No data found for {symbol} in the given date range.")
            continue

        scaled_prices = (stock_data['Close'] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]

        plt.plot(stock_data.index, scaled_prices, marker='o', linestyle='-', label=symbol)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.grid(True)

    plt.xlabel("Date")
    plt.ylabel("Relative Price Change")
    plt.title("Stock Prices Scaled to Same Start Point")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# fetch_and_plot_scaled_stocks(["BA", "AAPL", "TSLA"], "2017-01-04", "2017-02-25")

#plot different aspects of the data to look for feature interaction, complexity, homogeneity, multicollinearity

import sys
import numpy as np

sys.path.append('..')
from forecast import create_lagged_series as ls
import os
#from data import AAPL
import pandas as pd
sym = "AAPL"
lags = 5
start_date = "2001-01-04"
end_date = "2005-02-25"
# data = yf.download("AAPL", "2017-01-04", "2017-02-25")
#lagged_series = ls(AAPL, "2017-01-04", "2017-02-25")
ts = pd.read_csv(f'../data/{sym}.csv')

# Create the new lagged DataFrame
tslag = pd.DataFrame(index=ts.index)
tslag["Today"] = ts["adj_close"]
tslag["Volume"] = ts["volume"]

# Create the shifted lag series of prior trading period close values
for i in range(0, lags):
    tslag["Lag%s" % str(i + 1)] = ts["adj_close"].shift(i + 1)

# Create the returns DataFrame
tsret = pd.DataFrame(index=tslag.index)
tsret["Volume"] = tslag["Volume"]
tsret["Today"] = tslag["Today"].pct_change() * 100.0

# If any of the values of percentage returns equal zero, set them to # a small number (stops issues with QDA model in Scikit-Learn)
for i, x in enumerate(tsret["Today"]):
    if (abs(x) < 0.0001):
        tsret["Today"][i] = 0.0001

# Create the lagged percentage returns columns
for i in range(0, lags):
    tsret["Lag%s" % str(i + 1)] = \
        tslag["Lag%s" % str(i + 1)].pct_change() * 100.0
# Create the "Direction" column (+1 or -1) indicating an up/down day
tsret["Direction"] = np.sign(tsret["Today"])
tsret['datetime'] = ts['datetime']
tsret.set_index('datetime', inplace=True)
tsret.index = pd.to_datetime(tsret.index)
tsret = tsret[tsret.index >= start_date]
tsret = tsret.iloc[lags+1:]
lagged_series = tsret
# Complexity
# sns.pairplot(data)
# plt.show()
sns.pairplot(lagged_series[['Lag1','Lag2','Lag3','Lag4','Lag5','Today','Volume']])
plt.show()

# Dimensionality Reduction
from sklearn.decomposition import PCA

pca = PCA()
reduced = pca.fit(lagged_series)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.show()

# Homogeneity 
# Target variable distribution
sns.histplot(lagged_series['Direction'])
plt.show()
# Feature variable distribution
sns.histplot(data=lagged_series, x='Lag1', kde=True, label='Lag 1', alpha=0.5)
plt.show()
sns.histplot(data=lagged_series, x='Lag2', kde=True, label='Lag 1', alpha=0.5)
plt.show()
sns.histplot(data=lagged_series, x='Lag3', kde=True, label='Lag 1', alpha=0.5)
plt.show()

# Class Balance
sns.countplot(x='Lag1')
plt.show()

# Data Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
lagged_series['cluster'] = kmeans.fit_predict(lagged_series)
plt.scatter(lagged_series['Lag1'],lagged_series['Lag2'],lagged_series['Lag3'],
            lagged_series['Lag4'],lagged_series['Lag5'],c=lagged_series['cluster'],cmap='viridis')
plt.show()