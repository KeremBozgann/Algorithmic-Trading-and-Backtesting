import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import pandas as pd
import numpy as np

'''Mean-Variance Optimization (Markowitz Model)
Finds the optimal capital allocation that Maximizes the Sharpe ratio'''
def beta_model(stock_list):
    # stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    # data = yf.download(stock_list, start="2010-01-01", end="2020-01-01")['Adj Close']
    start_date = "2010-01-01"
    end_date = "2020-01-01"
    all_data = pd.DataFrame()

    for stock in stock_list:
        data = yf.download(stock, start=start_date, end=end_date)
        all_data[stock] = data['Adj Close']

    # Compute the expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(all_data)
    S = risk_models.sample_cov(all_data)

    ef = EfficientFrontier(mu, S)

    #obtain the optimal weights that maximizes the Sharpe ratio of the portfolio
    weights = ef.max_sharpe()
    cleaned_weights = list()
    for tup in weights:
        cleaned_weights.append(weights[tup])


    return cleaned_weights



def best_stock(stock_list):
    # stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    # data = yf.download(stock_list, start="2010-01-01", end="2020-01-01")['Adj Close']
    start_date = "2010-01-01"
    end_date = "2020-01-01"
    all_data = pd.DataFrame()

    for stock in stock_list:
        data = yf.download(stock, start=start_date, end=end_date)
        all_data[stock] = data['Adj Close']

    # Compute the expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(all_data)
    S = risk_models.sample_cov(all_data)

    best_stock = stock_list[np.argmax(mu)]


    return best_stock