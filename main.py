import numpy as np
from alpha import run_forecast

import pandas as pd
import matplotlib.pyplot as plt
from beta import beta_model, best_stock
import datetime



'''User input needed: selected_risk, stock_symbols, initial_capital, model_name, start_train_date, end_train_date, 
start_test_date, end_test_date'''

# enter a value between 0 to 100. Determines the number of stock exchanged at each transaction
    # based on the selected risk. A high risk--> higher amount of stocks exchanged
selected_risk = 50

# If model_name is "Rule Based", "Confident Logistic Regression" or  "Logistic Regression with Sum of Percentage Change Input", please
# give a single stock symbol below
stock_symbols = ['msft'] # Choose from the following: 'spy', 'msft', 'amzn', tsla', 'googl', 'meta'

#initial capital. The number of stocks exchanged is proportinal to the initial capital. Unit: Dollars
initial_capital = 100000

# model_name = "LDA"
# model_name = "QDA"
# model_name = "Logistic Regression"
# model_name = "LDA_BAGG"
# model_name = "Perceptron"
# model_name = "Gradient Boosting"
# model_name = "RandomForestClassifier"
model_name = "ANN"


# model_name = "Rule Based"
# model_name = "Confident Logistic Regression"
# model_name = "Logistic Regression with Sum of Percentage Change Input"



start_train_date, end_train_date, start_test_date, end_test_date = datetime.datetime(2004, 1, 10) , datetime.datetime(2014, 1, 10) ,\
                                                    datetime.datetime(2014, 1, 11), datetime.datetime(2018, 1, 11)


print(f"Selected Risk Level: {selected_risk}")
print(f"Stock Symbols: {stock_symbols}")
print(f"Initial Capital: {initial_capital}")
print('selected risk, ', selected_risk)
print('model name: ', model_name)




# Set a higher trading volume when the model is rule based or Confident Logistic Regression to compensate for the fact that
# they trade on less number of days
if model_name=="Rule Based" or model_name == "Confident Logistic Regression" or \
    model_name == "Logistic Regression with Sum of Percentage Change Input":
    trade_volume = 100 + (1000000 - 100)/100 *  selected_risk * initial_capital / 10000

else:
    trade_volume = 100 + (1000 - 100)/100 *  selected_risk * initial_capital / 10000

# Get the optimal allocation ratios suggested by the beta model. If single stock entered, weight is simply 1
weights = beta_model(stock_symbols)


equity_curve_list = list()
stock_close_list = list()

for i in range(len(stock_symbols)):

    # compute the trade volume and capital allocated to this symbol
    _trade_volume = trade_volume * weights[i]
    _initial_capital = initial_capital * weights[i]

    _symbol_list = []
    _symbol_list.append(stock_symbols[i])

    if not _initial_capital == 0:

        #run the simulation and return the equity curve
        total_gain, returns, equity_curve, drawdown = run_forecast(_symbol_list,
                                                                       _initial_capital, round(_trade_volume),
                                                                       model_name, start_train_date, end_train_date,
                                                                       start_test_date, end_test_date)

        returns_numpy = returns.to_numpy()
        equity_curve_numpy = equity_curve.to_numpy()

        equity_curve_zero_mean = (equity_curve_numpy - 1)[1:]


    # get the stock price
    df = pd.read_csv(f'data/{stock_symbols[i]}.csv')
    threshold_start = start_test_date
    threshold_end = end_test_date
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'] > threshold_start]
    df = df[df['datetime'] < threshold_end]

    stock_close = df['adj_close'].to_numpy()
    stock_close_norm= (stock_close - stock_close[0])/stock_close[0]

    stock_close_rat = stock_close/stock_close[0]

    equity_curve_list.append(equity_curve_numpy)
    stock_close_list.append(stock_close_rat)







# fig, ax = plt.subplots(figsize=(6, 4))  # Fix: Use plt.subplots instead of plt.figure
# ax.plot((equity_curve_numpy - 1) * 100, label="Algo")  # Add legend labels
# ax.plot((stock_close_norm ) * 100, label=f"{self.stock_symbols[0]}")  # Plot SPY data
#
#
#
# plt.xlabel('Time (Days)')
# plt.ylabel('Equity Curves (%)')
# plt.title(f'Equity Curve Over Time: Algo vs. {self.stock_symbols[0]}')
# plt.legend()  # Include legend
# plt.show()
#
# # plt.xticks(xticks)
# # plt.xticklabels(xticklabels)
#
# # # Ensure that only integer ticks are displayed on the y-axis
# # plt.yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.savefig('chart.png')  # Save the chart as an image file
# # # Display the Matplotlib chart
# plt.show(block=True)

sum_equity = 0
sum_stock = 0

for i in range(len(stock_symbols)):
    _initial_capital = initial_capital * weights[i]
    equity_curve = equity_curve_list[i]
    stock_close_rat = stock_close_list[i]

    sum_equity += equity_curve * _initial_capital
    sum_stock += stock_close_rat * _initial_capital


if model_name != "Rule Based" and model_name != "Confident Logistic Regression" and model_name != "Logistic Regression with Sum of Percentage Change Input":
    # get the price data for best stock (stock with the most equity gain)
    best_stoc, ind_best = best_stock(stock_symbols)

    # best_stock_close = df['adj_close'].to_numpy()
    # df_ = pd.read_csv(f'data/{self.stock_symbols[i]}.csv')
    # stock_close = df['adj_close'].to_numpy()
    #
    # stock_close_norm = (stock_close - stock_close[0]) / stock_close[0]
    #
    # stock_close_rat = stock_close / stock_close[0]

    # get the equity curve resulting from running the alpha model only on the best stock
    stock_best_curve = stock_close_list[ind_best]
    total_gain_best, returns_best, equity_curve_best, drawdown_best = run_forecast([best_stoc],
                                                                   initial_capital, round(trade_volume),
                                                                   model_name, start_train_date, end_train_date,
                                                                                       start_test_date,end_test_date)



from matplotlib import rc, rcParams
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')

linewidth = 4
font_ax = 20
font_leg = 20
font_ticks = 18


if  model_name == "Rule Based" or  model_name == "Confident Logistic Regression" or \
    model_name == "Logistic Regression with Sum of Percentage Change Input":

    fig, ax = plt.subplots(figsize=(6, 4))


    # Compare: Algorithm run on the stock vs. Buy and Hold on stock
    ax.plot(sum_equity, label=f"{model_name}", linewidth=linewidth)
    ax.plot(sum_stock, label=f"Stock: {stock_symbols[0]}", linewidth=linewidth)

    plt.xlabel(r'\textbf{Time (Days)}')
    plt.ylabel(r'\textbf{Equity Curves (\$)}')
    plt.title(fr'\textbf{{Equity Curve Over Time: Algo vs.}} \textbf{{{stock_symbols[0]}}}')

    plt.legend()  # Include legend
    plt.show()

    plt.savefig('chart.png')
    plt.show(block=True)

else:

    if len(stock_symbols) == 1:

        fig, ax = plt.subplots(figsize=(6, 4))

        # Compare: Algorithm run on Combined stocks, Buy and Hold Combined stocks, Algorithm run on single best stock, Buy and Hold Best stock
        ax.plot(sum_equity, label=f"{model_name}")
        ax.plot(sum_stock, label=f"Stock (Buy and Hold)")  # Plot combined stocks

        plt.xlabel(r'\textbf{Time (Days)}')
        plt.ylabel(r'\textbf{Equity Curves (\$)}')
        plt.title(fr'\textbf{{Equity Curve Over Time: Algo vs.}} \textbf{{{stock_symbols[0]}}}')

        plt.legend()
        plt.show()

        plt.savefig('chart.png')
        plt.show(block=True)

    else:

        fig, ax = plt.subplots(figsize=(6, 4))

        # Compare: Algorithm run on Combined stocks, Buy and Hold Combined stocks, Algorithm run on single best stock, Buy and Hold Best stock
        ax.plot(sum_equity, label=f"{model_name}")
        ax.plot(sum_equity, label=f"Algo - Combined Stocks")
        ax.plot(stock_best_curve * initial_capital, label=f"Best Stock")
        ax.plot(sum_stock, label=f"Combined Stocks")  # Plot combined stocks
        ax.plot(equity_curve_best.to_numpy() * initial_capital, label=f"Algo- Best Stock")  # Plot combined stocks

        plt.xlabel(r'\textbf{Time (Days)}')
        plt.ylabel(r'\textbf{Equity Curves (\$)}')
        plt.title(fr'\textbf{{Equity Curve Over Time: Algo vs.}} \textbf{{{stock_symbols[0]}}}')

        plt.legend()
        plt.show()

        plt.savefig('chart.png')
        plt.show(block=True)
