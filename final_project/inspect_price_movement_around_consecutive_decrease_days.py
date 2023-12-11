import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def plot_prices_around_decrease_day(symbol, n, pre_days, post_days, start_date, end_date, i):
    # Download stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Decrease'] = stock_data['Daily_Return'] < 0

    # Find occurrences of n consecutive decreases
    stock_data['Consecutive_Decreases'] = stock_data['Decrease'].rolling(window=n).sum() == n
    decrease_days = stock_data[stock_data['Consecutive_Decreases']].index

    if i < len(decrease_days):
        # Take the i-th occurrence
        target_day = decrease_days[i]

        # Determine the date range to plot
        start_plot = target_day - pd.Timedelta(days=pre_days)
        end_plot = target_day + pd.Timedelta(days=post_days)

        # Filter data for the plot range
        plot_data = stock_data[(stock_data.index >= start_plot) & (stock_data.index <= end_plot)]

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(plot_data['Close'], marker='o', linestyle='-')
        if target_day in plot_data.index:
            plt.scatter(target_day, plot_data.loc[target_day, 'Close'], color='red', zorder=5)  # Red dot at the target day
        plt.title(f"{symbol} Stock Prices Around {target_day.date()}")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.show()
    else:
        print(f"Index {i} is out of range. Only {len(decrease_days)} instances found.")

# Parameters
symbol = 'AAPL'
n_decrease_days = 5  # Number of consecutive decrease days
pre_days = 15  # Days before the decrease day
post_days = 15  # Days after the decrease day
start_date = '2010-01-01'  # Start date for the data
end_date = '2014-01-01'  # End date for the data
i = 5
plot_prices_around_decrease_day(symbol, n_decrease_days, pre_days, post_days, start_date, end_date, i)
