import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



from matplotlib import rc,rcParams
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')

linewidth = 4
font_ax = 20
font_leg = 20
font_ticks=  18


# Get the stock chart
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






# Get the scaled stock chart (scaled the stocks)
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

# fetch_and_plot_scaled_stocks(["BA", "WMT", "GOOGL"], "2004-11-10", "2004-11-30")


# For long term visualization
def fetch_and_plot_long_term_scaled_stocks(symbols, start_date, end_date):
    plt.figure(figsize=(12, 6))

    for symbol in symbols:
        # fetch stock data
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        # check if data is empty
        if stock_data.empty:
            print(f"No data found for {symbol} in the given date range.")
            continue

        # scale the stock prices to start at zero and show relative change
        scaled_prices = (stock_data['Close'] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]

        plt.plot(stock_data.index, scaled_prices, marker='', linestyle='-', label=symbol)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Set interval to every 6 months

    plt.grid(True)

    plt.xlabel("Date")
    plt.ylabel("Relative Price Change")
    plt.title("Long Term Scaled Stock Prices")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# fetch_and_plot_long_term_scaled_stocks(["BA", "WMT", "GOOGL"], "2010-01-01", "2010-05-31")




# Get the number of increase vs. decrease days following an increase trend
def analyze_post_increase_days(symbol, n):
    # Fetch historical stock data
    stock_data = yf.download(symbol)


    # Calculate daily returns
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()

    # Identify days with price increase
    stock_data['Increase'] = stock_data['Daily_Return'] > 0

    # Find sequences of n consecutive increases
    stock_data['Consecutive_Increases'] = stock_data['Increase'].rolling(window=n).sum() == n

    # Shift the consecutive increases column to align with the next day
    stock_data['Prev_Consecutive_Increases'] = stock_data['Consecutive_Increases'].shift(-n)

    # Count the number of increases and decreases after n consecutive increases
    increases_after_n_increases = stock_data[stock_data['Prev_Consecutive_Increases'] & stock_data['Increase']]['Increase'].count()
    decreases_after_n_increases = stock_data[stock_data['Prev_Consecutive_Increases'] & ~stock_data['Increase']]['Increase'].count()

    # Calculate the ratio
    ratio = increases_after_n_increases / decreases_after_n_increases if decreases_after_n_increases != 0 else float('inf')

    return increases_after_n_increases, decreases_after_n_increases, ratio
# symbol = 'SPY'
# n = 9
# increases, decreases, ratio = analyze_post_increase_days(symbol, n)
# print(f"Number of increases after {n} consecutive increases: {increases}")
# print(f"Number of decreases after {n} consecutive increases: {decreases}")
# print(f"Ratio of increases to decreases: {ratio}")


# Get the number of increase vs. decrease days following an decrease trend
def analyze_post_decrease_days(symbol, n):
    # Fetch historical stock data
    stock_data = yf.download(symbol)

    # Calculate daily returns
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()

    # Identify days with price decrease
    stock_data['Decrease'] = stock_data['Daily_Return'] < 0

    # Find sequences of n consecutive decreases
    stock_data['Consecutive_Decreases'] = stock_data['Decrease'].rolling(window=n).sum() == n

    # Shift the consecutive decreases column to align with the next day
    stock_data['Post_Decrease'] = stock_data['Consecutive_Decreases'].shift(1).fillna(False)

    # Count the number of increases and decreases after n consecutive decreases
    increases_after_n_decreases = stock_data[stock_data['Post_Decrease'] & (stock_data['Daily_Return'] > 0)].count()['Post_Decrease']
    decreases_after_n_decreases = stock_data[stock_data['Post_Decrease'] & (stock_data['Daily_Return'] < 0)].count()['Post_Decrease']

    # Calculate the ratio
    ratio = increases_after_n_decreases / decreases_after_n_decreases if decreases_after_n_decreases != 0 else float('inf')

    return increases_after_n_decreases, decreases_after_n_decreases, ratio


# symbol = 'GOOGL'
# n = 4
# increases, decreases, ratio = analyze_post_decrease_days(symbol, n)
#
# print(f"Number of increases after {n} consecutive decreases: {increases}")
# print(f"Number of decreases after {n} consecutive decreases: {decreases}")
# print(f"Ratio of increases to decreases: {ratio}")



# Plot the ratio of increases to decreases vs. the number of consecutive decrease days
def analyze_post_decrease_ratios(symbol, max_n):
    stock_data = yf.download(symbol)
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Decrease'] = stock_data['Daily_Return'] < 0
    ratios = []

    for n in range(1, max_n + 1):
        stock_data['Consecutive_Decreases'] = stock_data['Decrease'].rolling(window=n).sum() == n
        stock_data['Post_Decrease'] = stock_data['Consecutive_Decreases'].shift(1).fillna(False)

        increases = stock_data[stock_data['Post_Decrease'] & (stock_data['Daily_Return'] > 0)].count()['Post_Decrease']
        decreases = stock_data[stock_data['Post_Decrease'] & (stock_data['Daily_Return'] < 0)].count()['Post_Decrease']

        ratio = increases / decreases if decreases != 0 else float('inf')
        ratios.append(ratio)

    return ratios

# # List of stock symbols
# stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Replace with your desired stock symbols
# max_n = 10  # Maximum number of consecutive days of decrease
#
# plt.figure(figsize=(12, 8))
#
# # Plotting for each stock
# for stock in stocks:
#     ratios = analyze_post_decrease_ratios(stock, max_n)
#     plt.plot(range(1, max_n + 1), ratios, marker='o', linestyle='-', label=stock)
#
# plt.xlabel('Consecutive Decrease Days (n)')
# plt.ylabel('Ratio of Increases to Decreases')
# plt.title('Stock Price Increase/Decrease Ratio after n Consecutive Days of Decrease')
# plt.xticks(range(1, max_n + 1))
# plt.legend()
# plt.grid(True)
# plt.show()




# Plot the ratio of increases to decreases vs. the number of consecutive increase days
def analyze_post_increase_ratios(symbol, max_n, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Increase'] = stock_data['Daily_Return'] > 0
    ratios = []
    counts = []  # To store the number of data points for each n

    for n in range(1, max_n + 1):
        stock_data['Consecutive_Increases'] = stock_data['Increase'].rolling(window=n).sum() == n
        stock_data['Prev_Consecutive_Increases'] = stock_data['Consecutive_Increases'].shift(-n)

        increases = stock_data[stock_data['Prev_Consecutive_Increases'] & stock_data['Increase']]['Increase'].count()
        decreases = stock_data[stock_data['Prev_Consecutive_Increases'] & ~stock_data['Increase']]['Increase'].count()

        ratio = increases / decreases if decreases != 0 else float('inf')
        ratios.append(ratio)
        counts.append(stock_data['Consecutive_Increases'].sum())  # Count occurrences

    return ratios, counts

# stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', "BA"]
# stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', "TSLA"]
stocks = ['AAPL', 'AMZN', "TSLA", 'MSFT']

max_n = 10
start_date = '2010-01-01'  # Start date for the data
end_date = '2014-01-01'  # End date for the data

plt.figure(figsize=(12, 8))


plt.xlabel(r'\textbf{Number of Consecutive Increase Days}', fontsize=font_ax)
plt.ylabel(r'\textbf{Ratio of Increases to Decreases}', fontsize=font_ax)
plt.title(fr'\textbf{{Stock Price Increase/Decrease Ratio ({start_date} to {end_date})}}', fontsize=font_ax)

# plot
for stock in stocks:
    ratios, counts = analyze_post_increase_ratios(stock, max_n, start_date, end_date)
    n_values = range(1, max_n + 1)
    plt.plot(n_values, ratios, marker='o', linestyle='-', label=stock,  linewidth=linewidth, markersize = 10)

    # Annotating each data point with the count of instances in which a consecutive increase trend is observed
    for i, count in enumerate(counts):
        plt.annotate(str(count), (n_values[i], ratios[i]), textcoords="offset points", xytext=(0,10), ha='center')


plt.xlabel(r'\textbf{Consecutive Increase Days (n)}', fontsize=font_ax)
plt.ylabel(r'\textbf{Ratio of Increases to Decreases}', fontsize=font_ax)
plt.title(fr'\textbf{{Stock Price Increase/Decrease Ratio Between ({start_date} to {end_date})}}', fontsize=font_ax)
plt.xticks(range(1, max_n + 1))
plt.tick_params(axis='y', right=True, labelright=False)
plt.tick_params(axis='x', top=True, labeltop=False)
plt.grid(True, which='major', axis='both', alpha=0.3)
plt.xticks(fontsize=font_ticks)
plt.yticks(fontsize=font_ticks)
plt.legend(fontsize=font_ax)
plt.ylim(top =6.0)
plt.show()



# Plot the ratio of increases to decreases vs. the number of consecutive decrease days
def analyze_post_decrease_ratios(symbol, max_n, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Decrease'] = stock_data['Daily_Return'] < 0
    ratios = []
    counts = []

    for n in range(1, max_n + 1):
        stock_data['Consecutive_Decreases'] = stock_data['Decrease'].rolling(window=n).sum() == n
        stock_data['Post_Decrease'] = stock_data['Consecutive_Decreases'].shift(1).fillna(False)

        increases = stock_data[stock_data['Post_Decrease'] & (stock_data['Daily_Return'] > 0)].count()['Post_Decrease']
        decreases = stock_data[stock_data['Post_Decrease'] & (stock_data['Daily_Return'] < 0)].count()['Post_Decrease']

        ratio = increases / decreases if decreases != 0 else float('inf')
        ratios.append(ratio)
        counts.append(stock_data['Consecutive_Decreases'].sum())  # Count occurrences

    return ratios, counts


# stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', "BA", "META", "TSLA"]
# stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', "TSLA"]
stocks = ['AAPL', 'AMZN', "TSLA", 'MSFT']
max_n = 10  # Maximum number of consecutive days of decrease
start_date = '2010-01-01'  # Start date for the data
end_date = '2014-01-01'  # End date for the data

plt.figure(figsize=(12, 8))

# Plot
for stock in stocks:
    ratios, counts = analyze_post_decrease_ratios(stock, max_n, start_date, end_date)
    n_values = range(1, max_n + 1)
    plt.plot(n_values, ratios, marker='o', linestyle='-', label=stock,  linewidth=linewidth, markersize = 10)

    # Annotating each data point with the count of occurrences
    for i, count in enumerate(counts):
        plt.annotate(str(count), (n_values[i], ratios[i]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel(r'\textbf{Consecutive Decrease Days (n)}', fontsize=font_ax)
plt.ylabel(r'\textbf{Ratio of Increases to Decreases}', fontsize=font_ax)
plt.title(fr'\textbf{{Stock Price Increase/Decrease Ratio ({start_date} to {end_date})}}', fontsize=font_ax)
plt.xticks(range(1, max_n + 1))
plt.tick_params(axis='y', right=True, labelright=False)
plt.tick_params(axis='x', top=True, labeltop=False)
plt.grid(True, which='major', axis='both', alpha=0.3)
plt.xticks(fontsize=font_ticks)
plt.yticks(fontsize=font_ticks)
plt.legend(fontsize=font_ax)
plt.ylim(bottom =0.0)
plt.show()





