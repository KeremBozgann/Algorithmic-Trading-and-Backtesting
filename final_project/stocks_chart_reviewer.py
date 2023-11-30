import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

fetch_and_plot_one_stock("BA", "2017-01-4", "2017-02-25")  # A shorter range is used for better visibility
fetch_and_plot_one_stock("AAPL",  "2017-01-4", "2017-02-25")   # A shorter range is used for better visibility



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

fetch_and_plot_stocks(["BA", "AAPL", "TSLA"], "2017-01-4", "2017-02-25")   # Replace with your desired stocks and date range





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

fetch_and_plot_scaled_stocks(["BA", "AAPL", "TSLA"], "2017-01-04", "2017-02-25")
