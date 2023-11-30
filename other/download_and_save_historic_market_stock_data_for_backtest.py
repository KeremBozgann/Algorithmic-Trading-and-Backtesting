import yfinance as yf

def get_and_save_data(tick, start_date, end_date):

    yf_data = yf.download(tick, start=start_date, end=end_date)
    yf_data.sort_values(by=['Date'], inplace=True, ascending=True)
    # yf_data.reset_index(inplace=True, drop=False)
    yf_data.insert(0, 'Date', yf_data.index)
    # yf_data['Date']  = yf_data.index
    # yf_data['datetime2'] = yf_data.index
    yf_data['Date'] = yf_data['Date'].dt.strftime('%Y/%m/%d')
    yf_data = yf_data.rename(columns={'Volume': 'volume', 'Open': 'open', 'High':'high', 'Low': 'low','Close':'close',
                                      'Adj Close': 'adj_close', 'Date': 'datetime'})
    yf_data.dropna(inplace=True)
    print(yf_data.head())
    yf_data.to_csv(f'../data/{tick}.csv', index=False)


# tick = 'META'

tick = 'SPY'
# tick = 'AAPL'
# tick = 'META'
# tick = 'MSFT'

start_date = "2000-01-11"
end_date= "2023-01-01"
get_and_save_data(tick, start_date, end_date)