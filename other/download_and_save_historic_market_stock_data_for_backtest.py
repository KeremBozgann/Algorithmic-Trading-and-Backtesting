import yfinance as yf

def get_and_save_data(tick, start_date, end_date):

    yf_data = yf.download(tick, start=start_date, end=end_date)
    yf_data.sort_values(by=['Date'], inplace=True, ascending=True)
    yf_data.reset_index(inplace=True)
    yf_data['Date'] = yf_data['Date'].dt.strftime('%Y/%m/%d')
    yf_data = yf_data.rename(columns={'Volume': 'volume', 'Open': 'open', 'High':'high', 'Low': 'low','Close':'close',
                                      'Adj Close': 'adj_close', 'Date': 'datetime'})
    print(yf_data.head())
    yf_data.to_csv(f'../data/{tick}.csv', index=False)

tick = 'GOOGL'
# tick = 'SPY'

start_date = "2006-01-01"
end_date= "2023-01-01"
get_and_save_data(tick, start_date, end_date)