import yfinance as yf
import datetime


start_date="2007-1-1"
end_date=datetime.date.today().strftime("%Y-%m-%d")


yf_data = yf.download('SPY', start=start_date, end=end_date)



