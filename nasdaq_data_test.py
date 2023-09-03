import nasdaqdatalink
import quandl


quandl.ApiConfig.api_key = 'kvvC4xpcRLK8XEXXDzG-'
# data = nasdaqdatalink.get('OFDP/FUTURE_ESZ2014')
# print(data)

futures = []
quandl.ApiConfig.api_key = 'kvvC4xpcRLK8XEXXDzG-'
data = quandl.get('BATS/TSLA', start_date='2018-12-31', end_date='2018-12-31')

# quandl.get('FINRA/FNRA_TSLA')

apple_data= quandl.get('XNAS/AAPL', start_date = "2010-01-01",end_date= "2020-01-01" )
quandl.get("AAPL/EOD")
head = apple_data.head()
apple_data = quandl.get("EOD/AAPL", authtoken="kvvC4xpcRLK8XEXXDzG")