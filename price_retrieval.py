from __future__ import print_function
import datetime
import warnings
import MySQLdb as mdb
import requests
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

db_host = 'localhost'
db_user = 'sec_user'
db_pass = 'alfabeta33'
db_name = 'securities_master'
con = mdb.connect(db_host, db_user, db_pass, db_name)



def obtain_list_of_db_tickers():
    """
    Obtains a list of the ticker symbols in the database.
    """
    with con:
        cur = con.cursor()
        cur.execute("SELECT id, ticker FROM symbol")
        data = cur.fetchall()

        return [(d[0], d[1]) for d in data]


def get_daily_historic_data_yahoo( ticker, start_date="2000-1-1",
end_date=datetime.date.today().strftime("%Y-%m-%d") ):
    """
    Obtains data from Yahoo Finance returns and a list of tuples.
    ticker: Yahoo Finance ticker symbol, e.g. "GOOG" for Google, Inc. start_date: Start date in (YYYY, M, D) format
    end_date: End date in (YYYY, M, D) format
    """
    # Construct the Yahoo URL with the correct integer query parameters
    # for start and end dates. Note that some parameters are zero-based!

    # ticker_tup = (ticker, start_date[1]-1, start_date[2],
    # start_date[0], end_date[1]-1, end_date[2],
    # end_date[0])

    # yahoo_url = "http://ichart.finance.yahoo.com/table.csv"
    # yahoo_url = "https://finance.yahoo.com/chart/csv/"
    # yahoo_url += "?s=%s&a=%s&b=%s&c=%s&d=%s&e=%s&f=%s"
    # yahoo_url = yahoo_url % ticker_tup
    #
    # # Try connecting to Yahoo Finance and obtaining the data
    # # On failure, print an error message.
    # session = requests.Session()
    # retry = Retry(connect=3, backoff_factor=0.5)
    # adapter = HTTPAdapter(max_retries=retry)
    # session.mount('http://', adapter)
    # session.mount('https://', adapter)
    # yf_data = session.get(yahoo_url).text.split("\n")[1:-1]



    try:
        # yf_data = requests.get(yahoo_url).text.split("\n")[1:-1]
        yf_data = yf.download(tickers[-1][1], start=start_date, end=end_date)
        prices = []
        for i in range(len(yf_data)):
            entry = yf_data.iloc[i]
            # p = y.strip().split(',')
            prices.append(
                (entry.name.strftime("%Y-%m-%d %H:%M:%S"),
                entry[0],  entry[1], entry[2], entry[3], entry[4], entry[5]))

    except Exception as e:
        print("Could not download Yahoo data: %s" % e)
    return prices

def insert_daily_data_into_db( data_vendor_id, symbol_id, daily_data
):
    """
    Takes a list of tuples of daily data and adds it to the
    MySQL database. Appends the vendor ID and symbol ID to the data.
    daily_data: List of tuples of the OHLC data (with
    adj_close and volume)
    """

    db_host = 'localhost'
    db_user = 'sec_user'
    db_pass = 'alfabeta33'
    db_name = 'securities_master'
    con = mdb.connect(db_host, db_user, db_pass, db_name)


    # Create the time now
    now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    # Amend the data to include the vendor ID and symbol ID
    daily_data = [
        (data_vendor_id, symbol_id, d[0], now, now,
         d[1], d[2], d[3], d[4], d[5], d[6])
        for d in daily_data]
    # Create the insert strings
    column_str = """data_vendor_id, symbol_id, price_date, created_date, last_updated_date, open_price, high_price, low_price, close_price, adj_close_price, volume"""
    insert_str = ("%s, " * 11)[:-2]

    final_str = "INSERT INTO daily_price (%s) VALUES (%s)" % (column_str, insert_str)

    # final_str_test= "INSERT INTO daily_price (data_vendor_id, symbol_id, price_date, created_date, last_updated_date, open_price, high_price, low_price, close_price, adj_close_price, volume) VALUES ('1', '8057','2013-02-01 00:00:00','2023-08-30 01:25:37', '2023-08-30 01:25:37','31.5', '31.739999771118164', '30.469999313354492', '31.010000228881836', '28.75995445251465', '66789100.0')"

    # Using the MySQL connection, carry out an INSERT INTO for every symbol

    with con:
        cur = con.cursor()
        cur.executemany(final_str, daily_data)
        con.commit()

if __name__ == "__main__":
    # This ignores the warnings regarding Data Truncation # from the Yahoo precision to Decimal(19,4) datatypes warnings.filterwarnings(’ignore’)
    # Loop over the tickers and insert the daily historical # data into the database
    tickers = obtain_list_of_db_tickers()
    lentickers = len(tickers)
    for i, t in enumerate(tickers):
        print("Adding data for %s: %s out of %s" %
                (t[1], i+1, lentickers))
        yf_data = get_daily_historic_data_yahoo(t[1])
        insert_daily_data_into_db('1', t[0], yf_data)
    print("Successfully added Yahoo Finance pricing data to DB.")


#data_vendor_id = '1'
#symbol_id =  t[0]
#daily_data = yf_data
