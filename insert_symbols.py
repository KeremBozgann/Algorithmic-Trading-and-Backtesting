from __future__ import print_function
import datetime
from math import ceil
import bs4
import MySQLdb as mdb
import requests
import datetime


def obtain_parse_wiki_snp500(features):
    """
Download and parse the Wikipedia list of S&P500
constituents using requests and BeautifulSoup.
Returns a list of tuples for to add to MySQL.
"""
    # Stores the current time, for the created_at record
    now = datetime.datetime.utcnow()
    # Use requests and BeautifulSoup to download the
    # list of S&P500 companies and obtain the symbol table
    response = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

    soup = bs4.BeautifulSoup(response.text, features = features)
    # This selects the first table, using CSS Selector syntax # and then ignores the header row ([1:])
    symbolslist = soup.select('table')[0].select('tr')[1:]
    # Obtain the symbol information for each # row in the S&P500 constituent table
    symbols = []
    for i, symbol in enumerate(symbolslist):
        tds = symbol.select('td')
        symbols.append(
            (
            tds[0].select('a')[0].text, # Ticker
            'stock',
            tds[1].select('a')[0].text, # Name
            tds[3].text,
            'USD', now, now))
    return symbols

def insert_snp500_symbols(symbols):
    """
        Insert the S&P500 symbols into the MySQL database.
    """
    # Connect to the MySQL instance
    db_host = 'localhost'
    db_user = 'sec_user'
    db_pass = 'alfabeta33'
    db_name = 'securities_master'
    con = mdb.connect(
        host=db_host, user=db_user, passwd=db_pass, db=db_name
    )

#
#     cur = con.cursor()
#
#     cur.execute(" CREATE TABLE data_vendor (id int NOT NULL AUTO_INCREMENT, \
#         name varchar(64) NOT NULL, \
#     website_url varchar(255) NULL, \
#     support_email varchar(255) NULL, \
#     created_date datetime NOT NULL, \
#     last_updated_date datetime NOT NULL, \
#     PRIMARY KEY(id)) \
#     ENGINE = InnoDB AUTO_INCREMENT = 1 DEFAULT CHARSET = utf8; \
#     "
#    )
#
#     cur.execute("CREATE TABLE symbol ( \
# id int NOT NULL AUTO_INCREMENT, \
# exchange_id int NULL, \
# ticker varchar(32) NOT NULL, \
# instrument varchar(64) NOT NULL, \
# name varchar(255) NULL, \
# sector varchar(255) NULL, \
# currency varchar(32) NULL, \
#   created_date datetime NOT NULL, \
#   last_updated_date datetime NOT NULL, \
#   PRIMARY KEY (id), \
#    KEY index_exchange_id (exchange_id)) \
#             ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;")
#
#
#     cur.execute("CREATE TABLE daily_price ( \
#   id int NOT NULL AUTO_INCREMENT, \
# data_vendor_id int NOT NULL,\
# symbol_id int NOT NULL, \
# price_date datetime NOT NULL, \
# created_date datetime NOT NULL, \
# last_updated_date datetime NOT NULL, \
#  open_price decimal(19,4) NULL, \
# high_price decimal(19,4) NULL, \
# low_price decimal(19,4) NULL, \
# close_price decimal(19,4) NULL, \
# adj_close_price decimal(19,4) NULL, \
# volume bigint NULL, \
#  PRIMARY KEY (id), \
# KEY index_data_vendor_id (data_vendor_id), \
# KEY index_symbol_id (symbol_id) \
# ) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;")
#

    # Create the insert strings
    column_str = """ticker, instrument, name, sector, currency, created_date, last_updated_date"""
    insert_str = ("%s, " * 7)[:-2]
    final_str = """INSERT INTO symbol (%s) VALUES (%s)""" %  (column_str, insert_str)
    # Using the MySQL connection, carry out
    # an INSERT INTO for every symbol

    for i, symb in enumerate(symbols):
        temp_symb5 = symbols[i][5].strftime('%Y-%m-%d %H:%M:%S')
        temp_symb6 = symbols[i][6].strftime('%Y-%m-%d %H:%M:%S')
        symbols[i] = (symbols[i][0], symbols[i][1], symbols[i][2],symbols[i][3],symbols[i][4],
                      temp_symb5, temp_symb6)

    with con:
        cur = con.cursor()
        cur.executemany(final_str, symbols)
        con.commit()

if __name__ == "__main__":
    symbols = obtain_parse_wiki_snp500(features="html.parser" )
    insert_snp500_symbols(symbols)
    print("%s symbols were successfully added." % len(symbols))


