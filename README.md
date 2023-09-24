# Algorithmic-Trading-and-Backtesting
Implementation of a Trading Algorithm that uses a Support Vector Machine (SVN) to predict stock price movements. 


Make sure to install the "kivy" package of python before you run the "main.py" file which is used for the User Interface. 

Run "main.py", which is the main file and runs a "QDA" forecaster from the sklearn library for predicting the next day market price of the stock based on past data. 

The algorithm is risk sensitive and user inputs her risk sensitivity via sliding a risk bar in the GUI that is prompted after running the "main.py" file. 
Note that this is a very simple risk sensitivity model. The risk bar determines the number of stocks that is traded at each transaction, a higher risk corresponding to
a larger number of stocks exchanged at each transaction (up to 500) and the lower risk value corresponds to a smaller number of stocks (100 minimum). 

At the GUI, user is also expected to enter the symbols she wants algorithm to trade on (currently we have only SPY and AAPL stocks). You need to create a folder named "data" (at the 
same level as "main.py" file and download the data from yahoo finance using the "download_and_save_historic_market_stock_data_for_backtest.py" which resides inside the "other" folder. 
In this file, change the "tick" variable to the desired tick symbol to download the historic data for that symbol, which is saved inside the "data" folder upon running "download_and_save_historic_market_stock_data_for_backtest.py". 

User is also expected to input an initial amount of capital. Make sure to enter an amount that is large enough so that multiple stocks can be purchased. We recommend an initial 
capital value between $50,000 and $100,000. 


