
Python Dependencies:
    - numpy
    - pandas
    - pandas_datareader (pip install pandas_datareader)
    - yfinance
    - scikit-learn
    - matplotlib
    - xgboost
    - tensorflow
    - keras
    - seaborn
    - PyPortfolioOpt


Instructions to reproduce the equity curves in the final report:
 1) Go to "main.py" file
 2) Set "model_name" to desired alpha model
 3) Set selected risk to desired risk tolerance (between 0 to 100)
 4) Set "stock_symbols" to the desired list of stocks. If the model name is "Rule Based" or "Confident Logistic Regression", use only
    1 stock.
    Using multiple stocks in the stock_symbols list will activate the Beta model.
 5) Run the main.py file and observe the equity curve at the end.

Instructions to reproduce the accuracies of machine learning models (LDA, QDA, ANN, RandomForest, XGBoost, SVC) in the final report:
    1) Go to "final_project/forecasting_machine_learning.py"
    2) Set model_name to desired alpha model.
    3) Set symbol to the desired stock symbol to test the models on
    4) Run the file and observe the accuracy and confusion matrix

Instructions to reproduce the accuracies of Rule-Based and Confident Logistic Regression models in the final report:
    1) Go to "final_project/forecasting_rule_based.py"
    2) Set the symbol variable to desired stock symbol
    3) Run the file
    4) Observe the accuracy scores for Rule-Based, Multi-Source-Confident Logistic Regression and Multi-Source-Logistic Regression




"main.py": the main file and runs a forecaster from the sklearn library for predicting the next day market price of the stock based on past data.

"final_project/forecasting_rule_based.py":
            To test the accuracy of the following models in isolation: "Rule Based", "Confident Logistic Regression"
            These models can be tested in isolation (in terms of accuracy, isolated from the backtesting (simulation) environment ) in this file
"final_project/forecasting_machine_learning.py":
            To test the accuracy of the following models in isolation: "ANN", "Logistic Regression" , "LDA", "QDA", "RandomForestClassifier",
             "Gradient Boosting" and "SVC".
            These models can be tested in isolation (in terms of accuracy, isolated from the backtesting (simulation) environment ) in this file

"data" folder: Where daily stock price data for different stocks are stored.

"download_and_save_historic_market_stock_data_for_backtest.py":
            Use this file to create new stock price data. Set the variable "tick" to desired stock symbol to get the data for that symbol


"stocks_chart_reviewer.py": Creates the stock price increase/decrease ratio charts.

"inspect_price_movement_around_consecutive_decrease_days.py": Plotting tool to visualize the stock price change around the days which are
                                            succeeded by 5 consecutive days of decreases, in an attempt to get insights on data
