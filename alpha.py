from __future__ import print_function
import datetime
import pandas as pd
# from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import numpy as np
import yfinance as yf
from sklearn.svm import SVC
import xgboost as xgb

from keras import Sequential
from keras import layers
from strategy import Strategy
from event import SignalEvent
from backtest import Backtest
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from forecast import create_lagged_series
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
# snp_forecast.py

from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron



class SPYDailyForecastStrategy(Strategy):
    """
    S&P500 forecast strategy. It uses a Quadratic Discriminant Analyser to predict
    the returns for a subsequent time period and then generated long/exit signals based on the prediction.
    """

    def __init__(self, bars, events, model_name, start_train_date, end_train_date, start_test_date):
        self.bars = bars

        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.datetime_now = datetime.datetime.utcnow()
        self.model_start_date = datetime.datetime(2001, 1, 10)
        self.model_end_date = datetime.datetime(2005, 12, 31)
        self.model_start_test_date = datetime.datetime(2018, 1, 1)
        self.long_market = False
        self.short_market = False
        self.bar_index = 0
        self.model_name = model_name
        self.model = self.create_symbol_forecast_model()

    def retrieve_data(self):
        time_series = pd.read_csv(f'data/{self.symbol_list[0]}.csv')
        
        # Create a lagged series of the S&P500 US stock market index
        snpret = create_lagged_series(
        self.symbol_list[0], self.model_start_date, self.model_end_date, lags = 5
        )
        # Use the prior two days of returns as predictor # values, with direction as the response
        # X = snpret[["Lag1", "Lag2"]]
        X = snpret[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5"]]
        # X = snpret[["Lag1", "Lag2"]]

        y = snpret["Direction"]
        # Create training and test sets
        # split into testing and training
        start_test = pd.to_datetime(self.model_start_test_date)
        #start_test = snpret.index[0] +  pd.DateOffset(years=3)
        X.index = pd.to_datetime(X.index)
        y.index = pd.to_datetime(y.index)

        # make sure the data is indexed by the date
        time_series = time_series.set_index('datetime')
        time_series.index = pd.to_datetime(time_series.index)

        #splitting time series into train and test
        train_ts = time_series[time_series.index < start_test]
        test_ts = time_series[time_series.index >= start_test]

        #start_test = self.model_start_test_date (declared a few lines above)
        X_train = X[X.index < start_test]
        X_train = np.flip(X_train.to_numpy(),axis= 1)
        X_test = X[X.index >= start_test]
        X_test=  np.flip(X_test.to_numpy(), axis = 1)
        y_train = y[y.index < start_test]
        y_train = y_train.to_numpy()
        y_test = y[y.index >= start_test]
        y_test = y_test.to_numpy()

        #returns X train and y train for (1) lagged series and (2) original data
        return X_train, y_train, X_test, y_test, train_ts, test_ts

    def create_symbol_forecast_model(self):

        # initial data before transforming it to a lagged series
        # time_series = yf.download(
        # self.symbol_list[0],(self.model_start_date - datetime.timedelta(days=365)).strftime('%Y-%m-%d'),
        # self.model_end_date.strftime('%Y-%m-%d'))
        X_train, y_train, X_test, y_test, train_ts, test_ts = self.retrieve_data()

       
        start_test = pd.to_datetime(self.model_start_test_date)
        

        x_train_ts = train_ts[['open','high','low','volume']]
        y_train_ts = train_ts['close']

        x_test_ts = test_ts[['open','high','low','volume']]
        y_test_ts = test_ts['close']
       
       
        if self.model_name == "QDA":
            model = QDA()

        elif self.model_name == "LDA":
            model = LDA()

        elif self.model_name == "LDA_BAGG":
            lda = LDA()
            model = BaggingClassifier(base_estimator=lda, n_estimators=10, random_state=0)

        elif self.model_name == "Random_Forest":

            # train random forest model
            params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 50}
            model = xgb.XGBRFClassifier(**params)
            y_train[y_train==-1] = 0
            # model seems extremely optimistic with SPY data but works poor with AAPL data
            # check for multicollinearity or other issues

        elif self.model_name == "Gradient_Boosting":

            # model = xgb.XGBClassifier(n_estimators=100,objective='multi:softmax')
            params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 50}
            model = xgb.XGBClassifier(**params)
            y_train[y_train==-1] = 0
            # same issues as with Random Forest
        
        elif self.model_name == "Perceptron":
            model = Perceptron(fit_intercept = True)

        elif self.model_name == "Sequential":
            # normalizing the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            model = Sequential([
                # dense layers used for feedforward neural network
                # input shape with 64 neurons, ReLU activation function - ouputs input directly if positive, otherwise output is zero
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                # hidden layer with 32 neurons and ReLU activation function - used for learning patterns and representations in the data
                layers.Dense(32, activation='relu'),
                # output later with one neuron and sigmoid activation function - sigmoid commonly used in binary classigication - produces probability score. 
                layers.Dense(1, activation='sigmoid')
            ])

            # Adam optimizer combines RMSprop and momentum - faster convergence. two moving averages that are updated with moving average on squared and original gradients. (average of squared gradients = learning rate)
            # binary crossentropy measures difference between predicted and true probability distributiion
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            # optimizers tried: sgd, adam, adagrad
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # model = SVC()
        # model = LogisticRegression()
        if self.model_name == "Sequential":
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        else:
            model.fit(X_train, y_train)
        return model

    def calculate_signals(self, event):
        """
            Calculate the SignalEvents based on market data.
        """

        sym = self.symbol_list[0]
        dt = self.datetime_now
        if event.type == 'MARKET':
            self.bar_index += 1
            if self.bar_index > 5:
                lags = self.bars.get_latest_bars_values(
                    self.symbol_list[0], "adj_close", N=6
                )
                # lags_norm = pd.DataFrame({'Lag1': [lags[1]],
                #                           'Lag2':[lags[2]] ,'Lag3':[lags[3]] ,'Lag4': [lags[4]],'Lag5': [lags[5]]})
                lags_norm = pd.DataFrame({'Lags': [lags[0], lags[1], lags[2], lags[3],
                                                   lags[4], lags[5]]}).pct_change() * 100
                # lags_norm.loc[0, 'Lags'] = 0.01

                # pred_series = pd.Series(
                # {
                # 'Lag1': lags[1] * 100.0,
                # 'Lag2': lags[2] * 100.0
                # }
                #     )

                pred_series = pd.Series(
                {
                'Lag1': lags_norm.loc[1, 'Lags'] ,
                'Lag2': lags_norm.loc[2, 'Lags'] ,
                'Lag3': lags_norm.loc[3, 'Lags'] ,
                    'Lag4': lags_norm.loc[4, 'Lags'] ,
                    'Lag5': lags_norm.loc[5, 'Lags'] ,

                }
                    )
    
                # pred = self.model.predict(pred_series.values.reshape(1,-1))
                
                # for i in range(1,len(arima_pred)):
                #     arima_lagged[i] = (arima_pred[i]-arima_pred[i-1])/arima_pred[i-1]
                
            
                # pred = self.model.predict(pred_series)
                
                pred = self.model.predict(pred_series.values.reshape(1,-1))

                # pred = self.model.predict(pd.DataFrame({'Lag1':[pred_series.values[0]], 'Lag2':[pred_series.values[1]]}))
                if pred > 0 and not self.long_market:
                    self.long_market = True
                    signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                    self.events.put(signal)
                if pred <= 0 and self.long_market:
                    self.long_market = False
                    signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                    self.events.put(signal)



if __name__ == "__main__":
    csv_dir = 'data'  # CHANGE THIS!
    symbol_list = ['SPY']
    initial_capital = 100000.0
    heartbeat = 0.0
    start_date = datetime.datetime(2006, 1, 3)
    backtest = Backtest(
        csv_dir, symbol_list, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler, Portfolio, SPYDailyForecastStrategy
    )
    backtest.simulate_trading()

def run_snp_forecast(symbol_list, initial_capital, trade_volume, model_name, start_train_date, end_train_date, start_test_date):
    csv_dir = 'data'  # CHANGE THIS!
    # symbol_list = ['SPY']
    # initial_capital = 100000.0

    heartbeat = 0.0
    # start_year = 2006
    # start_year = 2015
    # start_date = datetime.datetime(start_train_date, 1, 3)
    start_date = start_train_date

    backtest = Backtest(
        csv_dir, symbol_list, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler, Portfolio, SPYDailyForecastStrategy, model_name,
        start_train_date, end_train_date, start_test_date
    )
    total_gain , returns, equity_curve, drawdown = backtest.simulate_trading(trade_volume)

    return total_gain, returns, equity_curve, drawdown



def run_adaboost_decision_stumps_forecast(symbol_list, initial_capital, trade_volume):
    csv_dir = 'data'  # CHANGE THIS!
    # symbol_list = ['SPY']
    # initial_capital = 100000.0

    heartbeat = 0.0
    start_date = datetime.datetime(2006, 1, 3)
    backtest = Backtest(
        csv_dir, symbol_list, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler, Portfolio, DecisionStumpForecastStrategy
    )
    total_gain , returns, equity_curve, drawdown = backtest.simulate_trading(trade_volume)

    return total_gain, returns, equity_curve, drawdown