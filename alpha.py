from __future__ import print_function
import datetime
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from strategy import Strategy
from event import SignalEvent
from backtest import Backtest
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from forecast import create_lagged_series
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from keras import Sequential
from keras import layers, regularizers
from sklearn.preprocessing import StandardScaler

class DailyForecastStrategy(Strategy):
    """
    Forecast strategy. Uses an alpha model to predict
    the movement of the stock price and then generate long/exit signals (buy/sell signals) based on the prediction.
    """

    def __init__(self, bars, events, model_name, start_train_date, end_train_date, start_test_date):
        self.bars = bars

        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.datetime_now = datetime.datetime.utcnow()
        self.model_start_date = start_train_date
        self.model_end_date = end_train_date
        self.model_start_test_date = start_test_date

        self.flag_buy_and_sell = False
        self.long_market = False
        self.short_market = False
        self.bar_index = 0
        self.model_name = model_name
        self.model = self.create_symbol_forecast_model()

    def create_symbol_forecast_model(self):

        if self.model_name ==  "Rule Based":
            return

        elif self.model_name ==  "Confident Logistic Regression":
            snpret = create_lagged_series(
                self.symbol_list[0], self.model_start_date, self.model_end_date, lags=4
            )
            # Get the data with window size of 4 (use previous stock prices of previous 4 days as input)
            X = snpret[["Lag1", "Lag2", "Lag3", "Lag4"]]
            y = snpret["Direction"]
            start_test = self.model_start_test_date

            X.index = pd.to_datetime(X.index)
            y.index = pd.to_datetime(y.index)

            X_train = X[X.index < start_test]
            X_train = X_train.to_numpy()

            y_train = y[y.index < start_test]
            y_train = y_train.to_numpy()


            # Get the instances where a consecutive decrease is observed. This training data is only used for Confident-Logistic regression
            is_negative = (X_train < 0).all(axis=1)
            X_train_consec_neg = X_train[is_negative]
            y_train_consec_neg = y_train[is_negative]
            model_keen_log = LogisticRegression(fit_intercept=True)
            model_keen_log.fit(X_train_consec_neg, y_train_consec_neg)
            return model_keen_log

        # Logistic regression when the percentage change between day0 and day 4 is added as a feature (called total perc)
        elif self.model_name ==  "Logistic Regression with Sum of Percentage Change Input":
            snpret = create_lagged_series(
                self.symbol_list[0], self.model_start_date, self.model_end_date, lags=4
            )
            X = snpret[["Lag1", "Lag2", "Lag3", "Lag4"]]
            X['total perc'] = (1 + X['Lag1']/100) * (1 + X['Lag2']/100) * (1 + X['Lag3']/100) * (1 + X['Lag4']/100) - 1
            X['total perc'] *= 100  # Convert from decimal to percentage

            y = snpret["Direction"]
            start_test = self.model_start_test_date

            X.index = pd.to_datetime(X.index)
            y.index = pd.to_datetime(y.index)

            X_train = X[X.index < start_test]
            X_train = X_train.to_numpy()

            y_train = y[y.index < start_test]
            y_train = y_train.to_numpy()

            is_negative = (X_train < 0).all(axis=1)
            X_train_consec_neg = X_train[is_negative]
            y_train_consec_neg = y_train[is_negative]

            model_keen_log = LogisticRegression(fit_intercept=True)

            model_keen_log.fit(X_train_consec_neg, y_train_consec_neg)

            return model_keen_log

        # For all the other machine learning models that trained on all the data
        else:
            snpret = create_lagged_series(
            self.symbol_list[0], self.model_start_date, self.model_end_date, lags = 5
            )

            # Get data
            X = snpret[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5"]]
            y = snpret["Direction"]

            start_test =  self.model_start_test_date
            X.index = pd.to_datetime(X.index)
            y.index = pd.to_datetime(y.index)

            # get train data
            X_train = X[X.index < start_test]
            X_train =X_train.to_numpy()

            y_train = y[y.index < start_test]
            y_train = y_train.to_numpy()


            if self.model_name == "QDA":
                model = QDA()

            elif self.model_name == "LDA":
                model = LDA()

            elif self.model_name == "LDA_BAGG":
                lda = LDA()
                model = BaggingClassifier(base_estimator=lda, n_estimators=10, random_state=0)

            elif self.model_name == "Perceptron":
                model = Perceptron(fit_intercept = True)
            elif self.model_name == "RandomForestClassifier":
                model = RandomForestClassifier(n_estimators=20, random_state=42)
            elif self.model_name == "Gradient Boosting":
                params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 50}
                model = xgb.XGBClassifier(**params)
                y_train[y_train == -1] = 0

            elif self.model_name == "ANN":
                # normalizing the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)

                model = Sequential([
                    # dense layers used for feedforward neural network
                    # input shape with 64 neurons, ReLU activation function - ouputs input directly if positive, otherwise output is zero
                    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.01),
                                 input_shape=(X_train.shape[1],)),
                    # drops half of the input units during training
                    # layers.Dropout(0.5),
                    # hidden layer with 32 neurons and ReLU activation function - used for learning patterns and representations in the data
                    layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01), ),
                    # layers.Dropout(0.5),
                    # output later with one neuron and tanh activation function - sigmoid commonly used in binary classigication - produces probability score.
                    layers.Dense(1, activation='tanh')
                    # tanh was the one we need to use bc it produces classification vals between -1 and 1 which we need to pass into backtesting
                ])

                # Adam optimizer combines RMSprop and momentum - faster convergence. two moving averages that are updated with moving average on squared and original gradients. (average of squared gradients = learning rate)
                # binary crossentropy measures difference between predicted and true probability distributiion
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                # optimizers tried: sgd, adam, adagrad.
                # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

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

                # Create input data:
                lags_norm = pd.DataFrame({'Lags': [lags[0], lags[1], lags[2], lags[3],
                                                   lags[4], lags[5]]}).pct_change() * 100

                pred_series = pd.Series(
                {
                'Lag1': lags_norm.loc[1, 'Lags'] ,
                'Lag2': lags_norm.loc[2, 'Lags'] ,
                'Lag3': lags_norm.loc[3, 'Lags'] ,
                    'Lag4': lags_norm.loc[4, 'Lags'] ,
                    'Lag5': lags_norm.loc[5, 'Lags'] ,

                }
                    )

                # rule based model prediction and buy/sell signal generation
                if self.model_name == "Rule Based":
                    if self.flag_buy_and_sell:
                        signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                        self.events.put(signal)
                        self.flag_buy_and_sell = False
                    else:
                        # consider only those instances where a 5 days of consecutive decrease is observed.  Otherwise, don't trade
                        if np.all(pred_series.values.reshape(1, -1)[-4:] < 0):
                            pred = 1
                            self.flag_buy_and_sell = True
                            signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                            self.events.put(signal)

                elif self.model_name ==  "Confident Logistic Regression":
                    if self.flag_buy_and_sell:
                        signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                        self.events.put(signal)
                        self.flag_buy_and_sell = False
                    else:
                        # consider only those instances where a 5 days of consecutive decrease is observed. Otherwise, don't trade
                        if np.all(pred_series.values.reshape(1, -1)[0, :5]< 0):
                            pred_prob = self.model.predict_proba(pred_series.values[:4].reshape(1, -1))
                            pred_inc = pred_prob[0, 1]
                            if pred_inc > 0.6:
                                pred = 1
                                self.flag_buy_and_sell = True
                                signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                                self.events.put(signal)

                elif  self.model_name ==  "Logistic Regression with Sum of Percentage Change Input":
                    if self.flag_buy_and_sell:
                        signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                        self.events.put(signal)
                        self.flag_buy_and_sell = False
                    else:
                        if np.all(pred_series.values.reshape(1, -1)[0, :4] < 0):
                            first_4_lags = pred_series.values[:4].reshape(1, -1)
                            sum = (1 + pred_series['Lag1'] / 100) * (1 + pred_series['Lag2'] / 100) * (1 + pred_series['Lag3'] / 100) * (
                                        1 + pred_series['Lag4'] / 100) - 1
                            sum *= 100  # Convert from decimal to percentage

                            sum_added = np.append(first_4_lags, np.array([[sum]]), axis = 1)
                            pred_prob = self.model.predict_proba(sum_added)
                            pred_inc = pred_prob[0, 1]
                            if pred_inc > 0.5:
                                pred = 1
                                self.flag_buy_and_sell = True
                                signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                                self.events.put(signal)
                else:
                    pred = self.model.predict(pred_series.values.reshape(1,-1))
                    if pred > 0 and not self.long_market:
                        self.long_market = True
                        signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                        self.events.put(signal)
                    if pred < 0 and self.long_market:
                        self.long_market = False
                        signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                        self.events.put(signal)

#Run the forecast simulation
def run_forecast(symbol_list, initial_capital, trade_volume, model_name, start_train_date, end_train_date, start_test_date
                     ,end_test_date):
    # data dir
    csv_dir = 'data'

    heartbeat = 0.0
    start_date = start_train_date

    # start the simulation
    backtest = Backtest(
        csv_dir, symbol_list, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler, Portfolio, DailyForecastStrategy, model_name,
        start_train_date, end_train_date, start_test_date, end_test_date
    )

    # return the results
    total_gain , returns, equity_curve, drawdown = backtest.simulate_trading(trade_volume)

    return total_gain, returns, equity_curve, drawdown