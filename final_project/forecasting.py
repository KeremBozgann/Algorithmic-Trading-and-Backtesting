import numpy as np
import os
import pandas as pd
import sys
# from forecast import create_lagged_series
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
# from sklearn.utils import MovingAverage
from sklearn.linear_model import Perceptron


def create_data(lags, train_ratio, val_ratio, test_ratio, symbol):
    sys.path.append('../')

    path= os.path.join("..", 'data', symbol+ ".csv")
    ts = pd.read_csv(path)

    ts['datetime'] = pd.to_datetime(ts['datetime'], format='%Y/%m/%d')
    start_date = '2010/01/01'
    end_date = '2018/01/01'
    start_date = pd.to_datetime(start_date, format='%Y/%m/%d')
    end_date = pd.to_datetime(end_date, format='%Y/%m/%d')
    ts = ts[(ts['datetime'] >= start_date) & (ts['datetime'] < end_date)]

    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["adj_close"]
    tslag["Volume"] = ts["volume"]

    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag["Lag%s" % str(i + 1)] = ts["adj_close"].shift(i + 1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change() * 100.0

    # If any of the values of percentage returns equal zero, set them to a small number (stops issues with QDA model in Scikit-Learn)
    for i, x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        tsret["Lag%s" % str(i + 1)] = \
            tslag["Lag%s" % str(i + 1)].pct_change() * 100.0
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret.iloc[lags+1:]


    X = tsret[[f"Lag{i+1}" for i in range(lags)]]
    y = tsret["Direction"]

    num_data = X.shape[0]

    num_train = int(num_data * train_ratio )

    num_valid_test = num_data - num_train

    num_val = int(num_data * val_ratio)
    num_test = num_valid_test - num_val

    X_train  = X[:num_train]
    y_train = y[:num_train]

    X_val = X[num_train:num_train + num_val]
    y_val = y[num_train:num_train + num_val]

    X_test = X[num_train + num_val: ]
    y_test = y[num_train + num_val: ]

    return X_train, y_train, X_val, y_val, X_test, y_test

    # tsret = tsret[tsret.index >= start_date]


# lags: number of days we look before the day we want to make a prediction
lags = 4

# train, validation, test data split ratios
train_ratio = 0.5
val_ratio = 0.5
test_ratio = 0.0

# symbol: stock symbol
# symbol = "SPY"
symbol = "AAPL"
# symbol = "GOOGL"
X_train, y_train, X_val, y_val, X_test, y_test = create_data(lags, train_ratio, val_ratio, test_ratio, symbol)



# your model goes here.
# model = QDA()

model = Perceptron(fit_intercept = True)
# model = LDA()
# model = SVC()
# model = LogisticRegression()


X_train_sign = np.sign(X_train)

is_negative = (X_train < 0).all(axis=1)
X_train_consec_neg = X_train[is_negative]
y_train_consec_neg = y_train[is_negative]

is_negative_val = (X_val < 0).all(axis=1)
X_val_consec_neg = X_val[is_negative_val]
y_val_consec_neg = y_val[is_negative_val]

# model.fit(X_train, y_train)

#
# model.fit(X_train, y_train)
# model.partial_fit(X_train_consec_neg, y_train_consec_neg)

model.fit(X_train_consec_neg, y_train_consec_neg)

y_val_consec_neg_pred =model.predict(X_val_consec_neg)

print("accuracy score", accuracy_score(y_val_consec_neg_pred, y_val_consec_neg))



#confusion matrix:
matrix = confusion_matrix(y_val_consec_neg, y_val_consec_neg_pred)
print(matrix)
# print(matrix.diagonal()/matrix.sum(axis=1))





