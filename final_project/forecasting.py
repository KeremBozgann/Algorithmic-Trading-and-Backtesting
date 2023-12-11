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


def create_data(lags, train_ratio, val_ratio, test_ratio, symbol, append_sum):
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
    if append_sum:
        X['total perc'] = (1 + X['Lag1'] / 100) * (1 + X['Lag2'] / 100) * (1 + X['Lag3'] / 100) * (1 + X['Lag4'] / 100) - 1
        X['total perc'] *= 100

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
# lags = 4
lags = 5

# train, validation, test data split ratios
train_ratio = 0.5
val_ratio = 0.5
test_ratio = 0.0

# symbol: stock symbol
# symbol = "GOOGL"
# symbol = "SPY"



append_sign = False
only_sign = False
append_sum = False

# symbol = "AAPL"
symbol = "AMZN"
# symbol = "TSLA"




symbol_list_keen_perc =["AAPL", "AMZN",  "TSLA"]

# X_train_keen_perc_list = list()
X_train_multi_source_list = list()
# X_val_keen_perc_list = list()
X_val_multi_source_list = list()
# X_test_keen_perc_list = list()
X_test_multi_source_list = list()

# y_train_keen_perc_list = list()
y_train_multi_source_list = list()
# y_val_keen_perc_list = list()
y_val_multi_source_list = list()
# y_test_keen_perc_list = list()
y_test_multi_source_list = list()

for symbol_keen_perc in symbol_list_keen_perc:
    X_train, y_train, X_val, y_val, X_test, y_test = create_data(lags, train_ratio, val_ratio, test_ratio, symbol, append_sum)

    X_train_multi_source_list.append(X_train)
    X_val_multi_source_list.append(X_val)
    X_test_multi_source_list.append(X_test)

    y_train_multi_source_list.append(y_train)
    y_val_multi_source_list.append(y_val)
    y_test_multi_source_list.append(y_test)



# your model goes here.
# model = QDA()

model_log = LogisticRegression(fit_intercept = True)
model_multi_source_log = LogisticRegression(fit_intercept = True)
# model = LDA()
# model = SVC()
# model = LogisticRegression()

X_train, y_train, X_val, y_val, X_test, y_test = create_data(lags, train_ratio, val_ratio, test_ratio, symbol, append_sum)
if append_sum:
    X_train_sign = np.sign(X_train.drop('total perc', axis=1))
    X_val_sign = np.sign(X_val.drop('total perc', axis=1))
    is_negative = (X_train.drop('total perc', axis=1) < 0).all(axis=1)
    is_negative_val = (X_val.drop('total perc', axis=1) < 0).all(axis=1)
else:
    X_train_sign = np.sign(X_train)
    X_val_sign = np.sign(X_val)
    is_negative = (X_train < 0).all(axis=1)
    is_negative_val = (X_val< 0).all(axis=1)


if append_sign:
    X_train_consec_neg = np.append(X_train[is_negative], X_train_sign[is_negative], axis=1)
    y_train_consec_neg = y_train[is_negative]

    X_val_consec_neg = np.append(X_val[is_negative_val], X_val_sign[is_negative_val] , axis =1)
    y_val_consec_neg = y_val[is_negative_val]

elif only_sign:
    X_train_consec_neg = X_train_sign[is_negative]
    y_train_consec_neg = y_train[is_negative]

    X_val_consec_neg = X_val_sign[is_negative_val]
    y_val_consec_neg = y_val[is_negative_val]
else:
    X_train_consec_neg = X_train[is_negative]
    y_train_consec_neg = y_train[is_negative]

    if append_sum:
        is_negative_val = (X_val.drop('total perc', axis=1) < 0).all(axis=1)
    else:
        is_negative_val = (X_val < 0).all(axis=1)

    X_val_consec_neg = X_val[is_negative_val]
    y_val_consec_neg = y_val[is_negative_val]



if append_sign:
    X_train_consec_neg_multi_source = np.zeros([0, X_train.shape[1] + lags])
    X_val_consec_neg_multi_source = np.zeros([0, X_train.shape[1] + lags])
    # X_test_consec_neg_keen_perc = np.zeros([0, X_train.shape[1] + lags])
    X_test_consec_neg_multi_source = np.zeros([0, X_train.shape[1] + lags])
else:
    # X_train_consec_neg_keen_perc = np.zeros([0,lags])
    X_train_consec_neg_multi_source = np.zeros([0,X_train.shape[1] ])
    # X_val_consec_neg_keen_perc= np.zeros([0,lags])
    X_val_consec_neg_multi_source= np.zeros([0,X_train.shape[1] ])
    # X_test_consec_neg_keen_perc = np.zeros([0,lags])
    X_test_consec_neg_multi_source = np.zeros([0,X_train.shape[1] ])


# y_train_consec_neg_keen_perc = np.zeros([0])
y_train_consec_neg_multi_source = np.zeros([0])
# y_val_consec_neg_keen_perc = np.zeros([0])
y_val_consec_neg_multi_source = np.zeros([0])
# y_test_consec_neg_keen_perc = np.zeros([0])
y_test_consec_neg_multi_source = np.zeros([0])

for i in range(len(X_train_multi_source_list)):
    if append_sum:
        _X_train_multi_source_sign = np.sign(X_train_multi_source_list[i].drop('total perc', axis=1))
        _X_val_multi_source_sign = np.sign(X_val_multi_source_list[i].drop('total perc', axis=1))
    else:
        _X_train_multi_source_sign = np.sign(X_train_multi_source_list[i])
        _X_val_multi_source_sign = np.sign(X_val_multi_source_list[i])

    is_negative_train_multi_source = (_X_train_multi_source_sign < 0).all(axis=1)
    is_negative_val_multi_source = (_X_val_multi_source_sign < 0).all(axis=1)

    if append_sign:
        _X_train_consec_neg_multi_source = np.append(X_train_multi_source_list[i][is_negative_train_multi_source],
                                                  _X_train_multi_source_sign[is_negative_train_multi_source], axis=1)
        _X_val_consec_neg_multi_source = np.append(X_val_multi_source_list[i][is_negative_val_multi_source],
                                                _X_val_multi_source_sign[is_negative_val_multi_source], axis=1)
    elif only_sign:
        _X_train_consec_neg_multi_source = _X_train_multi_source_sign[is_negative_train_multi_source]
        _X_val_consec_neg_multi_source = _X_val_multi_source_sign[is_negative_val_multi_source]

    else:
        _X_train_consec_neg_multi_source = X_train_multi_source_list[i][is_negative_train_multi_source]
        _X_val_consec_neg_multi_source = X_val_multi_source_list[i][is_negative_val_multi_source]

    _y_train_consec_neg_multi_source = y_train_multi_source_list[i][is_negative_train_multi_source]
    _y_val_consec_neg_multi_source = y_val_multi_source_list[i][is_negative_val_multi_source]

    X_train_consec_neg_multi_source = np.append(X_train_consec_neg_multi_source, _X_train_consec_neg_multi_source, axis = 0)
    X_val_consec_neg_multi_source = np.append(X_val_consec_neg_multi_source, _X_val_consec_neg_multi_source, axis = 0)

    y_train_consec_neg_multi_source = np.append(y_train_consec_neg_multi_source, _y_train_consec_neg_multi_source)
    y_val_consec_neg_multi_source = np.append(y_val_consec_neg_multi_source, _y_val_consec_neg_multi_source)


# model.fit(X_train, y_train)

# #
# model_perc.fit(X_train, y_train)
# model_perc.partial_fit(X_train_consec_neg, y_train_consec_neg)

model_log.fit(X_train_consec_neg, y_train_consec_neg)
y_val_consec_neg_pred =model_log.predict(X_val_consec_neg)
y_val_pred_always_neg = 1 * np.ones([len(y_val_consec_neg.to_numpy())])



model_multi_source_log.fit(X_train_consec_neg_multi_source, y_train_consec_neg_multi_source)
y_val_consec_neg_multi_source_pred =model_multi_source_log.predict(X_val_consec_neg)
y_val_consec_neg_multi_source_pred_always_neg = 1 * np.ones([len(y_val_consec_neg)])

y_val_consec_neg_multi_source_pred_probs =model_multi_source_log.predict_proba(X_val_consec_neg)
threshold = 0.6
multi_source_where_confident =(y_val_consec_neg_multi_source_pred_probs[:, 1] > threshold)


print("accuracy score Logistic Regression", accuracy_score(y_val_consec_neg_pred, y_val_consec_neg))
print("accuracy score Multi-Source-Logistic Regression", accuracy_score(y_val_consec_neg_multi_source_pred, y_val_consec_neg))

print(f"accuracy score Multi-Source-Logistic Regression confident predictions with {threshold} threshold:",
                    accuracy_score(y_val_consec_neg_multi_source_pred[multi_source_where_confident],
                                   y_val_consec_neg[multi_source_where_confident]))
print("accuracy score Rule Based", accuracy_score(y_val_pred_always_neg, y_val_consec_neg))


print(f"num total consecutive negative instances: {len(y_val_consec_neg_multi_source_pred)}, "
            f"num consecutive negative instances where model is confident: {np.sum(multi_source_where_confident)}")

#confusion matrix:
matrix = confusion_matrix(y_val_consec_neg, y_val_consec_neg_pred)
print(matrix)
# print(matrix.diagonal()/matrix.sum(axis=1))





