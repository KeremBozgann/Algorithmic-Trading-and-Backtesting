import os
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix


'''Tests the alpha (forecasting) models in isolation (isolated from the simulation environment) to test them for accuracy'''
def create_data(lags, train_ratio, val_ratio, test_ratio, symbol, append_sum):

    #get data
    sys.path.append('../')
    path= os.path.join("..", 'data', symbol+ ".csv")
    ts = pd.read_csv(path)
    ts['datetime'] = pd.to_datetime(ts['datetime'], format='%Y/%m/%d')
    start_date = '2010/01/01'
    end_date = '2018/01/01'

    #consider start date-end date portion of the data
    start_date = pd.to_datetime(start_date, format='%Y/%m/%d')
    end_date = pd.to_datetime(end_date, format='%Y/%m/%d')
    ts = ts[(ts['datetime'] >= start_date) & (ts['datetime'] < end_date)]
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["adj_close"]
    tslag["Volume"] = ts["volume"]

    # Create the shifted lag series (Window size is determined by the lags parameter)
    for i in range(0, lags):
        tslag["Lag%s" % str(i + 1)] = ts["adj_close"].shift(i + 1)
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change() * 100.0


    # If zero is observed, set it to a small value to prevent numerical instability issues
    for i, x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    #convert price signal to percentage change signal
    for i in range(0, lags):
        tsret["Lag%s" % str(i + 1)] = \
            tslag["Lag%s" % str(i + 1)].pct_change() * 100.0

    # Create the output (+1 or -1, +1 for increase, -1 for decrease)
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret.iloc[lags+1:]


    #Add the total percentage change from first day to (lags)th day as a new feature
    X = tsret[[f"Lag{i+1}" for i in range(lags)]]
    if append_sum:
        X['total perc'] = (1 + X['Lag1'] / 100) * (1 + X['Lag2'] / 100) * (1 + X['Lag3'] / 100) * (1 + X['Lag4'] / 100) - 1
        X['total perc'] *= 100


    y = tsret["Direction"]

    #Train Test split
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


# Lags: Window size (how far back we are looking from the current day)
lags = 4
# lags = 5

# train, validation, test data split ratios. Since we are only using logistic regression, we do not need a validation data.
#We actually use validation data as a test data, and set test ratio to 0
train_ratio = 0.5
val_ratio = 0.5
test_ratio = 0.0


# append the sign signal to percentage signal as an input to logistic regression, if set true
append_sign = False
#use only the sign signal as input (ignore the percentage signal) if set true
only_sign = False
#append the total percentage change from the first day to (lags)th day as a feature if set true
append_sum = False


#symbol to test the model
symbol = "AAPL"
# symbol = "AMZN"
# symbol = "TSLA"



# Multi source model: Use the data from all 3 stocks (aapl, amzn and tsla) to train the logistic regression
symbol_list_keen_perc =["AAPL", "AMZN",  "TSLA"]

# get the training /val/test data from each source in a list
X_train_multi_source_list = list()
X_val_multi_source_list = list()
X_test_multi_source_list = list()

y_train_multi_source_list = list()
y_val_multi_source_list = list()
y_test_multi_source_list = list()

for symbol_keen_perc in symbol_list_keen_perc:
    X_train, y_train, X_val, y_val, X_test, y_test = create_data(lags, train_ratio, val_ratio, test_ratio, symbol, append_sum)

    X_train_multi_source_list.append(X_train)
    X_val_multi_source_list.append(X_val)
    X_test_multi_source_list.append(X_test)

    y_train_multi_source_list.append(y_train)
    y_val_multi_source_list.append(y_val)
    y_test_multi_source_list.append(y_test)


# single source logistic regression model (only uses 1 stocks training data)
model_log = LogisticRegression(fit_intercept = True)
# multi source logistic regression (uses all 3 stocks' training data)
model_multi_source_log = LogisticRegression(fit_intercept = True)


# train split ratio for single source logistic regression
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
    X_test_consec_neg_multi_source = np.zeros([0, X_train.shape[1] + lags])
else:
    X_train_consec_neg_multi_source = np.zeros([0,X_train.shape[1] ])
    X_val_consec_neg_multi_source= np.zeros([0,X_train.shape[1] ])
    X_test_consec_neg_multi_source = np.zeros([0,X_train.shape[1] ])


y_train_consec_neg_multi_source = np.zeros([0])
y_val_consec_neg_multi_source = np.zeros([0])
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


# fit the single source logistic regression and predict on validation set
model_log.fit(X_train_consec_neg, y_train_consec_neg)
y_val_consec_neg_pred =model_log.predict(X_val_consec_neg)

# the predictions of a model that always predicts 1 (Rule-based model)
y_val_pred_always_neg = 1 * np.ones([len(y_val_consec_neg.to_numpy())])


# fit the multi source logistic regression and predict on validation set
model_multi_source_log.fit(X_train_consec_neg_multi_source, y_train_consec_neg_multi_source)
y_val_consec_neg_multi_source_pred =model_multi_source_log.predict(X_val_consec_neg)

# the predictions of a model that always predicts 1 (Rule-based model)
y_val_consec_neg_multi_source_pred_always_neg = 1 * np.ones([len(y_val_consec_neg)])

# In addition, get the probabilities from the multi-source model predictions (Confident logistic Regression model)
y_val_consec_neg_multi_source_pred_probs =model_multi_source_log.predict_proba(X_val_consec_neg)
threshold = 0.6
multi_source_where_confident =(y_val_consec_neg_multi_source_pred_probs[:, 1] > threshold)



# print("accuracy score Logistic Regression", accuracy_score(y_val_consec_neg_pred, y_val_consec_neg))
print("accuracy score Multi-Source-Logistic Regression", accuracy_score(y_val_consec_neg_multi_source_pred, y_val_consec_neg))
print(f"accuracy score for Multi-Source-Confident Logistic Regression with {threshold} threshold:",
                    accuracy_score(y_val_consec_neg_multi_source_pred[multi_source_where_confident],
                                   y_val_consec_neg[multi_source_where_confident]))
print("accuracy score of Rule Based", accuracy_score(y_val_pred_always_neg, y_val_consec_neg))


print(f"num total consecutive negative instances: {len(y_val_consec_neg_multi_source_pred)}, "
            f"num consecutive negative instances where model is confident: {np.sum(multi_source_where_confident)}")

#confusion matrix:
matrix = confusion_matrix(y_val_consec_neg, y_val_consec_neg_pred)
print(matrix)






