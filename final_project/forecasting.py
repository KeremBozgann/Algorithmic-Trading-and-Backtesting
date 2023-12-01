import numpy as np
import os
import pandas as pd
import sys
import xgboost as xgb

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


def create_data(lags, train_ratio, val_ratio, test_ratio, symbol):
    sys.path.append('../')

    path= os.path.join("..", 'data', symbol+ ".csv")
    ts = pd.read_csv(path)

    # close = (data['close'].to_numpy()).reshape(-1, 1)



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

    X = tsret[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5"]]
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
lags = 5

# train, validation, test data split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# symbol: stock symbol
symbol = "SPY"
# symbol = "GOOGL"
X_train, y_train, X_val, y_val, X_test, y_test = create_data(lags, train_ratio, val_ratio, test_ratio, symbol)



# your model goes here.
#model = QDA()
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 50}
model = xgb.XGBRFClassifier(**params)
y_train[y_train==-1] = 0
model.fit(X_train, y_train)

# model = LDA()
# model = SVC()
# model = LogisticRegression()

#model.fit(X_train, y_train)
y_val_pred =model.predict(X_val)
y_val_pred[y_val_pred==0] = -1
print("accuracy score", accuracy_score(y_val_pred, y_val))


#confusion matrix:
matrix = confusion_matrix(y_val, y_val_pred)
print(matrix.diagonal()/matrix.sum(axis=1))




