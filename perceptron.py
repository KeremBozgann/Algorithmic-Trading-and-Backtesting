from sklearn.linear_model import Perceptron
import numpy as np




clf_with_intercept = Perceptron(fit_intercept=True)
clf_with_intercept.fit(X, y)
