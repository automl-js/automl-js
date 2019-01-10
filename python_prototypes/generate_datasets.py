"""Generates a set of useful datasets for testing"""

import numpy as np
from pmlb import classification_dataset_names, fetch_data
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import json

def make_svr(n_samples=320, n_features=1, gamma=1.0, kernel='rbf'):
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    y = np.random.randn(n_samples)

    model = SVR(C=100000.0, gamma=gamma, kernel=kernel)

    model.fit(X, y)
    y = model.predict(X)

    json.dump({
        'X': X.tolist(),
        'y': y.tolist(),
        'type': 'regression'
    }, open('svr_%s_%sg.json' % (kernel, gamma), 'w'))

    plt.scatter(X[:, 0], y)
    plt.show()

if __name__ == "__main__":
    make_svr()