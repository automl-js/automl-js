"""Comparison of RBF and polynomial kernels for SVM"""
import numpy as np
from pmlb import classification_dataset_names, fetch_data
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

poly = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {
        'svc__kernel': ['poly'],
        'svc__degree': [2, 3],
        'svc__coef0': np.logspace(-3, 3, 13),
        'svc__C': np.logspace(-7, 5, 13)
    },
    cv=5,
    n_jobs=-1
)

rbf = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {
        'svc__kernel': ['rbf'],
        'svc__gamma': np.logspace(-3, 3, 13),
        'svc__C': np.logspace(-7, 5, 13)
    },
    cv=5,
    n_jobs=-1
)

dum = GridSearchCV(
    make_pipeline(StandardScaler(), DummyClassifier()),
    {
        'dummyclassifier__strategy': ['stratified', 'most_frequent', 'uniform']
    },
    cv=5,
    n_jobs=-1
)

n_max = 256

for dataset in classification_dataset_names:
    X, y = fetch_data(dataset, True)
    
    # maximum n_max samples
    if len(y) > n_max:
        S = np.random.permutation(len(y))[:n_max]
        I = np.zeros(len(y))
        I[S] = 1
        I = I > 0

        X = X[I]
        y = y[I]
    
    pscores = cross_val_score(poly, X, y, cv=5, n_jobs=-1)
    rscores = cross_val_score(rbf, X, y, cv=5, n_jobs=-1)
    dscores = cross_val_score(dum, X, y, cv=5, n_jobs=-1)

    names = ['RBF', "Poly", "Dummy"]
    values = [np.round(x,2) for x in [np.mean(rscores), np.mean(pscores), np.mean(dscores)]]

    result = "[" + ", ".join(["'%s': %s" % (a, b) for a, b in zip(names, values)]) + "],"

    print(result)