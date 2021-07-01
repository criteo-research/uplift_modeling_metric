#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anonymous
Controler of variable Z transformation model
"""


import time
from sklearn.model_selection import GridSearchCV, KFold


class ZClassifier:
    def __init__(self, classifier):

        self.model = classifier  # model
        self.numIters = None  # number of iteration of the algorithm
        self.time = 0  # execution time

    def fit(self, data, predictors):

        timeInit = time.time()  # initial time

        # Variable transformation
        data['Zoutcome'] = 0
        data['Zoutcome'] = data['treatment']*data['outcome'] + \
            (1-data['treatment'])*(1-data['outcome'])
        # Fit the model
        self.model.fit(data[predictors], data['Zoutcome'])

        # Train time
        self.time = time.time() - timeInit

    def predict(self, data, predictors):

        # pred uplift
        data['ITE'] = 2*self.model.predict_proba(data[predictors])[:, 1] - 1

    def predict_map(self, data, predictors):
        self.predict(data, predictors)

    def GridSearchCV(self, data, predictors, params):

        timeInit = time.time()  # initial time

        # Variable transformation
        data['Zoutcome'] = 0
        data['Zoutcome'] = data['treatment']*data['outcome'] + \
            (1-data['treatment'])*(1-data['outcome'])

        # Fit the model
        g1 = GridSearchCV(self.model, params, cv=KFold(
            n_splits=5, shuffle=True), refit=True)
        g1.fit(data[predictors], data['Zoutcome'])

        # Best paramters
        self.model = g1.best_estimator_

        # Train time
        self.time = time.time() - timeInit
