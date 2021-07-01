#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anonymous
Controler of two classifiers model
"""


import numpy as np
import time
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV


class TwoClassifiers:

    def __init__(self, model_t=None, model_c=None):

        self.model_t = model_t  # model on treatment group
        self.model_c = model_c  # model on control group
        self.numIters = None  # number of iteration of the algorithm
        self.time = 0  # execution time

    def fit(self, data, predictors):

        timeInit = time.time()  # initial time
        data_t = data[data['T'] == 1]  # treatment group
        data_c = data[data['T'] == 0]  # control group

        # Fit on treatment group
        self.model_t.fit(data_t[predictors], data_t['Y'])
        # Fit on control group
        self.model_c.fit(data_c[predictors], data_c['Y'])

        # Train time
        self.time = time.time() - timeInit

    def predict(self, data, predictors):

        # prediction of ITE
        data['ITE'] = self.model_t.predict_proba(
            data[predictors])[:, 1]-self.model_c.predict_proba(data[predictors])[:, 1]

        # prediction of type
        # data.loc[np.logical_and(np.logical_and(data['T'] == 1, data['Y'] == 1), self.model_c.predict(
        #     data[predictors]) == 0), 'typePredict'] = 'responder'
        # data.loc[np.logical_and(np.logical_and(data['T'] == 0, data['Y'] == 0), self.model_t.predict(
        #     data[predictors]) == 1), 'typePredict'] = 'responder'
        #
        # data.loc[np.logical_and(np.logical_and(data['T'] == 1, data['Y'] == 0), self.model_c.predict(
        #     data[predictors]) == 0), 'typePredict'] = 'doomed'
        # data.loc[np.logical_and(np.logical_and(data['T'] == 0, data['Y'] == 0), self.model_t.predict(
        #     data[predictors]) == 0), 'typePredict'] = 'doomed'
        #
        # data.loc[np.logical_and(np.logical_and(data['T'] == 1, data['Y'] == 1), self.model_c.predict(
        #     data[predictors]) == 1), 'typePredict'] = 'survivor'
        # data.loc[np.logical_and(np.logical_and(data['T'] == 0, data['Y'] == 1), self.model_t.predict(
        #     data[predictors]) == 1), 'typePredict'] = 'survivor'
        #
        # data.loc[np.logical_and(np.logical_and(data['T'] == 1, data['Y'] == 0), self.model_c.predict(
        #     data[predictors]) == 1), 'typePredict'] = 'anti-responder'
        # data.loc[np.logical_and(np.logical_and(data['T'] == 0, data['Y'] == 1), self.model_t.predict(
        #     data[predictors]) == 0), 'typePredict'] = 'anti-responder'
        #
        # # prediction of outcome treatment
        # data['outcomePredict'] = 0
        # data.loc[np.logical_and(
        #     data['typePredict'] == 'responder', data['T'] == 1), 'outcomePredict'] = 1
        # data.loc[data['typePredict'] == 'survivor', 'outcomePredict'] = 1
        # data.loc[np.logical_and(
        #     data['typePredict'] == 'anti-responder', data['T'] == 0), 'outcomePredict'] = 1
        #
        # data['outcome1predict'] = self.model_t.predict(data[predictors])
        # data['outcome0predict'] = self.model_c.predict(data[predictors])

    def predict_map(self, data, predictors):
        # prediction of ITE
        data['ITE'] = self.model_t.predict_proba(
            data[predictors])[:, 1]-self.model_c.predict_proba(data[predictors])[:, 1]

    def GridSearchCV(self, data, predictors, params):

        timeInit = time.time()  # initial time

        # Fit on treatment group
        g1 = GridSearchCV(self.model_t, params,
                          cv=KFold(n_splits=20, shuffle=True))
        g1.fit(data[data['T'] == 1][predictors],
               data[data['T'] == 1]['Y'])
        # Fit on control group
        g2 = GridSearchCV(self.model_c, params,
                          cv=KFold(n_splits=20, shuffle=True))
        g2.fit(data[data['T'] == 0][predictors],
               data[data['T'] == 0]['Y'])

        # Best paramters
        self.model_t = g1.best_estimator_
        self.model_c = g2.best_estimator_

        # Train time
        self.time = time.time() - timeInit

    def RandomizedSearchCV(self, data, predictors, params):

        # Fit on treatment group
        g1 = RandomizedSearchCV(self.model_t, params,
                                cv=KFold(n_splits=5, shuffle=True))
        g1.fit(data[data['T'] == 1][predictors],
               data[data['T'] == 1]['Y'])
        # Fit on control group
        g2 = RandomizedSearchCV(self.model_c, params,
                                cv=KFold(n_splits=5, shuffle=True))
        g2.fit(data[data['T'] == 0][predictors],
               data[data['T'] == 0]['Y'])

        # Best paramters
        self.model_t = g1.best_estimator_
        self.model_c = g2.best_estimator_
