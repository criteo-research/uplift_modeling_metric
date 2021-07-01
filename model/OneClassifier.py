#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anonymous
Controler of One classifiers model using treatment as a feature
"""


import time
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold


class OneClassifier:

    def __init__(self, model=None):

        self.model = model  # model on treatment group
        self.numIters = None  # number of iteration of the algorithm
        self.time = 0  # execution time

    def fit(self, data, predictors):

        timeInit = time.time()  # initial time

        # Take 'treatment' as a feature
        predictorTreatment = predictors + ['T']
        # Fit on treatment group
        self.model.fit(data[predictorTreatment], data['Y'])

        # Train time
        self.time = time.time() - timeInit

    def predict(self, data, predictors):

        # data['treatmentPredict'] = 1 - data['T']
        # predictorTreatment = predictors + ['treatmentPredict']
        # data['outcomePredict'] = self.model.predict(data[predictorTreatment])

        data['treatment1'] = 1
        data['treatment0'] = 0
        data['ITE'] = self.model.predict_proba(data[predictors+['treatment1']])[
            :, 1] - self.model.predict_proba(data[predictors+['treatment0']])[:, 1]

        # # prediction of type
        # data.loc[np.logical_and(np.logical_and(data['T'] == 1, data['Y']
        #                                        == 1), data['outcomePredict'] == 0), 'typePredict'] = 'responder'
        # data.loc[np.logical_and(np.logical_and(data['T'] == 0, data['Y']
        #                                        == 0), data['outcomePredict'] == 1), 'typePredict'] = 'responder'
        # data.loc[np.logical_and(
        #     data['Y'] == 0, data['outcomePredict'] == 0), 'typePredict'] = 'doomed'
        # data.loc[np.logical_and(
        #     data['Y'] == 1, data['outcomePredict'] == 1), 'typePredict'] = 'survivor'
        # data.loc[np.logical_and(np.logical_and(data['T'] == 1, data['Y'] == 0),
        #                         data['outcomePredict'] == 1), 'typePredict'] = 'anti-responder'
        # data.loc[np.logical_and(np.logical_and(data['T'] == 0, data['Y'] == 1),
        #                         data['outcomePredict'] == 0), 'typePredict'] = 'anti-responder'
        #
        # data['outcome1predict'] = self.model.predict(
        #     data[predictors+['treatment1']])
        # data['outcome0predict'] = self.model.predict(
        #     data[predictors+['treatment0']])
        #
        data = data.drop(['treatment1', 'treatment0'], axis=1, inplace=True)

    def predict_map(self, data, predictors):

        data['treatment1'] = 1
        data['treatment0'] = 0
        data['ITE'] = self.model.predict_proba(data[predictors+['treatment1']])[
            :, 1] - self.model.predict_proba(data[predictors+['treatment0']])[:, 1]
        data = data.drop(['treatment1', 'treatment0'], axis=1)

    def GridSearchCV(self, data, predictors, params):

        timeInit = time.time()  # initial time
        predictorTreatment = predictors + ['T']
        # Fit on treatment group
        g = GridSearchCV(self.model, params,
                         cv=KFold(n_splits=20, shuffle=True))
        g.fit(data[predictorTreatment], data['Y'])

        # Best paramters
        self.model = g.best_estimator_

        # Train time
        self.time = time.time() - timeInit
