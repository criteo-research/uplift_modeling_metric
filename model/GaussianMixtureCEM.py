#!/usr/bin/env python3

# - * -coding: utf - 8 - * -
"""
@author: Anonymous
Controler of Causal Expectation-Maximization (CEM) algorithm for a Gaussian Mixture distribution

From: https://github.com/CelineBeji/ECM
"""

import numpy as np
from scipy.stats import multivariate_normal as mvn
import time


class GaussianMixtureCEM:

    def __init__(self, max_iter=500, k=4, epsilonConv=0.000001):
        assert k == 4, "other values are not supported for the moment"
        self.max_iter = max_iter  # maximum number of iterations of the algorithm
        # number of components (=4 with binary treatment and binary outcome)
        self.k = k
        self.epsilonConv = epsilonConv  # convergence condition
        self.mus = []  # mean of Gaussian Mixture distribution
        self.sigma = []  # covariance of Gaussian Mixture distribution
        self.pis = np.zeros((self.k, 1))  # mixing coefficients
        self.logLikelihoods = []  # log-likelihood
        self.numIters = 0  # number of iteration of the algorithm
        self.time = 0  # execution time
        # probability of Y=1 in treatment group used to uplift (ITE) calculation
        self.proportionTreatment = 0
        # probability of Y=1 in control group used to uplift (ITE) calculation
        self.proportionControl = 0

    def fit(self, data, predictors):
        """
        fit the model on given data and return determine parameters:
           pi, mus, sigma, logLikelihoods, numIters, time, proportionTreatment, proportionControl
         input:
            data : Train data
            predictors : name of variables used to predict
        """

        timeInit = time.time()  # initial time
        n, p = data[predictors].shape  # size of data
        # t = np.ones((n, self.k))  # latent variables
        # Q = np.zeros((n, self.k))  # used to calculate the log-likelihood
        # initialization of latent variables
        tDistrib = 0.5 * np.ones((n, self.k))

        # condition on the maximum number of iterations
        while len(self.logLikelihoods) < self.max_iter:

            # -----------------
            # Expectation step
            # -----------------

            # if len(self.logLikelihoods) != 0:  # iterations (not initialization)
            #     for j in range(self.k):
            #         tDistrib[:, j] = self.pis[j] * \
            #             mvn.pdf(data[predictors], self.mus[j], self.sigma[j])

            # causality constraints
            T = (data['T'] == 1).values
            Y = (data['Y'] == 1).values
            mask = np.array([(T == Y), (1 - Y), Y, (T != Y) ]).T
            t = tDistrib * mask

            t = (t.T / np.sum(t, axis=1)).T  # normalization

            # for i in range(n):
            #     # print(i)
            #     if data['T'].iloc[i] == 0:  # Control group t = 0
            #         if data['Y'].iloc[i] == 0:  # y = 0
            #             # Responder + Doomed
            #             t[i, 0] = tDistrib[i, 0]
            #             t[i, 1] = tDistrib[i, 1]
            #             t[i, 2] = 0
            #             t[i, 3] = 0
            #
            #         else:  # y = 1
            #             # Survivor + Antiresponder
            #             t[i, 0] = 0
            #             t[i, 1] = 0
            #             t[i, 2] = tDistrib[i, 2]
            #             t[i, 3] = tDistrib[i, 3]
            #
            #     else:  # Treatment group t = 1
            #         if data['Y'].iloc[i] == 0:  # y = 0
            #             # Doomed + Antiresponder
            #             t[i, 0] = 0
            #             t[i, 1] = tDistrib[i, 1]
            #             t[i, 2] = 0
            #             t[i, 3] = tDistrib[i, 3]
            #
            #         else:  # y = 1
            #             # Responder + Survivor
            #             t[i, 0] = tDistrib[i, 0]
            #             t[i, 1] = 0
            #             t[i, 2] = tDistrib[i, 2]
            #             t[i, 3] = 0
            #
            #     t = (t.T / np.sum(t, axis=1)).T  # normalization

            # -----------------
            # Maximization step
            # -----------------

            t_sums = t.sum(axis=0)

            # Update mixing coefficients
            self.pis = t_sums.reshape((self.k, 1)) / n  # TODO: remove reshape
            # for j in range(self.k):
            #    self.pis[j] = sum(t[:, j])/n

            # Update means of Gaussians
            self.mus = np.dot(t.transpose(), data[predictors]) / t_sums[:, None]
            # self.mus = np.dot(t.transpose(), data[predictors])
            # for j in range(self.k):
            #     self.mus[j, :] = self.mus[j, :]/sum(t[:, j])

            # Update covariance of Gaussians
            d_predictors = data[predictors]
            rest = 10 ** (-2) * np.eye(p)
            self.sigma = [
                np.cov(d_predictors, aweights=t[:, j], rowvar=False, bias=True) + rest
                for j in range(self.k)
            ]

            # self.sigma = [np.eye(p)] * self.k
            # for j in range(self.k):
            #     xCentered = np.matrix(data[predictors] - self.mus[j])
            #     self.sigma[j] = np.array(
            #         1 / sum(t[:, j]) * np.dot(np.multiply(xCentered.T, t[:, j]), xCentered))
            #     self.sigma[j] = self.sigma[j] + (10**(-2)*np.eye(p))

            # -----------------
            # Check for convergence
            # -----------------

            for j in range(self.k):
                tDistrib[:, j] = self.pis[j] * mvn.pdf(data[predictors], self.mus[j], self.sigma[j])

            # Calculate log-likelihood
            logLikelihood = (t * np.log(tDistrib)).sum()
            self.logLikelihoods.append(logLikelihood)

            # for j in range(self.k):
                # print("pis"+self.pis[j])
                # print("mus"+self.mus[j])
                # print("sigma"+self.sigma[j])
                # Q[:, j] = np.multiply(t[:, j], np.log(
                #     self.pis[j] * mvn.pdf(data[predictors], self.mus[j], self.sigma[j])))
            # self.logLikelihoods.append(np.sum(np.sum(Q, axis=1)))

            # force at least one iteration
            if len(self.logLikelihoods) < 2:
                continue
            # continues the algorithm while the log-likelood varies by more than epsilonConv
            if self.logLikelihoods[-2] != 0 and np.abs(self.logLikelihoods[-1] - self.logLikelihoods[-2]) < self.epsilonConv:
                break

        # -----------------
        # Calcutaion of other parameters
        # -----------------

        # Proba of Y=1 in treatment group (used to ITE)
        self.proportionTreatment = np.logical_and(
            data['Y'] == 1, data['T'] == 1).sum() / (data['T'] == 1).sum()
        # Proba de Y=1 in control group (used to ITE)
        self.proportionControl = np.logical_and(
            data['Y'] == 1, data['T'] == 0).sum() / (data['T'] == 0).sum()
        # Train time
        self.time = time.time() - timeInit
        # Number of iterations
        self.numIters = len(self.logLikelihoods)

    def predict(self, data, predictors):
        """
        Predict with model parameters:
            Individual Treatment Effect (ITE),
            type Predict: responder, doomed, survivor, anti-responder
            z1, .., zk: probability distribution of each group
            outcome predict: outcome corresponding to the type Predict and Treatment (=0 or =1)
        """

        n, p = data[predictors].shape
        t = np.zeros((n, self.k))

        # distribution of latent variables
        for j in range(self.k):
            t[:, j] = self.pis[j] * \
                mvn.pdf(data[predictors], self.mus[j], self.sigma[j])
        t = (t.T / np.sum(t, axis=1)).T

        # prediction of ITE
        #data['ITE'] = t[:, 0] - t[:, 3]

        data['ITE'] = 2*(self.proportionTreatment * (t[:, 0]+t[:, 2]) - self.proportionControl *
                         (t[:, 2]+t[:, 3]))

        # data['ITE'] = (self.proportionTreatment * (t[:, 0]+t[:, 2]) - self.proportionControl *
        #               (t[:, 2]+t[:, 3])) / 2*(self.proportionTreatment - self.proportionControl)

        # prediction of type
        # data['groupe'] = np.array(
        #     [np.argmax([t[i, 0], t[i, 1], t[i, 2], t[i, 3]]) for i in range(n)])
        # data['typePredict'] = ''
        # data.loc[data['groupe'] == 0, 'typePredict'] = 'responder'
        # data.loc[data['groupe'] == 1, 'typePredict'] = 'doomed'
        # data.loc[data['groupe'] == 2, 'typePredict'] = 'survivor'
        # data.loc[data['groupe'] == 3, 'typePredict'] = 'anti-responder'
        #
        # # prediction of probability distribution
        # for j in range(self.k):
        #     data['z'+str(j)] = t[:, j]
        #
        # # prediction of outcome treatment
        # if 'T' in data.columns:
        #     data['outcomePredict'] = 0
        #     data.loc[np.logical_and(
        #         data['typePredict'] == 'responder', data['T'] == 1), 'outcomePredict'] = 1
        #     data.loc[data['typePredict'] == 'survivor', 'outcomePredict'] = 1
        #     data.loc[np.logical_and(
        #         data['typePredict'] == 'anti-responder', data['T'] == 0), 'outcomePredict'] = 1
        #
        #     data['outcome1predict'] = 0
        #     data['outcome0predict'] = 0
        #     data.loc[data['typePredict'] == 'responder', 'outcome1predict'] = 1
        #     data.loc[data['typePredict'] == 'survivor', 'outcome1predict'] = 1
        #     data.loc[data['typePredict'] == 'survivor', 'outcome0predict'] = 1
        #     data.loc[data['typePredict'] ==
        #              'anti-responder', 'outcome0predict'] = 1

    def predict_map(self, data, predictors):
        self.predict(data, predictors)

    def GridSearchCV(self, data, predictors, params):
        self.fit(data, predictors)
