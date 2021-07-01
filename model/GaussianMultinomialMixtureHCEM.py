#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2019

@author: Celine Beji
Controler of Hybrid Causal Expectation-Maximization (HCEM) algorithm for a Gaussian & independant Multinomial distribution
"""


import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multinomial as mn
import time


class GaussianMultinomialMixtureHCEM:

    def __init__(self, maxIter=500, k=4, epsilonConv=0.000001):
        self.maxIter = maxIter  # maximum number of iterations of the algorithm
        self.k = k  # number of components (=4 with binary treatment and binary outcome)
        self.epsilonConv = epsilonConv  # convergence condition
        self.mus = []  # mean of Gaussian distributions
        self.sigma = []  # covariance of Gaussian distributions
        self.probaMult = []  # probabilities of Multinomial distributions
        self.pis = np.zeros((self.k, 1))  # mixing coefficients
        self.logLikelihoods = []  # log-likelihood
        self.numIters = 0  # number of iteration of the algorithm
        self.time = 0  # execution time
        self.proportionTreatment = 0 # probability of Y=1 in treatment group used to uplift (ITE) calculation
        self.proportionControl = 0 # probability of Y=1 in control group used to uplift (ITE) calculation
        from collections import namedtuple

    def fit(self, data, predictors):
        """
        fit the model on given data and return assign parameters to the objet: 
            mus, sigma, logLikelihoods, numIters, time, proportionTreatment, proportionControl
        input:
            data : Train data
            predictors = [predictorsContinue, predictorsCategorial]
            predictorsContinue : name of continious variables used to predict (for Gaussian distributions)
            predictorsCategorial : name of categorail variables used to predict (for Multinomial distributions)
        """
        
        
        predictorsContinue = predictors[0]
        predictorsCategorial = predictors[1]

        timeInit = time.time()  # initial time
        if predictorsContinue != []:
            n, p = data[predictorsContinue].shape # n:number of individus, p: number of continue variables
        if predictorsCategorial != []:
            n, l = data[predictorsCategorial].shape # n:number of individus, l: number of categorial variables
        number_categorie = []  # number of modalities (initialization)
        [number_categorie.append(data[c].value_counts().size)
         for c in predictorsCategorial]  # number of modalities
        t = np.ones((n, self.k))  # latent variables
        tDistrib = 0.5 * np.ones((n, self.k)) # initialization of latent variables
        data = data.reset_index()  # Initalization of dataset index
        Q = np.zeros((n, self.k))  # used to calculate the log-likelihood

        # condition on the maximum number of iterations
        while len(self.logLikelihoods) < self.maxIter:

            # -----------------
            # Expectation step
            # -----------------

            if len(self.logLikelihoods) != 0:  # iterations (not initialization)
                if predictorsContinue != []:  # Gaussian distribution
                    for j in range(self.k):
                        tDistrib[:, j] = self.pis[j] * \
                            mvn.pdf(data[predictorsContinue],
                                    self.mus[j], self.sigma[j])

                else:  # Only mixing coefficient
                    for j in range(self.k):
                        tDistrib[:, j] = self.pis[j] * np.ones(n)

            # causality constraints
            for i in range(n):

                if data['treatment'].iloc[i] == 0:  # Control group t = 0
                    if data['outcome'].iloc[i] == 0:  # y = 0
                        # Responder + Doomed
                        if predictorsCategorial != []:  # Multinomial distributions
                            t[i, 0] = tDistrib[i, 0] * np.prod([self.probaMult[:, var, 0][self.probaMult[:, var, 4]
                                                                                          == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                            t[i, 1] = tDistrib[i, 1] * np.prod([self.probaMult[:, var, 1][self.probaMult[:, var, 4]
                                                                                          == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                            t[i, 2] = 0
                            t[i, 3] = 0
                        else:
                            t[i, 0] = tDistrib[i, 0]
                            t[i, 1] = tDistrib[i, 1]
                            t[i, 2] = 0
                            t[i, 3] = 0

                    else:  # y = 1
                        # Survivor + Antiresponder
                        if predictorsCategorial != []:  # Multinomial distributions
                            t[i, 0] = 0
                            t[i, 1] = 0
                            t[i, 2] = tDistrib[i, 2]*np.prod([self.probaMult[:, var, 2][self.probaMult[:, var, 4]
                                                                                        == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                            t[i, 3] = tDistrib[i, 3]*np.prod([self.probaMult[:, var, 3][self.probaMult[:, var, 4]
                                                                                        == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                        else:
                            t[i, 0] = 0
                            t[i, 1] = 0
                            t[i, 2] = tDistrib[i, 2]
                            t[i, 3] = tDistrib[i, 3]

                else:  # Treatment group t = 1
                    if data['outcome'].iloc[i] == 0:  # y = 0
                        # Doomed + Antiresponder
                        if predictorsCategorial != []:  # Multinomial distributions
                            t[i, 0] = 0
                            t[i, 1] = tDistrib[i, 1]*np.prod([self.probaMult[:, var, 1][self.probaMult[:, var, 4]
                                                                                        == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                            t[i, 2] = 0
                            t[i, 3] = tDistrib[i, 3]*np.prod([self.probaMult[:, var, 3][self.probaMult[:, var, 4]
                                                                                        == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                        else:
                            t[i, 0] = 0
                            t[i, 1] = tDistrib[i, 1]
                            t[i, 2] = 0
                            t[i, 3] = tDistrib[i, 3]

                    else:  # y = 1
                        # Responder + Survivor
                        if predictorsCategorial != []:  # Multinomial distributions
                            t[i, 0] = tDistrib[i, 0]*np.prod([self.probaMult[:, var, 0][self.probaMult[:, var, 4]
                                                                                        == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                            t[i, 1] = 0
                            t[i, 2] = tDistrib[i, 2]*np.prod([self.probaMult[:, var, 2][self.probaMult[:, var, 4]
                                                                                        == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                            t[i, 3] = 0
                        else:
                            t[i, 0] = tDistrib[i, 0]
                            t[i, 1] = 0
                            t[i, 2] = tDistrib[i, 2]
                            t[i, 3] = 0

                t = (t.T / np.sum(t, axis=1)).T  # normalization

            # -----------------
            # Maximization step
            # -----------------

            # Update mixing coefficients
            for j in range(self.k):
                self.pis[j] = sum(t[:, j])/n

            # Update Gaussian parameters
            if predictorsContinue != []:
                # Means
                self.mus = np.dot(t.transpose(), data[predictorsContinue])
                for j in range(self.k):
                    self.mus[j, :] = self.mus[j, :]/sum(t[:, j])
                # Covariances
                self.sigma = [np.eye(p)] * self.k
                for j in range(self.k):
                    xCentered = np.matrix(
                        data[predictorsContinue] - self.mus[j])
                    self.sigma[j] = np.array(
                        1 / sum(t[:, j]) * np.dot(np.multiply(xCentered.T, t[:, j]), xCentered))
                    # Ad noise to have the inverse matrix
                    self.sigma[j] = self.sigma[j] + (10**(-2)*np.eye(p))

            # Update Multinomial probabilities
            if predictorsCategorial != []:
                self.probaMult = np.zeros((max(number_categorie), l, self.k+1))
                for var in range(l):  # number of categorial data
                    for mod in range(number_categorie[var]):
                        self.probaMult[:number_categorie[var], var,
                                       self.k] = data[predictorsCategorial[var]].value_counts().index.values
                        self.probaMult[:number_categorie[var],
                                       var, self.k].sort()

                        for j in range(self.k):
                            self.probaMult[:number_categorie[var], var, j] = (pd.crosstab(
                                data[predictorsCategorial[var]], np.ones(n), values=t[:, j], aggfunc='sum')/sum(t[:, j]))[1]

            # -----------------
            # Check for convergence
            # -----------------

            # Calculate log-likelihood
            if predictorsContinue != []:
                if predictorsCategorial != []:
                    pdfBinom = np.zeros(n)
                    for j in range(self.k):
                        for i in range(n):
                            pdfBinom[i] = np.prod([self.probaMult[:, var, j][self.probaMult[:, var, 4]
                                                                             == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                        Q[:, j] = np.multiply(t[:, j], np.multiply(np.log(
                            self.pis[j]*mvn.pdf(data[predictorsContinue], self.mus[j], self.sigma[j])), pdfBinom))
                else:
                    for j in range(self.k):
                        Q[:, j] = np.multiply(t[:, j], np.log(
                            self.pis[j]*mvn.pdf(data[predictorsContinue], self.mus[j], self.sigma[j])))
            else:
                if predictorsCategorial != []:
                    pdfBinom = np.zeros(n)
                    for j in range(self.k):
                        for i in range(n):
                            pdfBinom[i] = np.prod([self.probaMult[:, var, j][self.probaMult[:, var, 4]
                                                                             == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                        Q[:, j] = np.multiply(t[:, j], pdfBinom)

            self.logLikelihoods.append(np.sum(np.sum(Q, axis=1)))

            # -----------------
            # Check for convergence
            # -----------------

            if len(self.logLikelihoods) < 2:
                continue
            if self.logLikelihoods[-2] != 0 and np.abs(self.logLikelihoods[-1] - self.logLikelihoods[-2]) < self.epsilonConv:
                break

        # -----------------
        # Calcutaion of other parameters
        # -----------------

        # Proba of Y=1 in treatment group (used to ITE)
        self.proportionTreatment = np.logical_and(
            data['outcome'] == 1, data['treatment'] == 1).sum()
        # Proba de Y=1 in control group (used to ITE)
        self.proportionControl = np.logical_and(
            data['outcome'] == 1, data['treatment'] == 0).sum()
        # Train time
        self.time = time.time() - timeInit
        # Number of iterations
        self.numIters = len(self.logLikelihoods)

    def predict(self, data, predictorsContinue=[], predictorsCategorial=[]):
        """
        Predict with model parameters:
            Individual Treatment Effect (ITE),
            type Predict: responder, doomed, survivor, anti-responder
            z1, .., zk: probability distribution of each group
            outcome predict: outcome corresponding to the type Predict and Treatment (=0 or =1)
        """

        if predictorsContinue != []:
            n, p = data[predictorsContinue].shape # n:number of individus, p: number of continue data
        if predictorsCategorial != []:
            n, l = data[predictorsCategorial].shape # n:number of individus, l: number of discrete data
        t = np.zeros((n, self.k))

        # distribution on latent variable t
        if predictorsContinue != []:
            if predictorsCategorial != []:

                pdfBinom = np.zeros(n)
                for j in range(self.k):
                    for i in range(n):
                        pdfBinom[i] = np.prod([self.probaMult[:, var, j][self.probaMult[:, var, 4]
                                                                         == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                    t[:, j] = self.pis[j] * np.multiply(
                        mvn.pdf(data[predictorsContinue], self.mus[j], self.sigma[j]), pdfBinom)

            else:
                for j in range(self.k):
                    t[:, j] = self.pis[j] * \
                        mvn.pdf(data[predictorsContinue],
                                self.mus[j], self.Sigma[j])

        else:
            if predictorsCategorial != []:
                pdfBinom = np.zeros(n)
                for j in range(self.k):
                    for i in range(n):
                        pdfBinom[i] = np.prod([self.probaMult[:, var, j][self.probaMult[:, var, 4]
                                                                         == data[predictorsCategorial[var]].iloc[i]][0] for var in range(l)])
                    t[:, j] = self.pis[j] * pdfBinom

        t = (t.T / np.sum(t, axis=1)).T

        # prediction of ITE
        data['ITE'] = self.proportionTreatment * \
            (t[:, 0]+t[:, 2])-self.proportionControl*(t[:, 2]+t[:, 3])

        # prediction of type
        data['groupe'] = np.array(
            [np.argmax([t[i, 0], t[i, 1], t[i, 2], t[i, 3]]) for i in range(n)])
        data['typePredict'] = ''
        data.loc[data['groupe'] == 0, 'typePredict'] = 'responder'
        data.loc[data['groupe'] == 1, 'typePredict'] = 'doomed'
        data.loc[data['groupe'] == 2, 'typePredict'] = 'survivor'
        data.loc[data['groupe'] == 3, 'typePredict'] = 'anti-responder'

        # prediction of probability distribution
        for j in range(self.k):
            data['z'+str(j)] = t[:, j]

        # prediction of outcome
        if 'treatment' in data.columns :
            data['outcomePredict'] = 0
            data.loc[np.logical_and(data['typePredict'] == 'responder', data[''] == 1), 'outcomePredict'] = 1
            data.loc[data['typePredict'] == 'survivor', 'outcomePredict'] = 1
            data.loc[np.logical_and(data['typePredict'] == 'anti-responder', data['treatment'] == 0), 'outcomePredict'] = 1

    def predict_map(self, data, predictors):
        self.predict(data, predictors)

    def GridSearchCV (self, data, predictors, params):
        self.fit(data, predictors)
