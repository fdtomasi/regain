#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Correlated Topic Model (CTM) in Python.

This module implements the CTM model as described in
http://www.cs.princeton.edu/~blei/papers/BleiLafferty2007.pdf

Like in LDA, the posterior distribution is impossible to compute.
We approximate it with a variational distribution. We then aim at minimizing
the Kullback Leibler divergence between the two distributions, which is
equivalent  to finding the variational distribution which maximizes a given
Likelihood bound.

"""

import logging
import copy

logger = logging.getLogger('gensim.models.ctmmodel')


import numpy as np # for arrays, array broadcasting etc.
# numpy.seterr(divide='ignore') # ignore 0*log(0) errors
from numpy.linalg import inv, det
from scipy.optimize import minimize, fmin_l_bfgs_b
from sklearn.utils.extmath import fast_logdet

from gensim import interfaces, utils
from six.moves import xrange


class SufficientStats():
    """
    Stores statistics about variational parameters during E-step in order
    to update CtmModel's parameters in M-step.

    `self.mu_stats` contains sum(lamda_d)

    `self.sigma_stats` contains sum(I_nu^2 + lamda_d * lamda^T)

    `self.beta_stats[i]` contains sum(phi[d, i] * n_d) where nd is the vector
    of word counts for document d.

    `self.numtopics` contains the number of documents the statistics are build on

    """

    def __init__(self, numtopics, numterms):
        self.numdocs = 0
        self.numtopics = numtopics
        self.numterms = numterms
        self.beta_stats = np.zeros([numtopics, numterms])
        self.mu_stats = np.zeros(numtopics)
        self.sigma_stats = np.zeros([numtopics, numtopics])

    def update(self, lamda, nu2, phi, doc):
        """
        Given optimized variational parameters, update statistics

        """

        # update mu_stats
        self.mu_stats += lamda

        # update \beta_stats[i], 0 < i < self.numtopics
        for n, c in doc:
            for i in xrange(self.numtopics):
                self.beta_stats[i, n] += c * phi[n, i]

        # update \sigma_stats
        self.sigma_stats += np.diag(nu2) + np.dot(lamda, lamda.transpose())

        self.numdocs += 1


from gensim.models import LdaModel
class CtmModel(LdaModel):
    """
    The constructor estimated Correlated Topic Model parameters based on a
    training corpus:

    >>> ctm = CtmModel(corpus, num_topics=10)

    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            estep_convergence=0.001, em_convergence=0.0001,
            em_max_iterations=50):
        """
        If given, start training from the iterable `corpus` straight away.
        If not given, the model is left untrained (presumably because you
        want to call `update()` manually).

        `num_topics` is the number of requested latent topics to be extracted
        from the training corpus.

        `id2word` is a mapping from word ids (integers) to words (strings).
        It is used to determine the vocabulary size, as well as for debugging
        and topic printing.

        The variational EM runs until the relative change in the likelihood
        bound is less than `em_convergence`.

        In each EM iteration, the E-step runs until the relative change in
        the likelihood bound is less than `estep_convergence`.

        """

        # store user-supplied parameters
        self.id2word = id2word
        self.estep_convergence = estep_convergence  # relative change we need to achieve in E-step
        self.em_convergence = em_convergence  # relative change we need to achieve in Expectation-Maximization
        self.em_max_iterations = em_max_iterations

        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute CTL over an empty collection (no terms)")

        self.num_topics = int(num_topics)

        # initialize a model with zero-mean, diagonal covariance gaussian and
        # random topics seeded from the corpus
        self.mu = np.zeros(self.num_topics)
        self.sigma = np.diagflat([1.0] * self.num_topics)
        self.sigma_inverse = inv(self.sigma)
        self.beta = np.random.uniform(0, 1, (self.num_topics, self.num_terms))

        # variational parameters
        self.lamda = np.zeros(self.num_topics)
        self.nu2 = np.ones(self.num_topics)  # nu^2
        self.phi = 1/float(self.num_topics) * np.ones([self.num_terms, self.num_topics])
        self.optimize_zeta()

        # in order to get the topics graph, we need to store the
        # optimized lamda for each document
        self.observed_lamda = np.zeros([len(corpus)])

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.expectation_maximization(corpus)

    def __str__(self):
        return "CtmModel(num_terms=%s, num_topics=%s)" % \
                (self.num_terms, self.num_topics)

    def expectation_maximization(self, corpus):
        """
        Expectation-Maximization algorithm.
        During E-step, variational parameters are optimized with fixed model parameters.
        During M-step, model parameters are optimized given statistics collected in E-step.

        """
        for iteration in xrange(self.em_max_iterations):
            old_bound = self.corpus_bound(corpus)

            # print (iteration)
            # print "bound before E-step %f" %(old_bound)
            # E-step and collect sufficient statistics for the M-step
            statistics = self.do_estep(corpus)

            # M-step
            self.do_mstep(statistics)

            new_bound = self.corpus_bound(corpus)

            # print "bound after M-step %f" %(new_bound)

            if (new_bound - old_bound) / old_bound < self.em_convergence:
                break

    def do_estep(self, corpus):

        # initialize empty statistics
        statistics = SufficientStats(self.num_topics, self.num_terms)

        for d, doc in enumerate(corpus):

            # variational_inference modifies the variational parameters
            model = copy.deepcopy(self)
            model.variational_inference(doc)

            # collect statistics for M-step
            statistics.update(model.lamda, model.nu2, model.phi, doc)

        return statistics

    def do_mstep(self, sstats):
        """
        Optimize model's parameters using the statictics collected
        during the e-step

        """

        for i in xrange(self.num_topics):
            beta_norm = np.sum(sstats.beta_stats[i])
            self.beta[i] = sstats.beta_stats[i] / beta_norm

        self.mu = sstats.mu_stats / sstats.numdocs

        self.sigma = sstats.sigma_stats + np.multiply(self.mu, self.mu.transpose())
        self.sigma_inverse = inv(self.sigma)

    def bound(self, doc, lamda=None, nu2=None):
        """
        Estimate the variational bound of a document

        """
        if lamda is None:
            lamda = self.lamda

        if nu2 is None:
            nu2 = self.nu2

        N = sum([cnt for _, cnt in doc])  # nb of words in document

        bound = 0.0

        # E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lamda, \nu) + sum_n,i { \phi_{n,i}*log(\phi_{n,i}) }
        bound = - np.sum(np.diag(nu2) * self.sigma_inverse) + fast_logdet(self.sigma_inverse)
        
        bound -= (lamda - self.mu).transpose().dot(self.sigma_inverse).dot(lamda - self.mu)
        bound += np.sum(np.log(nu2)) + self.num_topics  # TODO safe_log
        
        bound /= 2
        # print "first term %f for doc %s" %(bound, doc)

        # \sum_n { E[log p(z_n | \eta)] - sum_i {\lamda_i * \phi_{n, i}}
        sum_exp = np.exp(lamda + 0.5 * nu2).sum()
        bound -= (N * (sum_exp / self.zeta - 1. + np.log(self.zeta)))

        # print "second term %f for doc %s" %(bound, doc)

        # E[log p(w_n | z_n, \beta)] - sum_n,i { \phi_{n,i}*log(\phi_{n,i})
        bound += sum(c * (self.phi[n] * (lamda + np.log(self.beta[:, n]) - np.log(self.phi[n]))).sum()
            for (n, c) in doc)

        return bound

    def corpus_bound(self, corpus):
        """
        Estimates the likelihood bound for the whole corpus by summing over
        all the documents in the corpus.

        """
        return sum([self.bound(doc) for doc in corpus])

    def variational_inference(self, doc):
        """
        Optimize variational parameters (zeta, lamda, nu, phi) given the
        current model and a document
        This method modifies the model self.

        """

        bound = self.bound(doc)
        new_bound = bound
        # print "bound before variational inference %f" %(bound)

        for iteration in xrange(self.em_max_iterations):

            # print ("bound before zeta opt %f" %(self.bound(doc)))
            self.optimize_zeta()

            # print ("bound before lamda opt %f" %(self.bound(doc)))
            self.optimize_lamda(doc)

            self.optimize_zeta()

            # print ("bound before nu2 opt %f" %(self.bound(doc)))
            self.optimize_nu2(doc)

            self.optimize_zeta()

            # print ("bound before phi opt %f" %(self.bound(doc)))
            self.optimize_phi(doc)

            bound, new_bound = new_bound, self.bound(doc)

            relative_change = abs((bound - new_bound)/bound)

            if (relative_change < self.estep_convergence):
                break
            break

        # print ("bound after variational inference %f" %(bound))

        return bound

    def optimize_zeta(self):
        # self.zeta = sum([np.exp(self.lamda[i] + 0.5 * self.nu2[i])
        #     for i in xrange(self.num_topics)])
        self.zeta = np.exp(self.lamda + 0.5 * self.nu2).sum()

    def optimize_phi(self, doc):
        for n, _ in doc:
#             phi_norm = sum([np.exp(self.lamda[i]) * self.beta[i, n]
#             for i in xrange(self.num_topics):
#                 self.phi[n, i] = np.exp(self.lamda[i]) * self.beta[i, n] / phi_norm
            self.phi[n] = np.exp(self.lamda) * self.beta[:, n] / np.exp(self.lamda * self.beta[:, n]).sum()

    def optimize_lamda(self, doc):
        N = sum([c for _, c in doc])
        def f(lamda):
            return self.bound(doc, lamda=lamda)

        def df(lamda):
            """Returns dL/dlamda"""
            result = -np.dot(self.sigma_inverse, (lamda - self.mu))
            result += np.sum([c * self.phi[n, :] for n, c in doc])
            result -= N * np.exp(lamda + 0.5 * self.nu2) / self.zeta

            return result

        # We want to maximize f, but np only implements minimize, so we
        # minimize -f
        res = minimize(lambda x: -f(x), self.lamda, method='BFGS', jac=lambda x: -df(x))

        self.lamda = res.x

    def optimize_nu2(self, doc):
        N = np.sum([c for _, c in doc])
        
        def f(nu2):
            return self.bound(doc, nu2=nu2)

        def df(nu2):
            """
            Returns dL/dnu2

            """
#             result = np.zeros(self.num_topics)
#             for i in xrange(self.num_topics):
#                 result[i] = - 0.5 * self.sigma_inverse[i, i]
#                 result[i] -= N/(2*self.zeta) * np.exp(self.lamda[i] + 0.5 * nu2[i])
#                 result[i] += 1/(2*nu2[i])  # TODO safe_division
            
            res = -0.5 * np.diag(self.sigma_inverse)
            res -= 0.5 * N / self.zeta * np.exp(self.lamda + 0.5 * nu2)
            res += 0.5 / nu2
            return res

        # constraints : we need nu2[i] >= 0
        constraints = [(0, None)] * self.num_topics
        result = fmin_l_bfgs_b(lambda x: -f(x), self.nu2, fprime=lambda x: -df(x), bounds=constraints)
        self.nu2 = result[0]