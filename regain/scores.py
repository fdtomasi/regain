# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from sklearn.utils.extmath import fast_logdet


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def BIC(emp_cov, precision):
    """Bayesian Information Criterion for Gaussian models"""
    return log_likelihood(emp_cov, precision) - (np.sum(precision != 0) - precision.shape[0])


def EBIC(emp_cov, precision, n=100, epsilon=0.5):
    """E - Bayesian Information Criterion for Gaussian models.

    It penalizes more then BIC by multplying the degrees of freedom also based
    on sample size.
    """

    likelihood = log_likelihood(emp_cov, precision)
    of_nonzero = np.sum(precision != 0) - precision.shape[0]
    penalty = np.log(n) / n * of_nonzero + 4 * epsilon * np.log(precision.shape[0]) / n * of_nonzero
    return likelihood - penalty


def EBIC_m(emp_cov, precision, n=100, epsilon=0.5):
    """E - Bayesian Information Criterion for Gaussian models.

    It penalizes more then BIC and E-BIC by multplying the degrees of freedom
    based on sample size and the number of variables.
    """
    likelihood = log_likelihood(emp_cov, precision)
    of_nonzero = np.sum(precision != 0) - precision.shape[0]
    p = precision.shape[0]
    penalty = np.log(n) / n * of_nonzero + 4 * epsilon * np.log(p * (p - 1) / 2) / n * of_nonzero
    return likelihood - penalty


def log_likelihood_t(emp_cov, precision):
    """Gaussian log-likelihood without constant term in time"""
    score = 0
    for e, p in zip(emp_cov, precision):
        score += fast_logdet(p) - np.sum(e * p)
    return score


def BIC_t(emp_cov, precision):
    """Bayesian Information Criterion for Gaussian models in time."""

    precision = np.array(precision)
    return log_likelihood_t(emp_cov, precision) - (np.sum(precision != 0) - precision.shape[1] * precision.shape[0])


def EBIC_t(emp_cov, precision, n=100, epsilon=0.5):
    """E - Bayesian Information Criterion for Gaussian models in time.

    It penalizes more then BIC by multplying the degrees of freedom also based
    on sample size.
    """
    likelihood = log_likelihood_t(emp_cov, precision)
    n_variables = precision.shape[1] * precision.shape[0]
    of_nonzero = np.sum(precision != 0) - n_variables
    penalty = np.log(n) / n * of_nonzero + 4 * epsilon * np.log(n_variables) / n * of_nonzero
    return likelihood - penalty


def EBIC_m_t(emp_cov, precision, n=100, epsilon=0.5):
    """E - Bayesian Information Criterion for Gaussian models in time.

    It penalizes more then BIC and E-BIC by multplying the degrees of freedom
    based on sample size and the number of variables.
    """
    likelihood = log_likelihood_t(emp_cov, precision)
    n_variables = precision.shape[1] * precision.shape[0]
    of_nonzero = np.sum(precision != 0) - n_variables
    penalty = np.log(n) / n * of_nonzero + 4 * epsilon * np.log(n_variables * (n_variables - 1) / 2) / n * of_nonzero
    return likelihood - penalty
