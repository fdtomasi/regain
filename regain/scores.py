import numpy as np

from sklearn.utils.extmath import fast_logdet


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def BIC(emp_cov, precision):
    return log_likelihood(emp_cov, precision) - \
        (np.sum(precision != 0) - precision.shape[0])


def EBIC(emp_cov, precision, n=100, epsilon=0.5):
    likelihood = log_likelihood(emp_cov, precision)
    of_nonzero = np.sum(precision != 0) - precision.shape[0]
    penalty = np.log(n)/n*of_nonzero + \
        4 * epsilon * np.log(precision.shape[0])/n * of_nonzero
    return likelihood - penalty


def EBIC_m(emp_cov, precision, n=100, epsilon=0.5):
    likelihood = log_likelihood(emp_cov, precision)
    of_nonzero = np.sum(precision != 0) - precision.shape[0]
    p = precision.shape[0]
    penalty = np.log(n)/n*of_nonzero + \
        4 * epsilon * np.log(p*(p-1)/2)/n * of_nonzero
    return likelihood - penalty


def log_likelihood_t(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    score = 0
    for e, p in zip(emp_cov, precision):
        score += fast_logdet(p) - np.sum(e * p)
    return score


def BIC_t(emp_cov, precision):
    precision = np.array(precision)
    return log_likelihood_t(emp_cov, precision) - \
        (np.sum(precision != 0) - precision.shape[1]*precision.shape[0])


def EBIC_t(emp_cov, precision, n=100, epsilon=0.5):
    likelihood = log_likelihood_t(emp_cov, precision)
    n_variables = precision.shape[1]*precision.shape[0]
    of_nonzero = np.sum(precision != 0) - n_variables
    penalty = np.log(n)/n*of_nonzero + \
        4 * epsilon * np.log(n_variables)/n * of_nonzero
    return likelihood - penalty


def EBIC_m_t(emp_cov, precision, n=100, epsilon=0.5):
    likelihood = log_likelihood_t(emp_cov, precision)
    n_variables = precision.shape[1]*precision.shape[0]
    of_nonzero = np.sum(precision != 0) - n_variables
    penalty = np.log(n)/n*of_nonzero + \
        4 * epsilon * np.log(n_variables(n_variables-1)/2)/n * of_nonzero
    return likelihood - penalty
