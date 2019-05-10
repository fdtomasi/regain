import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid
from scipy.special import binom
from sklearn.base import clone


class GraphicalModelStabilitySelection():

    def __init__(self, estimator, sampling_size=100, n_repetitions=100,
                 params_grid=dict(), refit=True):
        self.estimator = estimator
        self.sampling_size = sampling_size
        self.n_repetitions = n_repetitions
        self.params_grid = params_grid
        self.refit = refit

    def fit(self, X, y=None):

        n, p = X.shape

        # check params
        if self.n_repetitions < 10:
            raise ValueError("Insert a number of repetitions that is higher or"
                             "equal than 10")

        if self.params_grid == dict():
            raise ValueError("Please specify an interval for the parameters "
                             "search")

        if self.sampling_size >= n:
            raise ValueError("The sampling size has to be lower than the "
                             "number ofsamples. Found %d, should be lower "
                             "than %d, suggested %d" %
                             (self.sampling_size, n, int(10*np.sqrt(n))))
        new_params = {}
        for key, value in self.params_grid.items():
            new_params[key] = [1/v for v in np.sort(value)]

        pg = ParameterGrid(new_params)
        res = {}
        instabilities = []
        params_list = []
        estimator = clone(self.estimator)
        for i, params in enumerate(pg.__iter__()):
            estimator.set_params(**params)
            ss = ShuffleSplit(n_splits=self.n_repetitions,
                              test_size=n - self.sampling_size,
                              train_size=self.sampling_size)
            connettivity_matrices = []
            for train, _ in ss.split(X):
                X_sampled = X[train, :]
                estimator.fit(X_sampled)
                connettivity_matrices.append(estimator.get_precision())

            mean_connectivity = np.zeros_like(connettivity_matrices[0])
            for c in connettivity_matrices:
                binarized = (c.copy() != 0).astype(int)
                mean_connectivity += binarized
            mean_connectivity /= self.n_repetitions

            xi_matrix = 2 * mean_connectivity*(1 - mean_connectivity)
            upper = xi_matrix[np.triu_indices_from(xi_matrix)]
            global_instability = np.sum(upper)/binom(p, 2)

            res[i] = {'params': params,
                      'matrices': connettivity_matrices,
                      'score': global_instability}
            instabilities.append(global_instability)
            params_list.append(params)

        # monotonize instabilities
        monotonized_instabilities = [instabilities[0]] + \
                                    [np.max(instabilities[:i])
                                     for i in range(1, len(instabilities))]
        best_params_ix = \
            np.where(np.array(monotonized_instabilities) <= 0.05)[0][-1]
        self.best_params = params_list[best_params_ix]

        if self.refit:
            self.best_estimator_ = clone(estimator)
            self.best_estimator_.set_params(**self.best_params)
            self.best_estimator_.fit(X)

        return self
