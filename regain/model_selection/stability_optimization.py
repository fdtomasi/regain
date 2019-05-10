import time
import warnings
from collections import defaultdict
from functools import partial
from itertools import product

import numpy as np
from scipy.special import binom
from scipy.stats import rankdata
from sklearn.base import clone, is_classifier
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection import GridSearchCV, ParameterGrid, ShuffleSplit
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (_aggregate_score_dicts,
                                                 _fit_and_score)
from sklearn.utils import deprecated
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable


def global_instability(estimators):
    precisions = [estimator.get_precision() for estimator in estimators]

    if precisions[0].ndim == 2:
        n_times = 1
        triu_idx = np.triu_indices_from(precisions[0], 1)
        mean_connectivity = np.zeros_like(precisions[0])[triu_idx]
        for c in precisions:
            mean_connectivity += (c[triu_idx].copy() != 0).astype(int)
    else:
        # for tri dimensional matrices
        n_times = precisions[0].shape[0]
        triu_idx = np.triu_indices_from(precisions[0][0], 1)
        mean_connectivity = np.array(
            [
                np.zeros_like(precisions[0][0])[triu_idx]
                for i in range(n_times)
            ])
        for c in precisions:
            for i in range(n_times):
                mean_connectivity[i] += (c[i][triu_idx].copy() !=
                                         0).astype(int)

    mean_connectivity /= len(estimators)
    xi_matrix = 2 * mean_connectivity * (1 - mean_connectivity)
    return np.sum(xi_matrix) / (binom(precisions[0].shape[1], 2) * n_times)


class GraphicalModelStabilitySelection(GridSearchCV):
    def __init__(
            self, estimator, param_grid, scoring=None, n_jobs=None,
            iid='deprecated', refit=True, cv='warn', verbose=0,
            pre_dispatch='2*n_jobs', error_score='raise-deprecating',
            return_train_score=False, n_repetitions=10, sampling_size=10):
        super().__init__(
            estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
            error_score=error_score, return_train_score=return_train_score,
            param_grid=param_grid)
        self.n_repetitions = n_repetitions
        self.sampling_size = sampling_size

    @deprecated
    def fit__(self, X, y=None):
        n, p = X.shape
        # check params
        if self.n_repetitions < 10:
            raise ValueError(
                "Insert a number of repetitions that is higher or"
                "equal than 10")

        if self.param_grid == dict():
            raise ValueError(
                "Please specify an interval for the parameters "
                "search")

        if self.sampling_size >= n:
            raise ValueError(
                "The sampling size has to be lower than the "
                "number ofsamples. Found %d, should be lower "
                "than %d, suggested %d" %
                (self.sampling_size, n, int(10 * np.sqrt(n))))
        new_params = {}
        for key, value in self.param_grid.items():
            new_params[key] = [1 / v for v in np.sort(value)]

        pg = ParameterGrid(new_params)
        res = {}
        instabilities = []
        params_list = []
        estimator = clone(self.estimator)
        for i, params in enumerate(pg.__iter__()):
            estimator.set_params(**params)
            ss = ShuffleSplit(
                n_splits=self.n_repetitions, test_size=n - self.sampling_size,
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

            xi_matrix = 2 * mean_connectivity * (1 - mean_connectivity)
            upper = xi_matrix[np.triu_indices_from(xi_matrix)]
            global_instability = np.sum(upper) / binom(p, 2)

            res[i] = {
                'params': params,
                'matrices': connettivity_matrices,
                'score': global_instability
            }
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

        self.cv_results_ = res
        return self

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str)
                    or  # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError(
                    "For multi-metric scoring, the parameter "
                    "refit must be set to a scorer key or a "
                    "callable to refit an estimator with the "
                    "best parameter setting on the whole "
                    "data and make the best_* attributes "
                    "available for that metric. If this is "
                    "not needed, refit should be set to "
                    "False explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            # refit_metric = 'score'
            refit_metric = 'instability'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers, fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True, return_times=True,
            return_parameters=False, error_score=self.error_score,
            verbose=self.verbose, return_estimator=True)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator), X, y, train=train, test=test,
                        parameters=parameters, **fit_and_score_kwargs)
                    for parameters, (train, test) in product(
                        candidate_params, cv.split(X, y, groups)))

                if len(out) < 1:
                    raise ValueError(
                        'No fits were performed. '
                        'Was the CV iterator empty? '
                        'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        'cv.split and cv.get_n_splits returned '
                        'inconsistent results. Expected {} '
                        'splits, got {}'.format(
                            n_splits,
                            len(out) // n_candidates))

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
                return results

            self._run_search(evaluate_candidates)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, (int, np.integer)):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0
                        or self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s" %
                                           refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" %
                                           refit_metric][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _format_results(self, candidate_params, scorers, n_splits, out):
        n_candidates = len(candidate_params)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (
                train_score_dicts, test_score_dicts, test_sample_counts,
                fit_time, score_time, estimators) = zip(*out)
        else:
            (
                test_score_dicts, test_sample_counts, fit_time, score_time,
                estimators) = zip(*out)

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        results = {}

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array,
                             dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" %
                            (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average(
                    (array - array_means[:, np.newaxis])**2, axis=1,
                    weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray, np.empty(n_candidates, ), mask=True,
                dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(
            test_sample_counts[:n_splits], dtype=np.int)

        if self.iid != 'deprecated':
            warnings.warn(
                "The parameter 'iid' is deprecated in 0.22 and will be "
                "removed in 0.24.", DeprecationWarning)
            iid = self.iid
        else:
            iid = False

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store(
                'test_%s' % scorer_name, test_scores[scorer_name], splits=True,
                rank=True, weights=test_sample_counts if iid else None)
            if self.return_train_score:
                _store(
                    'train_%s' % scorer_name, train_scores[scorer_name],
                    splits=True)

        estimators = np.asarray(estimators).reshape(n_candidates, n_splits)
        array_means = np.array(
            [global_instability(e_split) for e_split in estimators])

        # monotonize instabilities - require ordered parameters,
        # from high sparsity to low
        monotonized_instabilities = [array_means[0]] + [
            np.max(array_means[:i]) for i in range(1, array_means.size)
        ]
        monotonized_instabilities = np.array(monotonized_instabilities)
        # discard high values
        monotonized_instabilities[monotonized_instabilities > 0.05] = -np.inf

        key_name = 'test_instability'
        results['raw_%s' % key_name] = array_means
        results['mean_%s' % key_name] = monotonized_instabilities
        results["rank_%s" % key_name] = np.asarray(
            rankdata(-monotonized_instabilities, method='min'), dtype=np.int32)

        return results
