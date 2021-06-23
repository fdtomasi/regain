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

import time

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
import warnings
from collections import defaultdict
from functools import partial
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom
from scipy.stats import rankdata
from sklearn.base import clone, is_classifier
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection import GridSearchCV, ParameterGrid, ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts, _fit_and_score
from sklearn.utils import deprecated
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable

warnings.simplefilter("ignore")


def global_instability(estimators):
    """Computes instability of the graphs inferred from estimators.

    Parameters
    ----------

    estimators: list of fitted graphical models estimator
        Each estimator contains the inferred adjacency matrix that is taken
        to compute the global instability of the list of estimator.


    Returns
    -------
    float:
        Instability value.
    """

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
        mean_connectivity = np.array([np.zeros_like(precisions[0][0])[triu_idx] for i in range(n_times)])
        for c in precisions:
            for i in range(n_times):
                mean_connectivity[i] += (c[i][triu_idx].copy() != 0).astype(int)

    mean_connectivity /= len(estimators)
    xi_matrix = 2 * mean_connectivity * (1 - mean_connectivity)
    return np.sum(xi_matrix) / (binom(precisions[0].shape[1], 2) * n_times)


def graphlet_instability(estimators):
    """
    Computes graphlet instability of the graphs inferred from estimators.

    Parameters
    ----------

    estimators: list of fitted graphical models estimator
        Each estimator contains the inferred adjacency matrix that is taken
        to compute the global instability of the list of estimator.


    Returns
    -------
    float:
        Graphlet instability value.
    """

    from netanalytics.graphlets import GCD, graphlet_degree_vectors
    import networkx as nx

    n = len(estimators)
    precisions = [estimator.get_precision() for estimator in estimators]

    if precisions[0].ndim == 2:
        gdvs = []
        for p in precisions:
            g = nx.from_numpy_array(p - np.diag(np.diag(p)))
            gdvs.append(graphlet_degree_vectors(list(g.nodes), list(g.edges), graphlet_size=4))
        distances = []
        for i in range(len(gdvs)):
            for j in range(i + 1, len(gdvs)):
                distances.append(GCD(gdvs[i], gdvs[j])[1])
    else:
        n_times = precisions[0].shape[0]
        gdvs = []
        for p in precisions:
            times = []
            for t in range(n_times):
                g = nx.from_numpy_array(p[t] - np.diag(np.diag(p[t])))
                times.append(graphlet_degree_vectors(list(g.nodes), list(g.edges), graphlet_size=4))
            gdvs.append(times)

        distances = []
        for i in range(len(gdvs)):
            for j in range(i + 1, len(gdvs)):
                distance = 0
                for t in range(n_times):
                    distance += GCD(gdvs[i][t], gdvs[j][t])[1]
                distances.append(distance / n_times)
    return 2 / (n * (n - 1)) * np.sum(distances)


def upper_bound(estimators):
    precisions = [estimator.get_precision() for estimator in estimators]
    if precisions[0].ndim == 2:
        n_times = 1
        triu_idx = np.triu_indices_from(precisions[0], 1)
        mean_connectivity = np.zeros_like(precisions[0])[triu_idx]
        for c in precisions:
            mean_connectivity += (c[triu_idx].copy() != 0).astype(int)
    else:
        # for tri-dimensional matrices
        n_times = precisions[0].shape[0]
        triu_idx = np.triu_indices_from(precisions[0][0], 1)
        mean_connectivity = np.array([np.zeros_like(precisions[0][0])[triu_idx] for i in range(n_times)])
        for c in precisions:
            for i in range(n_times):
                mean_connectivity[i] += (c[i][triu_idx].copy() != 0).astype(int)
    mean_connectivity /= len(estimators)
    xi_matrix = 2 * mean_connectivity * (1 - mean_connectivity)

    p = precisions[0].shape[1]
    theta_hat = np.sum(xi_matrix)
    theta_hat = theta_hat / (p * (p - 1) / 2) if precisions[0].ndim == 2 else theta_hat / (n_times * p * (p - 1) / 2)
    return 4 * theta_hat * (1 - theta_hat)


def _check_param_order(param_grid):
    """Ensure that the parameters are in descending order.

    This is required for stability to be correctly computed.
    """
    if hasattr(param_grid, "items"):
        param_grid = [param_grid]

    pg = []
    for p in param_grid:
        for name, v in p.items():
            pg.append((name, np.sort(v)[::-1]))

    return dict(pg)


class GraphicalModelStabilitySelection(GridSearchCV):
    """Stability based search over specified parameter values for an estimator.

    It implements a "fit" and a "score" method.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    scoring : string, callable, list/tuple, dict or None, default: None
        Not used.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    cv : int, cross-validation generator or an iterable, optional
        Not used.

    refit : boolean, string, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_parameters_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.
        The refitted estimator is made available at the ``best_estimator_``
        attribute.

    verbose : integer
        Controls the verbosity: the higher, the more messages.
    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.
    return_train_score : boolean, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    mode: string, optional default='stars'
        The alternative option is gstars.
        If set to stars computes only the single edge stability. If gstars is
        passed it computes graphlet stability.

    sampling_size: int, optional default None
        The sample size to each repetition of the stability procedure. If None
        value is taken as ` int(min(10*np.sqrt(X.shape[0]), X.shape[0]-10))`

    """

    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs=None,
        iid="deprecated",
        refit=True,
        cv="warn",
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score="raise-deprecating",
        mode="stars",
        return_train_score=False,
        n_repetitions=10,
        sampling_size=None,
    ):
        super(GraphicalModelStabilitySelection, self).__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=StratifiedShuffleSplit(train_size=sampling_size, n_splits=n_repetitions),
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
            param_grid=param_grid,
        )
        self.mode = mode
        self.n_repetitions = n_repetitions
        self.sampling_size = sampling_size
        self.param_grid = _check_param_order(param_grid)

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

        if self.sampling_size is None:
            self.sampling_size = int(min(10 * np.sqrt(X.shape[0]), X.shape[0] - 10))
            self.cv = StratifiedShuffleSplit(train_size=self.sampling_size, n_splits=self.n_repetitions)
        if y is not None:
            n_classes = np.unique(y).shape[0]
            if self.sampling_size % n_classes != 0:
                warnings.warn("Changing sampling size, divisible for the " "number of classes.")
                self.sampling_size = (self.sampling_size // n_classes) * n_classes
                self.cv = StratifiedShuffleSplit(
                    n_splits=self.n_repetitions,
                    train_size=self.sampling_size,
                    test_size=X.shape[0] - self.sampling_size,
                )
        else:
            y = np.ones(X.shape[0])
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if (
                self.refit is not False
                and (
                    not isinstance(self.refit, str)
                    or self.refit not in scorers  # This will work for both dict / list (tuple)
                )
                and not callable(self.refit)
            ):
                raise ValueError(
                    "For multi-metric scoring, the parameter "
                    "refit must be set to a scorer key or a "
                    "callable to refit an estimator with the "
                    "best parameter setting on the whole "
                    "data and make the best_* attributes "
                    "available for that metric. If this is "
                    "not needed, refit should be set to "
                    "False explicitly. %r was passed." % self.refit
                )
            else:
                refit_metric = self.refit
        else:
            # refit_metric = 'score'
            refit_metric = "instability"

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
            return_estimator=True,
        )
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
                        " totalling {2} fits".format(n_splits, n_candidates, n_candidates * n_splits)
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        **fit_and_score_kwargs
                    )
                    for parameters, (train, test) in product(candidate_params, cv.split(X, y, groups))
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. " "Was the CV iterator empty? " "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(all_candidate_params, scorers, n_splits, all_out)
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
                    raise TypeError("best_index_ returned is not an integer")
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError("best_index_ index out of range")
            else:
                self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
                print(self.best_index_)
                self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(**self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers["score"]

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _format_results(self, candidate_params, scorers, n_splits, out):
        n_candidates = len(candidate_params)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_score_dicts, test_score_dicts, test_sample_counts, fit_time, score_time, estimators) = zip(*out)
        else:
            (test_score_dicts, test_sample_counts, fit_time, score_time, estimators) = zip(*out)

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
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results["std_%s" % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(rankdata(-array_means, method="min"), dtype=np.int32)

        _store("fit_time", fit_time)
        _store("score_time", score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=np.int)

        if self.iid != "deprecated":
            warnings.warn(
                "The parameter 'iid' is deprecated in 0.22 and will be " "removed in 0.24.", DeprecationWarning
            )
            iid = self.iid
        else:
            iid = False

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store(
                "test_%s" % scorer_name,
                test_scores[scorer_name],
                splits=True,
                rank=True,
                weights=test_sample_counts if iid else None,
            )
            if self.return_train_score:
                _store("train_%s" % scorer_name, train_scores[scorer_name], splits=True)

        estimators = np.asarray(estimators).reshape(n_candidates, n_splits)
        array_means = np.array([global_instability(e_split) for e_split in estimators])

        # monotonize instabilities - require ordered parameters,
        # from high sparsity to low

        monotonized_instabilities = [array_means[0]] + [np.max(array_means[:i]) for i in range(1, array_means.size)]
        monotonized_instabilities = np.array(monotonized_instabilities)
        self.monotonized_instabilities = np.copy(monotonized_instabilities)

        if self.mode.lower() == "gstars":
            graphlets_stability = np.array([graphlet_instability(e_split) for e_split in estimators])
            self.graphlets_instabilities = np.copy(graphlets_stability)

            upper_bounds = np.array([upper_bound(e_split) for e_split in estimators])
            upper_bounds = [upper_bounds[0]] + [np.max(upper_bounds[:i]) for i in range(1, upper_bounds.size)]
            self.upper_bounds = np.array(upper_bounds)
            lb = np.where(np.array(monotonized_instabilities) <= 0.05)[0]
            ub = np.where(np.array(upper_bounds) <= 0.05)[0]
            lb = lb[-1] if lb.size != 0 else len(monotonized_instabilities)
            ub = ub[-1] if ub.size != 0 else 0
            self.lower_bound = lb
            self.upper_bound = ub
            graphlets_stability[0:ub] = np.inf
            graphlets_stability[lb + 1 :] = np.inf

            key_name = "test_instability"
            results["raw_%s" % key_name] = array_means
            results["mean_%s" % key_name] = monotonized_instabilities
            results["rank_%s" % key_name] = np.asarray(rankdata(graphlets_stability, method="min"), dtype=np.int32)
        else:
            # discard high values
            monotonized_instabilities[monotonized_instabilities > 0.05] = -np.inf
            key_name = "test_instability"
            results["raw_%s" % key_name] = array_means
            results["mean_%s" % key_name] = monotonized_instabilities
            results["rank_%s" % key_name] = np.asarray(
                rankdata(-monotonized_instabilities, method="min"), dtype=np.int32
            )
        self.results = results
        return results

    def plot(self, axis=None, figsize=(15, 10), filename="", fontsize=15):
        """
        Function that plots the instability curves obtained on data.
        """
        matplotlib.rcParams.update({"font.size": fontsize})
        if self.mode.lower() == "gstars":
            if axis is None:
                fig, axis = plt.subplots(2, figsize=figsize)
            axis[0].plot(self.monotonized_instabilities, label="Instabilities")
            axis[0].plot(np.array(self.upper_bounds), label="Upper bound instabilities")
            axis[0].axhline(0.05, color="red")
            axis[0].axvline(self.lower_bound, color="violet", label="Lower bound")
            axis[0].axvline(self.upper_bound, color="green", label="Upper bound")
            axis[0].grid()
            axis[0].legend()
            axis[0].set_xticks(np.arange(len(self.monotonized_instabilities)))
            axis[0].set_xticklabels(self.results["params"])
            axis[1].plot(self.graphlets_instabilities, label="Graphlet instabilities")
            axis[1].axvline(self.lower_bound, color="violet")
            axis[1].axvline(self.upper_bound, color="green")
            axis[1].grid()
            axis[1].legend()
            axis[1].set_xticks(np.arange(len(self.monotonized_instabilities)))
            axis[1].set_xticklabels(self.results["params"])
            for tick in axis[0].get_xticklabels():
                tick.set_rotation(90)
            for tick in axis[1].get_xticklabels():
                tick.set_rotation(90)

            plt.tight_layout()
            if filename != "":
                plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
            plt.show()
        else:
            if axis is None:
                fig, axis = plt.subplots(figsize=figsize)
            axis.set_title("Monotonized instabilities")
            axis.plot(self.monotonized_instabilities)
            axis.axhline(0.05, color="red")
            axis.set_xticks(np.arange(len(self.monotonized_instabilities)))
            axis.set_xticklabels(self.results["params"])
            for tick in axis.get_xticklabels():
                tick.set_rotation(90)
            if filename != "":
                plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
            plt.show()
