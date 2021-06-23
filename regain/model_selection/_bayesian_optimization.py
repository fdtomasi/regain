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
"""Deprecated.

This module to port GPyOpt in a `sklearn-usable` way has been deprecated
in favor of scikit-optimize, which natively offer a Bayesian replacement
for the GridSearchCV.
"""
import numpy as np
from functools import partial
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection import cross_val_score
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import deprecated

try:
    import GPyOpt
    from GPyOpt.core.task.objective import SingleObjective
    from GPyOpt.core.task.cost import CostModel
    from GPyOpt.core.task.space import Design_space
    from GPyOpt.util.arguments_manager import ArgumentsManager
    from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
except ImportError:
    raise ImportError("Module GPyOpt is missing. Cannot use bayesian optimization")


@deprecated()
class _BayesianOptimization(GPyOpt.methods.BayesianOptimization, BaseSearchCV):
    """Wrapper for BayesianOptimization object from GPyOpt package.

    This is done to use estimator in the same way as scikit-learn.
    For more information, see GPyOpt package.
    """

    def __init__(
        self,
        estimator,
        domain=None,
        constraints=None,
        cost_withGradients=None,
        model_type="GP",
        X=None,
        Y=None,
        initial_design_numdata=5,
        initial_design_type="random",
        acquisition_type="EI",
        normalize_Y=True,
        exact_feval=False,
        acquisition_optimizer_type="lbfgs",
        model_update_interval=1,
        evaluator_type="sequential",
        batch_size=1,
        num_cores=1,
        verbosity_model=False,
        verbosity=False,
        de_duplication=False,
        max_iter=50,
        refit=True,
        cv=None,
        scoring=None,
        n_jobs=1,
        verbose=False,
        **kwargs
    ):
        """Initialise the estimator."""
        # super(BayesianOptimization, self).__init__(
        #     f=f, domain=domain, constraints=constraints,
        #     cost_withGradients=cost_withGradients, model_type=model_type,
        #     X=X, Y=Y, initial_design_numdata=initial_design_numdata,
        #     initial_design_type=initial_design_type,
        #     acquisition_type=acquisition_type, normalize_Y=normalize_Y,
        #     exact_feval=exact_feval,
        #     acquisition_optimizer_type=acquisition_optimizer_type,
        #     model_update_interval=model_update_interval,
        #     evaluator_type=evaluator_type, batch_size=batch_size,
        #     num_cores=num_cores,
        #     verbosity=verbosity, verbosity_model=verbosity_model,
        #     maximize=True, de_duplication=de_duplication, **kwargs)
        self.modular_optimization = False
        self.initial_iter = True
        self.verbosity = verbosity
        self.verbosity_model = verbosity_model
        self.model_update_interval = model_update_interval
        self.de_duplication = de_duplication
        self.kwargs = kwargs

        # --- Handle the arguments passed via kargs
        self.problem_config = ArgumentsManager(kwargs)

        # --- CHOOSE design space
        self.constraints = constraints
        self.domain = domain or {}
        self.space = Design_space(self.domain, self.constraints)

        # --- CHOOSE objective function
        self.objective_name = kwargs.get("objective_name", "") or "no_name"
        self.batch_size = batch_size
        self.num_cores = num_cores
        # self.maximize = True
        # self.objective = None

        # --- CHOOSE the cost model
        self.cost = CostModel(cost_withGradients)

        # --- CHOOSE initial design
        self.X = X  # parameters
        self.Y = Y  # evaluation
        self.initial_design_type = initial_design_type
        self.initial_design_numdata = initial_design_numdata

        # --- CHOOSE the model type. If an instance of a GPyOpt model is passed (possibly user defined), it is used.
        # note that this 2 options are not used with the predefined model
        self.model_type = model_type
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y

        if "model" in self.kwargs and isinstance(kwargs["model"], GPyOpt.models.base.BOModel):
            self.model = kwargs["model"]
            self.model_type = "User defined model used."
            if self.verbose:
                print("Using a model defined by the used.")
        else:
            self.model = self._model_chooser()

        # --- CHOOSE the acquisition optimizer_type
        # This states how the discrete variables are handled (exact search or rounding)
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.acquisition_optimizer = AcquisitionOptimizer(
            self.space, self.acquisition_optimizer_type, model=self.model
        )

        # --- CHOOSE acquisition function. If an instance of an acquisition is passed (possibly user defined), it is used.
        self.acquisition_type = acquisition_type
        if "acquisition" in self.kwargs and isinstance(kwargs["acquisition"], GPyOpt.acquisitions.AcquisitionBase):
            self.acquisition = kwargs["acquisition"]
            self.acquisition_type = "User defined acquisition used."
            if self.verbose:
                print("Using an acquisition defined by the used.")
        else:
            self.acquisition = self._acquisition_chooser()

        # --- CHOOSE evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()

        # --- Create optimization space
        self.cost = CostModel(self.cost)
        self.normalization_type = "stats"  # not added in the API

        self.estimator = estimator
        self.cv = cv
        self.max_iter = max_iter
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs

    @property
    def param_names(self):
        return [element["name"] for element in self.domain]

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
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

        score_function = partial(
            cross_val_score,
            X=X,
            y=y,
            groups=groups,
            scoring=self.scoring,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            fit_params=fit_params,
        )
        self.f = partial(
            _fit_score,
            mdl=self.estimator,
            param_names=self.param_names,
            score_function=score_function,
        )

        self.objective = SingleObjective(self.f, self.batch_size, self.objective_name)
        self._init_design_chooser()

        self.run_optimization(max_iter=self.max_iter, verbosity=self.verbosity)

        self.best_index_ = self.Y.argmin()
        self.best_params_ = dict(zip(self.param_names, 10 ** self.X[self.best_index_]))
        self.best_score_ = self.Y[self.Y.argmin()]

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers["score"]

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

        return self

    def plot_acquisition(self):
        old_X, old_Y = self.X, self.Y
        # remove inf, to avoid error in plot_acquisition
        self.X = self.X[~np.isinf(self.Y).ravel()]
        self.Y = self.Y[~np.isinf(self.Y).ravel()]
        super(_BayesianOptimization, self).plot_acquisition()
        # restore original X and Y
        self.X, self.Y = old_X, old_Y


def _fit_score(x, score_function, mdl=None, param_names=None):
    x = 10 ** np.atleast_2d(x)
    return -np.array([np.mean(score_function(mdl.set_params(**dict(zip(param_names, pars))))) for pars in x])[:, None]
