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

"""Discriminant analysis with REGAIN."""

import numpy as np
from scipy import linalg
from six.moves import range
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import check_array, check_X_y, deprecated
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from regain.utils import ensure_posdef

__all__ = ("DiscriminantAnalysis", "PrecomputedDiscriminantAnalysis")


class DiscriminantAnalysis(QuadraticDiscriminantAnalysis):
    """Quadratic Discriminant Analysis using LTGL

    A classifier with a quadratic decision boundary, generated
    by fitting class conditional densities to the data
    and using Bayes' rule.

    The model fits a Gaussian density to each class using LTGL.

    Parameters
    ----------
    estimator : class
        Estimator to compute precision and covariance matrices

    priors : array, optional, shape = [n_classes]
        Priors on classes

    Attributes
    ----------
    covariance_ : list of array-like, shape = [n_features, n_features]
        Covariance matrices of each class.

    means_ : array-like, shape = [n_classes, n_features]
        Class means.

    priors_ : array-like, shape = [n_classes]
        Class priors (sum to 1).

    rotations_ : list of arrays
        For each class k an array of shape [n_features, n_k], with
        ``n_k = min(n_features, number of elements in class k)``
        It is the rotation of the Gaussian distribution, i.e. its
        principal axis.

    scalings_ : list of arrays
        For each class k an array of shape [n_k]. It contains the scaling
        of the Gaussian distributions along its principal axes, i.e. the
        variance in the rotated coordinate system.
    """

    def __init__(self, estimator, ensure_posdef=False, priors=None):
        self.estimator = estimator
        self.priors = np.asarray(priors) if priors is not None else None
        self.ensure_posdef = ensure_posdef

    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("estimator does not implement a `fit` method.")
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("The number of classes has to be greater than" " one; got %d class" % (n_classes))
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors
        # means = []
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            # meang = Xg.mean(0)
            # means.append(meang)
            if len(Xg) == 1:
                raise ValueError(
                    "y has only 1 sample in class %s, covariance " "is ill defined." % str(self.classes_[ind])
                )
            # data.append(Xg)

        self.estimator.fit(X, y)
        self.precision_ = self.estimator.precision_

        if hasattr(self.estimator, "covariance_"):
            self.covariance_ = self.estimator.covariance_
        elif not self.ensure_posdef:
            self.covariance_ = np.array([linalg.pinvh(p) for p in self.precision_])

        if self.ensure_posdef:
            # replace diagonal
            ensure_posdef(self.precision_, inplace=True)
            self.covariance_ = np.array([linalg.pinvh(p) for p in self.precision_])
        self.means_ = self.estimator.location_
        return self

    @deprecated("it will be removed in v0.2.0. Use `_decision_function` instead")
    def _decision_function2(self, X):
        check_is_fitted(self, "classes_")

        X = check_array(X)
        precisions = self.get_observed_precision()
        norm2 = []
        for i in range(len(self.classes_)):
            Xm = X - self.means_[i]
            # X2 = np.dot(Xm, R * (S ** (-0.5)))
            X2 = np.linalg.multi_dot((Xm, precisions[i], Xm.T))
            norm2.append(np.diag(X2))
        norm2 = np.array(norm2).T  # shape = [len(X), n_classes]
        u = np.asarray([-fast_logdet(s) for s in precisions])
        return -0.5 * (norm2 + u) + np.log(self.priors_)

    def _decision_function(self, X):
        rotations, scalings = [], []
        for p in self.covariance_:
            R, S, Rt = np.linalg.svd(p, full_matrices=False)
            rotations.append(Rt.T)
            scalings.append(S)
        self.rotations_ = rotations
        self.scalings_ = scalings
        return super(DiscriminantAnalysis, self)._decision_function(X)

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class PrecomputedDiscriminantAnalysis(BaseEstimator):
    def __init__(self, precision):
        self.precision_ = np.array(precision)

    def fit(self, X, y=None):
        """Dummy fit."""
        pass
