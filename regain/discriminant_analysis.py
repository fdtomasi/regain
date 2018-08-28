"""Discriminant analysis with REGAIN."""

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from regain.covariance.graph_lasso_ import fast_logdet
from regain.covariance.time_graph_lasso_ import TimeGraphLasso


class DiscriminantAnalysis(TimeGraphLasso, QuadraticDiscriminantAnalysis):
    """docstring for QuadraticDiscriminantAnalysis."""
    def __init__(self, alpha=0.01, beta=1., mode='admm', rho=1.,
                 time_on_axis='first', tol=1e-4, rtol=1e-4,
                 psi='laplacian', max_iter=100,
                 verbose=False, assume_centered=False,
                 return_history=False,
                 update_rho_options=None,
                 priors=None):
        super(DiscriminantAnalysis, self).__init__(
            alpha=alpha, beta=beta, mode=mode, rho=rho,
            time_on_axis=time_on_axis, tol=tol, rtol=rtol,
            psi=psi, max_iter=max_iter,
            verbose=verbose, assume_centered=assume_centered,
            return_history=return_history,
            update_rho_options=update_rho_options)
        self.priors = np.asarray(priors) if priors is not None else None

    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

            .. versionchanged:: 0.19
               ``store_covariances`` has been moved to main constructor as
               ``store_covariance``

            .. versionchanged:: 0.19
               ``tol`` has been moved to main constructor.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors
        means = []
        data = []
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError('y has only 1 sample in class %s, covariance '
                                 'is ill defined.' % str(self.classes_[ind]))
            data.append(Xg)

        super(DiscriminantAnalysis, self).fit(data)
        self.means_ = np.asarray(means)
        return self

    def fit_precomputed(self, X, y, precisions):
        """Fit the model according to the given training data and parameters.

            .. versionchanged:: 0.19
               ``store_covariances`` has been moved to main constructor as
               ``store_covariance``

            .. versionchanged:: 0.19
               ``tol`` has been moved to main constructor.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors
        means = []
        data = []
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError('y has only 1 sample in class %s, covariance '
                                 'is ill defined.' % str(self.classes_[ind]))
            data.append(Xg)

        self.precision_ = precisions
        self.means_ = np.asarray(means)
        return self

    def _decision_function(self, X):
        check_is_fitted(self, 'classes_')

        X = check_array(X)
        norm2 = []
        for i in range(len(self.classes_)):
            Xm = X - self.means_[i]
            # X2 = np.dot(Xm, R * (S ** (-0.5)))
            X2 = np.linalg.multi_dot((Xm, self.precision_[i], Xm.T))
            norm2.append(np.diag(X2))
        norm2 = np.array(norm2).T   # shape = [len(X), n_classes]
        u = np.asarray([- fast_logdet(s) for s in self.precision_])
        return (-0.5 * (norm2 + u) + np.log(self.priors_))

    def _decision_function2(self, X):
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
