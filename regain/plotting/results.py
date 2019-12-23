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
"""Utilities to plot results."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.utils.deprecation import deprecated


def plot_curve(
        true, predictions, mode='roc', ax=None, filename=None, fontsize=15,
        colors=None, multiple=False):
    """Plot a validation curve.

    Compare the predictions with different methods with the true adjacency
    matrix stored in true. If more than one comparison has been done it is
    possible to provide a list of true matrices and a list of predictions.

    Parameters
    ----------
    true: array-like, list of array-like
        The true adjacency matrix of the graph.
        If a list is provided the parameter multiple must be set to True.
    predictions : dict
        Keys is the algorithm, values list of predictions.
    mode : ('roc', 'precision_recall')
        Which curve to plot.
    ax: Matplotlib Axes object, optional default=None
        Draw the graph in the specified Matplotlib axes.
    filename: str, optional default=None
        Path to file where to save the resulting plot. If None no plot is
        saved.
    fontsize: int, optional default=15
        Size of font in the final plot.
    colors: list of colors, optional default=None
        Colors to use to plot the curves, the colors shoud be provided in the
        same order of the methods in the dictionary predictions for
        correspondence.
    multiple: boolean, optional default=False
        If True the parameter true is intended as a list and the roc curves
        are computed for different true adjacency matrix.

    """
    matplotlib.rcParams.update({'font.size': fontsize})
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(
        ([0, 1] if mode == 'roc' else [1, 0]), [0, 1], linestyle='--', lw=2,
        color='k', label='Chance', alpha=.8)
    curve_func = roc_curve if mode == 'roc' else precision_recall_curve
    for c, (key, preds) in enumerate(predictions.items()):
        if len(preds) == 1:
            if multiple:
                raise ValueError('Multiple true matrices and only one '
                                 'prediction')
            true = (~np.isclose(true, 0, rtol=1e-7)).astype(int).ravel()
            preds_new = []
            for p in preds:
                preds_new.append(p.ravel())
            preds = preds_new
            fpr, tpr, thresholds = curve_func(true, preds[0])
            kwargs = dict(
                lw=1, color=colors[c] if colors is not None else None)
            if mode == 'roc':
                roc_auc = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr,
                    label='%s %s (AUC = %0.2f)' % (mode.upper(), str(key),
                                                   roc_auc),
                    **kwargs)
            else:
                roc_auc = auc(tpr, fpr)
                ax.plot(
                    tpr, fpr,
                    label='%s %s (AUC = %0.2f)' % (mode.upper(), str(key),
                                                   roc_auc),
                    **kwargs)
        else:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            if multiple:
                true = [(~np.isclose(t, 0, rtol=1e-7)).astype(int).ravel()
                        for t in true]
            else:
                true = (~np.isclose(true, 0, rtol=1e-7)).astype(int).ravel()
            preds = [p.ravel() for p in preds]

            for i, p in enumerate(preds):
                if multiple:
                    t = true[i]
                else:
                    t = true
                fpr, tpr, thresholds = curve_func(t, p)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = int(mode != 'roc')
                roc_auc = auc(fpr, tpr) if mode == 'roc' else auc(tpr, fpr)
                aucs.append(roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = int(mode == 'roc')
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(
                mean_fpr, mean_tpr,
                color=colors[c] if colors is not None else None,
                label=r'Mean %s %s (AUC = %0.2f $\pm$ %0.2f)' %
                (mode, str(key), mean_auc, std_auc), lw=2, alpha=.9)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr, tprs_lower, tprs_upper, alpha=.2)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    if mode == 'roc':
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    else:
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')

    ax.legend(loc="lower right")
    ax.grid()
    if filename is not None:
        plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


@deprecated
def plot_roc_comparison(
        true, predictions, ax=None, filename=None, fontsize=15,
        colors=('red', 'blue', 'yellow')):
    return plot_curve(
        true, predictions, mode='roc', ax=ax, filename=filename,
        fontsize=fontsize, colors=colors)


@deprecated
def plot_precision_recall_comparison(
        true, predictions, ax=None, filename=None, fontsize=15,
        colors=('red', 'blue', 'yellow')):
    return plot_curve(
        true, predictions, mode='precision_recall', ax=ax, filename=filename,
        fontsize=fontsize, colors=colors)
