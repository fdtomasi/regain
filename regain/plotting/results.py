import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc
from scipy import interp


def plot_roc_curves(true, preds, ax=None, fontsize=15):
    matplotlib.rcParams.update({'font.size': fontsize})
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    if true.ndim != 2:
        true = (true != 0).astype(int).ravel()
        preds_new = []
        for p in preds:
            preds_new.append(p.ravel())
        preds = preds_new

    for i, p in enumerate(preds):
        fpr, tpr, thresholds = roc_curve(true, p)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    plt.show()


def plot_roc_comparison(true, predictions, ax=None, filename="", fontsize=15,
                        colors=['red', 'blue', 'yellow']):
    matplotlib.rcParams.update({'font.size': fontsize})
    """

    preds:dict
        Dictionary with keys type of algorithm and values list of predictions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    c=0
    for key, preds in predictions.items():
        if len(preds) == 1:
            if true.ndim != 2:
                true = (true != 0).astype(int).ravel()
                preds_new = []
                for p in preds:
                    preds_new.append(p.ravel())
                preds = preds_new

            fpr, tpr, thresholds = roc_curve(true, preds[0])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1, color=colors[c],
                    label='ROC %s (AUC = %0.2f)' % (str(key), roc_auc))

        else:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            if true.ndim != 2:
                true = (true != 0).astype(int).ravel()
                preds_new = []
                for p in preds:
                    preds_new.append(p.ravel())
                preds = preds_new

            for i, p in enumerate(preds):
                fpr, tpr, thresholds = roc_curve(true, p)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                ax.plot(fpr, tpr, lw=1, color=colors[c], alpha=0.3)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color=colors[c],
                    label=r'Mean ROC %s (AUC = %0.2f $\pm$ %0.2f)' %
                    (str(key), mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                            alpha=.2, label=r'$\pm$ 1 std. dev.')
        c+=1
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    plt.grid()
    if filename != "":
        plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def plot_precision_recall_comparison(true, predictions, ax=None, filename="",
                                     fontsize=15, colors=['red', 'blue', 'yellow']):
    matplotlib.rcParams.update({'font.size': fontsize})
    """

    preds:dict
        Dictionary with keys type of algorithm and values list of predictions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot([1, 0], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    c = 0
    for key, preds in predictions.items():
        if len(preds) == 1:


            if true.ndim != 2:
                true = (true != 0).astype(int).ravel()
                preds_new = []
                for p in preds:
                    preds_new.append(p.ravel())
                preds = preds_new

            fpr, tpr, thresholds = precision_recall_curve(true, preds[0])
            roc_auc = auc(tpr, fpr)
            ax.plot(tpr, fpr, lw=1, color=colors[c],
                    label='PR %s (AUC = %0.2f)' % (str(key), roc_auc)
                    )

        else:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            if true.ndim != 2:
                true = (true != 0).astype(int).ravel()
                preds_new = []
                for p in preds:
                    preds_new.append(p.ravel())
                preds = preds_new

            for i, p in enumerate(preds):
                fpr, tpr, thresholds = precision_recall_curve(true, p)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(tpr, fpr)
                aucs.append(roc_auc)
                ax.plot(fpr, tpr, lw=1, color=colors[c],alpha=0.3)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc( mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color=colors[c],
                    label=r'Mean PR %s (AUC = %0.2f $\pm$ %0.2f)' %
                    (str(key), mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                            alpha=.2, label=r'$\pm$ 1 std. dev.')
        c += 1
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.legend(loc="lower right")
    plt.grid()
    if filename != "":
        plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
