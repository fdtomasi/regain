import numpy as np
from sklearn.metrics import v_measure_score

from regain.utils import error_norm_time, structure_error


format_2e = lambda x: "{:.2e} (\pm {:.2e})".format(
    (np.nanmean(np.array(x).astype(float))),
    (np.nanstd(np.array(x).astype(float))))
format_3f = lambda x: "{:.3f} (\pm {:.3f})".format(
    (np.nanmean(np.array(x).astype(float))),
    (np.nanstd(np.array(x).astype(float))))


def set_results(
        vs, model, name, i, labels_true, labels_pred, thetas_true_sparse,
        thetas_true_rep, obs_precs_sparse, obs_precs, tac):
    vs.setdefault((name, i), {}).setdefault('model', []).append(model)
    vs.setdefault(
        (name, i),
        {}).setdefault('v_measure',
                       []).append(v_measure_score(labels_true, labels_pred))
    vs.setdefault((name, i), {}).setdefault('structure_error', []).append(
        structure_error(
            thetas_true_sparse, obs_precs_sparse, no_diagonal=True))
    vs.setdefault(
        (name, i),
        {}).setdefault('error_norm',
                       []).append(error_norm_time(thetas_true_rep, obs_precs))
    vs.setdefault((name, i), {}).setdefault('error_norm_sparse', []).append(
        error_norm_time(thetas_true_sparse, obs_precs_sparse))
    vs.setdefault((name, i), {}).setdefault('time', []).append(tac)


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == (s.min() if s.name in ['time', 'error_norm'] else s.max())
    s.loc[is_max] = s[is_max].apply(lambda x: '\\textbf{%s}' % (x))
    return ['background-color: yellow' if v else '' for v in is_max]


def highlight_max_std(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    ss = s.str.split(' ').apply(lambda x: x[0]).astype(float)
    is_max = ss.astype(float) == (
        ss.min() if s.name in ['time', 'error_norm'] else ss.max())
    s.loc[is_max] = s[is_max].apply(lambda x: '\\bm{%s}' % (x))
    return ['background-color: yellow' if v else '' for v in is_max]