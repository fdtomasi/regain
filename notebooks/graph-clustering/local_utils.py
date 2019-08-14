import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score

from regain.utils import error_norm_time, structure_error

def convert_dict_to_df(input_dict, max_samples=5000):
    neww = {}
    for k, v in input_dict.items():
        if k[1] > max_samples: continue
        new = {}
        for kk, vv in v.items():
            new.update({(kk, i): vvv for i, vvv in enumerate(vv)})
        neww[k] = new

    res_df = pd.DataFrame(neww)
    res_df.index.name = ('measure', 'iter')

    rr = res_df.T.reset_index()
    rr = rr.rename(columns={'level_0': 'method', 'level_1': 'samples'})
    rr.method = rr.method.str.upper()
    return rr.set_index(['method', 'samples'])


def ensure_ticc_valid(new_r):
    for i in range(10):
        new_r['valid', i] = True

    for r in new_r.loc['TICC', 'model'].iterrows():
        sampl = r[0]
        for k, rrr in (r[1].iteritems()):
            if np.alltrue(rrr.labels_ == np.zeros_like(rrr.labels_)):
                new_r.loc[('TICC', sampl), ('structure_error', k)] = None


format_2e = lambda x: "{:.2e} (\pm {:.2e})".format(
    (np.nanmean(np.array(x).astype(float))),
    (np.nanstd(np.array(x).astype(float))))
format_3f = lambda x: "{:.3f} (\pm {:.3f})".format(
    (np.nanmean(np.array(x).astype(float))),
    (np.nanstd(np.array(x).astype(float))))


def set_results(
        vs, model, name, i, labels_true, labels_pred, thetas_true_sparse,
        thetas_true_rep, obs_precs_sparse, obs_precs, tac):
    th = name in ['wp', 'ticc']
    vs.setdefault((name, i), {}).setdefault('model', []).append(model)
    vs.setdefault(
        (name, i),
        {}).setdefault('v_measure',
                       []).append(v_measure_score(labels_true, labels_pred))
    vs.setdefault((name, i), {}).setdefault('structure_error', []).append(
        structure_error(
            thetas_true_sparse, obs_precs_sparse, no_diagonal=True,
            thresholding=th, eps=1e-5))
    vs.setdefault(
        (name, i),
        {}).setdefault('error_norm',
                       []).append(error_norm_time(thetas_true_rep, obs_precs))
    vs.setdefault((name, i), {}).setdefault('error_norm_sparse', []).append(
        error_norm_time(
            thetas_true_sparse, obs_precs_sparse))
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