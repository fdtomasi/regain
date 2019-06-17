import warnings

import numpy as np

from sklearn.utils import check_array
import matplotlib.pyplot as plt

from .utils import erdos_renyi


def clustering_coefficient(X):
    degrees = np.sum(X, axis=1)
    D = np.zeros(X.shape[0])
    for node in range(X.shape[0]):
        neighbors = np.where(X[node,:]!=0)[0]
        subset = X[neighbors, :]
        subset = subset[:, neighbors]
        D[node] = np.sum(subset)/2

    C_v = 0
    for i, d in enumerate(degrees):
        if d <= 1:
            continue
        #print(D[i])
        #print(degrees[i])
        C_v += 2*D[i]/(degrees[i]*(degrees[i] -1))
    degree_greter = degrees.copy()
    degree_greter[np.where(degree_greter<=1)] = 0
    #print(np.sum(degree_greter!=0))
    C_v /= np.sum(degree_greter!=0)
    return C_v

def thresholding(X, mode='5', min_v=0.01, max_v=0.09, make_plot=False,
                 ax=None, label=''):
    """
    Params
    ------

    X: numpy.array, shape=(n,n)
    mode: string, optional
        The way of thresholding such matrix
        - '1' the 1% of the element of each row is taken
        - '5' the 5% of the element of each row is taken
        - 'global' the 75% of the elements of all the matrix are taken according
            to their decreasing order
        - 'cl_coeff' the threshold is selected comparing the
          clustering coefficient with the one of a random graph
          "LEAL, Luis Guillermo; LOPEZ, Camilo; LOPEZ-KLEINE, Liliana.
           Construction and comparison of gene co-expression networks shows
           complex plant immune responses. PeerJ, 2014, 2: e610."
    """

    X = check_array(X)
    n = X.shape[0]
    X_new = X.copy()
    mode = str(mode).lower()

    if mode == '1' or mode == '5':
        how_many = int(round(int(mode)*n/100))
        print(how_many)
        indices = np.argsort(X, axis=1)
        to_discard = indices[:, 0:-how_many]
        for r in range(X.shape[0]):
            X_new[r, to_discard[r]] = 0
        return X_new

    if mode == 'global':
        indices = np.unravel_index(np.argsort(X, axis=None), X.shape)
        how_many = int(round(75/100*X.size))
        indices =(indices[0][0:-how_many], indices[1][0:-how_many])
        X_new[indices] = 0
        return X_new

    if mode=='cl_coeff':
        with warnings.catch_warnings(RuntimeWarning):
            warnings.simplefilter("ignore")
            if np.max(X)>1:
                X_new = X_new - np.min(X_new)
                X_new *= 1/np.max(X)

            prev_diff = -5
            diffs = []
            value = -1
            result = None
            found = False
            for v in np.arange(min_v, max_v, 0.01):
                X_old = X_new.copy()
                X_new[np.where(X_new<v)] = 0
                X_thr = X_new.copy()
                X_thr = (X_thr != 0).astype(np.int)
                np.fill_diagonal(X_thr, 0)
                C_v = clustering_coefficient(X_thr)

                N = X_new.shape[0]#np.sum(degrees!=0)
                k_bar = np.sum(degrees)/N
                k_d = np.sum(degrees**2)/N
                C_r_v = (k_d - k_bar)**2/(k_bar**3 *N)
                #print("Clustering coefficient %.4f, random clustering coefficient %.4f " % (C_v, C_r_v))
                diff = C_v - C_r_v
                diffs.append(diff)
                if np.abs(diff) < prev_diff and not found:



                    value = v - 0.01
                    result = X_old
                    found = True
                prev_diff = np.abs(diff)

            if make_plot:
                if ax is None:
                    fig, ax = plt.figure(figsize=(5,5))
                ax.plot(np.arange(0, len(diffs)), diffs, marker='o',
                       label=label)
                ax.set_xlabel(r'$\tau_v$')
                ax.set_ylabel(r' $|C(\tau_v) - C_r(\tau_v)|$ ')
                #plt.xlim(0.01, 0.99)
                #plt.xticks(np.arange(0, len(diffs)), (np.arange(0.01, 0.99, 0.01))
            #print("Thresholding value %.2f"%value)
            return result


def thresholding_generating_graphs(X, min_v=0.01, max_v=0.99, make_plot=False,
                 ax=None, label='', n_repetitions=10):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        X_new = X - np.min(X)
        X_new *= 1/np.max(X)

        mean_diffs = []
        std_diffs = []
        for v in np.arange(min_v, max_v, 0.01):
            print("Threshold ", v)
            X_old = X_new.copy()
            X_new[np.where(X_new<v)] = 0
            X_thr = X_new.copy()
            X_thr = (X_thr != 0).astype(np.int)
            np.fill_diagonal(X_thr, 0)

            n = X.shape[0]
            m = np.sum(X_thr)/2

            diffs = []
            for rep in range(n_repetitions):
                random_graph = erdos_renyi(n, m)
                C_v = clustering_coefficient(X_thr)
                C_r_v = clustering_coefficient(random_graph)

                diff = C_v - C_r_v
                diffs.append(diff)
                print("Done repetition ", rep)

                mean_diffs.append(np.mean(diffs))
                std_diffs.append(np.std(diffs))

        mean_diffs = np.array(mean_diffs)
        std_diffs = np.array(std_diffs)
        if make_plot:
            if ax is None:
                fig, ax = plt.figure(figsize=(5,5))
            ax.fill_between(mean_diffs, mean_diffs-std_diffs, mean_diffs + std_diffs,
                   label=label)
            ax.set_xlabel(r'$\tau_v$')
            ax.set_ylabel(r' $|C(\tau_v) - C_r(\tau_v)|$ ')
            #plt.xlim(0.01, 0.99)
            #plt.xticks(np.arange(0, len(diffs)), (np.arange(0.01, 0.99, 0.01))
        #print("Thresholding value %.2f"%value)
        return mean_diffs, std_diffs
