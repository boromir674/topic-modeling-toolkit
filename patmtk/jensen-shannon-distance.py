import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm


def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    p, q = np.asarray(p), np.asarray(q)
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return entropy(p,m, base=base)/2. +  entropy(q, m, base=base)/2.

def _norm(v):
    if sum(v) != 1:
        return v / norm(v, ord=1)
    return v


def JSD(P, Q):
    _P, _Q = P / norm(P, ord=1), Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return (entropy(_P, _M) + entropy(_Q, _M)) / 2.


