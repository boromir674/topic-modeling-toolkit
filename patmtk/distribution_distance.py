import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm

# def JSD(P, Q):
#     _P, _Q = P / norm(P, ord=1), Q / norm(Q, ord=1)
#     _M = 0.5 * (_P + _Q)
#     return (entropy(_P, _M) + entropy(_Q, _M)) / 2.
#
#
# def jsd(p, q, base=np.e):
#     '''
#         Implementation of pairwise `jsd` based on
#         https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
#     '''
#     p, q = np.asarray(p), np.asarray(q)
#     p, q = p/p.sum(), q/q.sum()
#     m = 1./2*(p + q)
#     return entropy(p,m, base=base)/2. +  entropy(q, m, base=base)/2.

def jsdiv(P, Q):
    _P, _Q = tuple((_norm(_) for _ in [P, Q]))
    _M = 0.5 * (_P + _Q)
    return (entropy(_P, _M) + entropy(_Q, _M)) / 2.

def _norm(v):
    if unormalized(v):
        return v / norm(v, ord=1)
    return v


def unormalized(l):
    s = 0
    for i in l:
        s += i
        if s - 1 > 1e-4:
            return True
    return False


class DivergenceComputer:

    def from_model(self, artm_model, class1, class2, topic_names):
        """

        :param artm.ARTM artm_model:
        :param class1:
        :param class2:
        :param str or list topic_names:
        :return:
        """
        if type(topic_names) == str:
            topic_names = [topic_names]
        artm_model.get_phi(topic_names=topic_names, class_ids=[class1, class2], )
        return JSD([], [])

    def from_results(self, results_file, class1, class2, topic_name):
        return self.from_model(results_file, class1, class2, topic_name)


