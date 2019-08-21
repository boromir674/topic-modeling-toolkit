import pytest
from reporting.psi import PsiReporter


@pytest.fixture(scope='module')
def pr(megadata_dir):
    pr = PsiReporter()
    pr.dataset = megadata_dir
    return pr


@pytest.fixture(scope='module')
def best_seperation_performance():
    return {3: {'string': 'liberal_Class           70.8 71.9\n'
                          'centre_Class       70.8      65.1\n'
                          'conservative_Class 71.9 65.1     \n',
                'best-model': 'sgo_1.phi',
                'data': [[0, 70.8, 71.9],
                         [70.8, 0, 65.1],
                         [71.9, 65.1, 0]]},
            }


#####################
@pytest.fixture(scope='module', params=['sgo_1.phi'])
def candidate_model(pr, request):
    """Use to develop amodel asserting improvement in the political spectrum topology (separation resulting of computed p(c|t) \forall c \in C, distances using the symmetric KL divergence"""
    return [[float('{:.1f}'.format(y)) for y in x] for x in pr.values([request.param], topics_set='domain')[0]]


def test_divergence_distances_computer(pr, best_seperation_performance):
    label = best_seperation_performance[3]['best-model']
    b = pr.pformat([label], topics_set='domain', show_class_names=True)
    assert b == best_seperation_performance[3]['string']

    lll = pr.values([label], topics_set='domain')[0]
    assert [[float('{:.1f}'.format(y)) for y in x] for x in lll] == [[0, 70.8, 71.9],
                                                                     [70.8, 0, 65.1],
                                                                     [71.9, 65.1, 0]]

# T.O.D.O. implement a function that takes a KL distances matrix (see 'best_seperation_performance' fixture) and return a quantitative measure of quality
# def fitness(data):
#     pass

def sane_separation(data):
    for i, row in enumerate(data):
        assert sorted(row[:i], reverse=True) == row[:i]
        assert row[i] == 0
        assert sorted(row[i:], reverse=False) == row[i:]


def test_pct_distributions_KL_distances_regression(candidate_model):
    sane_separation(candidate_model)
