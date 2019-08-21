import pytest
from reporting.psi import PsiReporter


@pytest.fixture(scope='module')
def pr():
    return PsiReporter()

@pytest.fixture(scope='module')
def best_seperation_performance():
    return {3: 'liberal_Class           70.8 71.9\n'
               'centre_Class       70.8      65.1\n'
               'conservative_Class 71.9 65.1     \n'}


def test_pct_distributions_KL_distances_regression(pr, megadata_dir, best_seperation_performance):
    pr.dataset = megadata_dir
    b = pr.pformat(['sgo_1.phi'], topics_set='domain', show_class_names=True)
    assert b == best_seperation_performance[3]
