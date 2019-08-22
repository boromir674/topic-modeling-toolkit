import pytest
from reporting.psi import PsiReporter


@pytest.fixture(scope='module')
def pr(megadata_dir):
    pr = PsiReporter()
    pr.dataset = megadata_dir
    return pr


@pytest.fixture(scope='module', params=[3])
def datasets(megadata_dir, request):
    """Dataset and models trained on it parametrized on the number of document classes defined"""
    return {3: {'dataset': megadata_dir,
                'models': [{
                    'string': 'liberal_Class           70.8 71.9\n'
                              'centre_Class       70.8      65.1\n'
                              'conservative_Class 71.9 65.1     \n',
                    'label': 'sgo_1.phi',
                    'data': [[0, 70.8, 71.9],
                             [70.8, 0, 65.1],
                             [71.9, 65.1, 0]]},
                    # {'string': '',
                    #  'label': 'candidate',
                    #  'data': [[]]}
                ]
            }}[request.param]


#
# @pytest.fixture(scope='module', params=[3])
# def best_models(request):
#     """Models achieving best separation between the p(c|t) distributions and their exected Psi-matrix related results"""
#     return {3: [{'string': 'liberal_Class           70.8 71.9\n'
#                           'centre_Class       70.8      65.1\n'
#                           'conservative_Class 71.9 65.1     \n',
#                 'label': 'sgo_1.phi',
#                 'data': [[0, 70.8, 71.9],
#                          [70.8, 0, 65.1],
#                          [71.9, 65.1, 0]]},
#                 # {'string': '',
#                 #  'label': 'candidate',
#                 #  'data': [[]]}
#                 ],
#             }[request.param]
# #
# # @pytest.fixture(scope='module')
# # def
#
# def sane_separation(data):
# [[float('{:.1f}'.format(y)) for y in x] for x in pr.values([request.param], topics_set='domain')[0]]

def test_sanity(pr, datasets, regression_model_path):
    pr.dataset = datasets['dataset']
    for d in datasets['models']:
        array = [[float('{:.1f}'.format(k)) for k in z] for z in pr.values([d['label']], topics_set='domain')[0]]
        for i, row in enumerate(array):
            assert sorted(row[:i], reverse=True) == row[:i]
            assert row[i] == 0
            assert sorted(row[i:], reverse=False) == row[i:]


# @pytest.mark.parametrize("model_phi_path", [
#     'sgo_1.phi',
#     'candidate.phi'
# ])
def test_divergence_distances_computer(pr, regression_model_path, datasets):
    pr.dataset = datasets['dataset']
    for d in datasets['models']:
        b = pr.pformat([d['label']], topics_set='domain', show_class_names=True)
        assert b == d['string']

        lll = pr.values([d['label']], topics_set='domain')[0]
        assert [[float('{:.1f}'.format(y)) for y in x] for x in lll] == d['data']


#
#
#
# #####################
# @pytest.fixture(scope='module', params=['sgo_1.phi', 'candidate.phi'])
# def candidate_model(pr, regression_model_path, request):
#     """Use to develop amodel asserting improvement in the political spectrum topology (separation resulting of computed p(c|t) \forall c \in C, distances using the symmetric KL divergence"""
#     return [[float('{:.1f}'.format(y)) for y in x] for x in pr.values([request.param], topics_set='domain')[0]]
#
#
# # T.O.D.O. implement a function that takes a KL distances matrix (see 'best_seperation_performance' fixture) and return a quantitative measure of quality
# # def fitness(data):
# #     pass
#
#
#
# def test_pct_distributions_KL_distances_regression(candidate_model):
#     sane_separation(candidate_model)
#
