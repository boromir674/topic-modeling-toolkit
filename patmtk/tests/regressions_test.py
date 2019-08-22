import pytest
from reporting.psi import PsiReporter
from patm.modeling import Experiment
from patm.modeling.trainer import TrainerFactory

@pytest.fixture(scope='module')
def pr(megadata_dir):
    pr = PsiReporter()
    pr.dataset = megadata_dir
    return pr


@pytest.fixture(scope='module', params=[3])
def datasets(megadata_dir, pr, request):
    """Dataset and models trained on it parametrized on the number of document classes defined"""
    data = {3: {'dataset': megadata_dir,
                'models': [
                    {
                        'label': 'sgo_1.phi',
                        'expected-string': 'liberal_Class           70.8 71.9\n'
                                           'centre_Class       70.8      65.1\n'
                                           'conservative_Class 71.9 65.1     \n',
                        'expected-matrix': [[0, 70.8, 71.9],
                                            [70.8, 0, 65.1],
                                            [71.9, 65.1, 0]]
                    },
                    # {
                        # 'label': 'candidate.phi',
                        # 'expected-string': '',
                        # 'expected-matrix': [[]]
                     # }
                ]
            }}
    subdict = data[request.param]
    for model_data in subdict['models']:
        model_data['reported-distances'] = [[float('{:.1f}'.format(k)) for k in z] for z in pr.values([model_data['label']], topics_set='domain')[0]]
        model_data['reported-string'] = pr.pformat([model_data['label']], topics_set='domain')
    return subdict


def test_sanity(pr, datasets):
    pr.dataset = datasets['dataset']
    for d in datasets['models']:
        # array = [[float('{:.1f}'.format(k)) for k in z] for z in pr.values([d['label']], topics_set='domain')[0]]
        for i, row in enumerate(d['reported-distances']):
            assert row[:i] == sorted(row[:i], reverse=True)
            assert row[i] == 0
            assert row[i:] == sorted(row[i:], reverse=False)

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
@pytest.fixture(scope='module',)
def known_expected(datasets):
    d = datasets.copy()
    d['models'] = [x for x in d['models'] if 'expected-string' in x and 'expected-matrix' in x]
    return d


def test_divergence_distances_computer(pr, known_expected): # , regression_model_path):
    pr.dataset = known_expected['dataset']
    for d in known_expected['models']:
        b = pr.pformat([d['label']], topics_set='domain', show_class_names=True)
        assert d['reported-string'] == d['expected-string']
        assert d['reported-distances'] == d['expected-matrix']


# def test_loading(datasets, megadata_dir):
#     new_exp_obj = Experiment(megadata_dir)
#     trainer = TrainerFactory().create_trainer(megadata_dir)
#     trainer.register(new_exp_obj)
#     for d in datasets['models']:
#         loaded_model = new_exp_obj.load_experiment(d['label'].replace('.phi', ''))
#
#         assert loaded_model.regularizer_wrappers
#     return loaded_model, new_exp_obj
# @pytest.mark.parametrize("model_phi_path", [
#     'sgo_1.phi',
#     'candidate.phi'
# ])


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
