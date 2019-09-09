import os
import pytest
from topic_modeling_toolkit.reporting import PsiReporter
from topic_modeling_toolkit.patm import Experiment, TrainerFactory


@pytest.fixture(scope='session')
def megadata_dir(unittests_data_dir):
    return os.path.join(unittests_data_dir, 'megadata')


@pytest.fixture(scope='session')
def regression_model_path(megadata_dir):
    trainer = TrainerFactory().create_trainer(megadata_dir, exploit_ideology_labels=True, force_new_batches=False)
    experiment = Experiment(megadata_dir)
    topic_model = trainer.model_factory.create_model('candidate', os.path.join(MODULE_DIR, 'regression.cfg'), reg_cfg=os.path.join(MODULE_DIR, 'test-regularizers.cfg'),
                                                     show_progress_bars=False)
    train_specs = trainer.model_factory.create_train_specs()
    trainer.register(experiment)
    experiment.init_empty_trackables(topic_model)
    trainer.train(topic_model, train_specs, effects=False, cache_theta=True)
    experiment.save_experiment(save_phi=True)
    # CHANGE THIS SO THAT YOU DO NOT OVERRIDE the previously persisted model phi matrix object and results json
    return 'candidate.phi'


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
                    {
                        'label': 'candidate.phi',
                        'expected-string': 'liberal_Class           34.9 35.0\n'
                                           'centre_Class       34.9      29.2\n'
                                           'conservative_Class 35.0 29.2     \n',
                        'expected-matrix': [[0, 34.9, 35],
                                            [34.9, 0, 29.2],
                                            [35, 29.2, 0]]
                     }
                ]
            }}
    subdict = data[request.param]
    for model_data in subdict['models']:
        model_data['reported-distances'] = [[float('{:.1f}'.format(k)) for k in z] for z in pr.values([model_data['label']], topics_set='domain')[0]]
        model_data['reported-string'] = pr.pformat([model_data['label']], topics_set='domain', show_model_name=False, show_class_names=True)
    return subdict

# Insert fixture 'regression_model_path', which is the path to a newly trained model, with settings in regression.cfg, to test if it places the ideology classes relatively in correct ordering on the political spectrum
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


# # T.O.D.O. implement a function that takes a KL distances matrix (see 'best_seperation_performance' fixture) and return a quantitative measure of quality
# # def fitness(data):
# #     pass
