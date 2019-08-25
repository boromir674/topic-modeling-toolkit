import pytest
import os
from collections import Counter
from functools import reduce


@pytest.fixture(scope='session')
def expected_labeling_parameters(tuning_parameters, expected_constant_params, expected_explorable_params):
    static_flag = {True: list(expected_constant_params.keys()),
                   False: []}
    explorable_flag = {True: list(expected_explorable_params.keys()),
                       False: []}
    return sorted(explorable_flag[tuning_parameters['append_explorables']]) + sorted(static_flag[tuning_parameters['append_static']])


@pytest.fixture(scope='session')
def model_names(tuning_parameters, training_params, expected_labeling_parameters):
    """alphabetically sorted expected model names to persist (phi and results)"""
    def _mock_label(labeling, params_data):
        inds = Counter()
        for l in labeling:
            if type(params_data[l]) != list or len(params_data[l]) == 1:
                yield params_data[l]
            else:
                inds[l] += 1
                yield params_data[l][inds[l] - 1]
    nb_models = reduce(lambda k,l: k*l, [len(x) if type(x) == list else 1 for x in training_params.values()])
    TUNE_LABEL_PREFIX = 'unittest'

    # prefix = ''
    # if tuning_parameters['prefix_label']:
    #     prefix = tuning_parameters['prefix_label'] + '_'
    prefix = TUNE_LABEL_PREFIX + '_'
    return sorted([prefix + '_'.join(str(x) for x in _mock_label(expected_labeling_parameters, training_params)) for i in range(nb_models)])


def test_tuning(tuner_obj, regularizers_specs, expected_constant_params, expected_explorable_params, expected_labeling_parameters, model_names, collections_root_dir):
    assert sorted(tuner_obj.constants) == sorted(list(expected_constant_params.keys()))
    assert sorted(tuner_obj.explorables) == sorted(list(expected_explorable_params.keys()))
    assert sorted(list(tuner_obj.active_regularizers.keys())) == sorted(list(regularizers_specs.keys()))
    assert tuner_obj._labeling_params == expected_labeling_parameters
    for m in model_names:
        assert os.path.isfile(os.path.join(tuner_obj._dir, 'results', '{}.json'.format(m)))
        assert os.path.isfile(os.path.join(tuner_obj._dir, 'models', '{}.phi'.format(m)))
