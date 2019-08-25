import pytest
import os
from collections import Counter
from functools import reduce


def test_tuning(tuner_obj, regularizers_specs, expected_constant_params, expected_explorable_params, expected_labeling_parameters, model_names):
    assert sorted(tuner_obj.constants) == sorted(list(expected_constant_params.keys()))
    assert sorted(tuner_obj.explorables) == sorted(list(expected_explorable_params.keys()))
    assert sorted(list(tuner_obj.regularization_specs.keys())) == sorted(list(regularizers_specs.keys()))
    assert tuner_obj._labeling_params == expected_labeling_parameters
    for m in model_names:
        assert os.path.isfile(os.path.join(tuner_obj._dir, 'results', '{}.json'.format(m)))
        assert os.path.isfile(os.path.join(tuner_obj._dir, 'models', '{}.phi'.format(m)))
