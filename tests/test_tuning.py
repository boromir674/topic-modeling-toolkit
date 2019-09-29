import pytest
import os
from collections import Counter
from functools import reduce
from topic_modeling_toolkit.patm.tuning.tuner import ParametersMixture, LabelingDefinition, RegularizationSpecifications


def test_tuning(tuner_obj, regularizers_specs, expected_constant_params, expected_explorable_params, expected_labeling_parameters, model_names):
    assert sorted(list(tuner_obj.constants)) == sorted([x[0] for x in expected_constant_params])
    assert sorted(tuner_obj.explorables) == sorted([x[0] for x in expected_explorable_params])
    assert sorted(list(tuner_obj.regularization_specs.types)) == sorted([x[0] for x in regularizers_specs])
    assert sorted(tuner_obj._labeler.parameters) == sorted(expected_labeling_parameters)
    assert tuner_obj.regularization_specs == RegularizationSpecifications([
        ('label-regularization-phi-dom-cls', [('tau', 1e5)]),
        ('decorrelate-phi-dom-def', [('tau', 1e4)])
        ])


    for m in model_names:
        assert os.path.isfile(os.path.join(tuner_obj.dataset, 'results', '{}.json'.format(m)))
        assert os.path.isfile(os.path.join(tuner_obj.dataset, 'models', '{}.phi'.format(m)))


def test_parameter_mixture():
    pm = ParametersMixture([('gg', [1,2,3]), ('a', 1), ('b', [1]), ('c', ['e', 'r'])])
    assert pm.steady == ['a', 'b']
    assert pm.explorable == ['gg', 'c']


# def test_labeling():
#     lb = LabelingDefinition()