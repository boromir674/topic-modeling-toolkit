import os
import pytest


DEFAULT_MODALITY = '@default_class'
IDEOLOGY_MODALITY = '@labels_class'


def model_properties(model):
    """
    :param patm.modeling.topic_model.TopicModel model:
    :return:
    """
    return (len(model.topic_names),
            model.document_passes,
            float(len(model.background_topics)) / len(model.topic_names),
            model.modalities_dictionary[DEFAULT_MODALITY],
            model.modalities_dictionary[IDEOLOGY_MODALITY],
            {long_type: model._reg_longtype2name[long_type] for long_type in model.regularizer_unique_types},
            model.definition2evaluator_name
    )


class TestTrain(object):

    def test_trained_model(self, trained_model_n_experiment, train_settings):
        assert model_properties(trained_model_n_experiment[0]) == (
            int(train_settings['learning']['nb_topics']),
            int(train_settings['learning']['document_passes']),
            float(train_settings['information']['background-topics-percentage']),
            float(train_settings['information']['default-class-weight']),
            float(train_settings['information']['ideology-class-weight']),
            train_settings['regularizers'],
            train_settings['scores']
        )
        assert 'sparsity-phi-@ic' in trained_model_n_experiment[1].trackables

    def test_model_loading(self, trained_model_n_experiment, train_settings):
        model, experiment = trained_model_n_experiment
