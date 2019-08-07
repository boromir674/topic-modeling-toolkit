
import os
import ConfigParser

import pytest

from patm.modeling import TrainerFactory, Experiment




MODULE_DIR = os.path.dirname(os.path.realpath(__file__))


TRAIN_CFG = os.path.join(MODULE_DIR, 'test-train.cfg')
REGS_CFG = os.path.join(MODULE_DIR, 'test-regularizers.cfg')
MODEL_1_LABEL = 'test-model-1'

DEFAULT_MODALITY = '@default_class'
IDEOLOGY_MODALITY = '@labels_class'


@pytest.fixture(scope='session')
def test_model(collections_root_dir, test_collection_name):
    model_trainer = TrainerFactory(collections_root_dir=collections_root_dir).create_trainer(test_collection_name, exploit_ideology_labels=True, force_new_batches=True)
    experiment = Experiment(os.path.join(collections_root_dir, test_collection_name), model_trainer.cooc_dicts)
    model_trainer.register(experiment)  # when the model_trainer trains, the experiment object keeps track of evaluation metrics

    topic_model = model_trainer.model_factory.create_model(MODEL_1_LABEL, TRAIN_CFG, reg_cfg=REGS_CFG, show_progress_bars=False)
    train_specs = model_trainer.model_factory.create_train_specs()
    experiment.init_empty_trackables(topic_model)
    model_trainer.train(topic_model, train_specs, effects=False, cache_theta=True)

    return topic_model

    # if args.save:
    #     experiment.save_experiment(save_phi=True)
    #     print
    #     "Saved results and model '{}'".format(args.label)
    # return str(tmpdir.mkdir(TEST_COLLECTIONS_ROOT_DIR_NAME))

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

    def test_trained_model(self, test_model, train_settings):
        assert model_properties(test_model) == (
            int(train_settings['learning']['nb_topics']),
            int(train_settings['learning']['document_passes']),
            float(train_settings['information']['background-topics-percentage']),
            float(train_settings['information']['default-class-weight']),
            float(train_settings['information']['ideology-class-weight']),
            {k: v for k, v in train_settings['regularizers'].items() if v},
            {k: v for k, v in train_settings['scores'].items() if v}
        )