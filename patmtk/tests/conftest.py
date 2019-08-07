

import os
import ConfigParser
import pytest

from processors import Pipeline, PipeHandler
from patm.build_coherence import CoherenceFilesBuilder

from patm.modeling.trainer import TrainerFactory
from patm.modeling import Experiment


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

TEST_PIPELINE_CFG = os.path.join(MODULE_DIR, 'test-pipeline.cfg')
TRAIN_CFG = os.path.join(MODULE_DIR, 'test-train.cfg')
REGS_CFG = os.path.join(MODULE_DIR, 'test-regularizers.cfg')


TEST_COLLECTIONS_ROOT_DIR_NAME = 'unittests-collections'

TEST_COLLECTION = 'unittest-dataset'
MODEL_1_LABEL = 'test-model-1'


@pytest.fixture(scope='session')
def collections_root_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp(TEST_COLLECTIONS_ROOT_DIR_NAME))

@pytest.fixture(scope='session')
def test_collection_name():
    return TEST_COLLECTION


@pytest.fixture(scope='session')
def test_collection_dir(collections_root_dir):
    return os.path.join(collections_root_dir, TEST_COLLECTION)


@pytest.fixture(scope='session', params=[[100, 100]])
def sample_n_real(request):
    return request.param


@pytest.fixture(scope='session')
def pipe_n_quantities(sample_n_real):
    return [TEST_PIPELINE_CFG] + sample_n_real + [1297, 833, 834, 771]


@pytest.fixture(scope='session')
def test_dataset(collections_root_dir, pipe_n_quantities):
    """A dataset ready to be used for topic modeling training. Depends on the input document sample size to take and resulting actual size"""
    pipe_handler = PipeHandler(collections_root_dir, 'posts', sample=pipe_n_quantities[1])
    pipe_handler.pipeline = Pipeline.from_cfg(pipe_n_quantities[0])
    text_dataset = pipe_handler.preprocess(TEST_COLLECTION, add_class_labels_to_vocab=True)
    coh_builder = CoherenceFilesBuilder(os.path.join(collections_root_dir, TEST_COLLECTION))
    coh_builder.create_files(cooc_window=10, min_tf=0, min_df=0, apply_zero_index=False)
    return text_dataset

# PARSE UNITTEST CFG FILES

def parse_cfg(cfg):
    config = ConfigParser.ConfigParser()
    config.read(cfg)
    return {section: dict(config.items(section)) for section in config.sections()}


@pytest.fixture(scope='session')
def train_settings():
    """These settings (learning, reg components, score components, etc) are used to train the model in 'trained_model' fixture. A dictionary of cfg sections mapping to dictionaries with settings names-values pairs."""
    _ = parse_cfg(TRAIN_CFG)
    _['regularizers'] = {k: v for k, v in _['regularizers'].items() if v}
    _['scores'] = {k: v for k, v in _['scores'].items() if v}
    return _


@pytest.fixture(scope='session')
def reg_settings():
    """These regularizers' initialization (eg tau coefficient value/trajectory) settings are used to train the model in 'trained_model' fixture. A dictionary of cfg sections mapping to dictionaries with settings names-values pairs."""
    return parse_cfg(REGS_CFG)


@pytest.fixture(scope='session')
def trainer_n_experiment(test_dataset, collections_root_dir):
    trainer = TrainerFactory(collections_root_dir=collections_root_dir).create_trainer(test_dataset.name, exploit_ideology_labels=True, force_new_batches=True)
    experiment = Experiment(os.path.join(collections_root_dir, test_dataset.name), trainer.cooc_dicts)
    trainer.register(experiment)  # when the model_trainer trains, the experiment object keeps track of evaluation metrics
    return trainer, experiment


@pytest.fixture(scope='session')
def trained_model_n_experiment(trainer_n_experiment):
    trainer, experiment = trainer_n_experiment
    topic_model = trainer.model_factory.create_model(MODEL_1_LABEL, TRAIN_CFG, reg_cfg=REGS_CFG, show_progress_bars=False)
    train_specs = trainer.model_factory.create_train_specs()
    experiment.init_empty_trackables(topic_model)
    trainer.train(topic_model, train_specs, effects=False, cache_theta=True)
    return topic_model, experiment
