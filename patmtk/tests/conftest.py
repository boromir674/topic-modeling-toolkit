

import os
import ConfigParser
import pytest

from processors import PipeHandler, Pipeline


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PIPELINE_CFG = os.path.join(MODULE_DIR, 'test-pipeline.cfg')
TEST_COLLECTIONS_ROOT_DIR_NAME = 'unittests-collections'

TEST_COLLECTION = 'unittest-dataset'

TRAIN_CFG = os.path.join(MODULE_DIR, 'test-train.cfg')
REGS_CFG = os.path.join(MODULE_DIR, 'test-regularizers.cfg')
MODEL_1_LABEL = 'test-model-1'



@pytest.fixture(scope='session')
def collections_root_dir(tmpdir_factory):
    return  str(tmpdir_factory.mktemp(TEST_COLLECTIONS_ROOT_DIR_NAME))
    # img.save(str(fn))
    # return fn
    #
    # return str(tmpdir.mkdir(TEST_COLLECTIONS_ROOT_DIR_NAME))

@pytest.fixture(scope='session')
def test_collection_name():
    return TEST_COLLECTION

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
    return pipe_handler.preprocess(TEST_COLLECTION, add_class_labels_to_vocab=True)



# PARSE UNITTEST CFG FILES

def parse_cfg(cfg):
    config = ConfigParser.ConfigParser()
    config.read(cfg)
    return {section: dict(config.items(section)) for section in config.sections()}


@pytest.fixture(scope='session')
def train_settings():
    return parse_cfg(TRAIN_CFG)

@pytest.fixture(scope='session')
def reg_settings():
    return parse_cfg(REGS_CFG)
#
#
# ## TRAIN DECOMPOSE
# @pytest.fixture(scope='session')
# def learning_settings(train_settings):
#     return train_settings.section('learning')
#
# @pytest.fixture(scope='session')
# def information_settings(train_settings):
#     return train_settings.section('information')
#
#
# ### REGULARIZERS
# @pytest.fixture(scope='session')
# def regularizer_settings(train_settings):
#     return train_settings.section('regularizers')
#
# @pytest.fixture(scope='session')
# def regularizer_names(regularizer_settings):
#     return sorted(v for v in regularizer_settings.values() if v)
#
# @pytest.fixture(scope='session')
# def regularizer_types(regularizer_settings):
#     return sorted(k for k,v in regularizer_settings.items() if v)
#
#
# ### SCORES
# @pytest.fixture(scope='session')
# def score_settings(train_settings):
#     return train_settings.section('scores')
#
# @pytest.fixture(scope='session')
# def scorer_names(score_settings):
#     return sorted(v for v in score_settings.values() if v)
#
# @pytest.fixture(scope='session')
# def scorer_types(score_settings):
#     return sorted(k for k,v in score_settings.items() if v)
