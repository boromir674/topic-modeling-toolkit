

import os
import pytest

from processors import PipeHandler, Pipeline


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PIPELINE_CFG = os.path.join(MODULE_DIR, 'test-pipeline.cfg')
TEST_COLLECTIONS_ROOT_DIR_NAME = 'unittests-collections'

TEST_COLLECTION = 'unittest-dataset'


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
