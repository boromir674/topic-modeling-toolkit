
import os
import pytest

from processors import PipeHandler, Pipeline
from patm.build_coherence import CoherenceFilesBuilder

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PIPELINE_CFG = os.path.join(MODULE_DIR, 'test-pipeline.cfg')
TEST_COLLECTIONS_ROOT_DIR_NAME = 'unittests-collections'

TEST_COLLECTION = 'unittest-dataset'


@pytest.fixture(scope='module')
def collections_root_dir(tmpdir):
    return str(tmpdir.mkdir(TEST_COLLECTIONS_ROOT_DIR_NAME))


@pytest.fixture(scope='module')
def pipe_handler(collections_root_dir):
    return PipeHandler(collections_root_dir, 'posts', sample=500)


@pytest.fixture(scope='module')
def preprocess_pipe():
    return Pipeline.from_cfg(TEST_PIPELINE_CFG)


@pytest.fixture(scope='module')
def coherence_builder(collections_root_dir):
    return CoherenceFilesBuilder(os.path.join(collections_root_dir, TEST_COLLECTION))


class DatasetTransformationTester(object):

    def test_transform(self, pipe_handler, preprocess_pipe, coherence_builder):
        pipe_handler.pipeline = preprocess_pipe
        test_dataset = pipe_handler.preprocess(TEST_COLLECTION, add_class_labels_to_vocab=True)
        coherence_builder.create_files(cooc_window=10, min_tf=0, min_df=0, apply_zero_index=False)

        assert test_dataset.name == TEST_COLLECTION
        assert test_dataset.id == '100_lowercase_monospace_unicode_deaccent_normalize-lemmatize_minlength-2_maxlength-25_ngrams-1_nobelow-1_noabove-0.5_weight-counts_format1-uci.format2-vowpal'
        assert test_dataset._col_len == 100
        assert test_dataset.nb_bows == 1297
        assert test_dataset.unique == 833



        # self.bowf = weights_file
        # self.words = words_file
        # self.vowpal = vowpal_file
