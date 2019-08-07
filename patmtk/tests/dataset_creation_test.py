

import os
import ConfigParser
from collections import OrderedDict
import pytest

from processors import PipeHandler, Pipeline
from patm.build_coherence import CoherenceFilesBuilder

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PIPELINE_CFG = os.path.join(MODULE_DIR, 'test-pipeline.cfg')
TEST_COLLECTIONS_ROOT_DIR_NAME = 'unittests-collections'

TEST_COLLECTION = 'unittest-dataset'



@pytest.fixture(scope='module')
def coherence_builder(collections_root_dir):
    return CoherenceFilesBuilder(os.path.join(collections_root_dir, TEST_COLLECTION))


class TestDatasetTransformation:

    def test_transform(self, pipe_n_quantities, test_dataset, coherence_builder):
        coherence_builder.create_files(cooc_window=10, min_tf=0, min_df=0, apply_zero_index=False)
        assert test_dataset.name == TEST_COLLECTION
        assert test_dataset.id == self._id(pipe_n_quantities[2], pipe_n_quantities[0])
        assert test_dataset._col_len == pipe_n_quantities[2]
        assert test_dataset.nb_bows == pipe_n_quantities[3]
        assert test_dataset._unique_words == pipe_n_quantities[4]

        assert self.file_len(test_dataset.bowf) == pipe_n_quantities[3] + 3
        assert self.file_len(test_dataset.words) == pipe_n_quantities[5]  # this hack is used because the currently used sample (which takes the first 500 documents) of 500 docs includes only one label
        assert self.file_len(test_dataset.vowpal) == pipe_n_quantities[2]

        assert self.file_len(os.path.join(test_dataset.root_dir, 'cooc_0_tf.txt')) == pipe_n_quantities[6]
        assert self.file_len(os.path.join(test_dataset.root_dir, 'cooc_0_df.txt')) == pipe_n_quantities[6]
        assert self.file_len(os.path.join(test_dataset.root_dir, 'ppmi_0_tf.txt')) == pipe_n_quantities[6]
        assert self.file_len(os.path.join(test_dataset.root_dir, 'ppmi_0_df.txt')) == pipe_n_quantities[6]
        assert os.path.isfile(os.path.join(test_dataset.root_dir, '{}.pkl'.format(test_dataset.id)))

    def _formatter(self, pipe_settings):
        return lambda x: x if x in ['lowercase', 'monospace', 'unicode', 'deaccent'] else '{}-{}'.format(x, pipe_settings[x])

    def _id(self, full_docs, cfg):
        config = ConfigParser.ConfigParser()
        config.read(cfg)
        d = OrderedDict([(var, value) for var, value in config.items('preprocessing')])
        formatter = self._formatter(d)
        b = '{}_{}'.format(full_docs, '_'.join(formatter(x) for i, x in enumerate(d.keys()) if i < len(d) - 1))
        b += '_format1-uci.format2-vowpal'
        return b

    @staticmethod
    def file_len(file_path):
        with open(file_path) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
