
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


@pytest.fixture(scope='session')
def collections_root_dir(tmpdir):
    return str(tmpdir.mkdir(TEST_COLLECTIONS_ROOT_DIR_NAME))


@pytest.fixture(scope='session', params=[[100, 100]])
def sample_n_real():
    return request.param


@pytest.fixture(scope='session')
def pipe_n_quantities(sample_n_real):
    return [TEST_PIPELINE_CFG] + sample_n_real + [1297, 833, 834, 771]


@pytest.fixture(scope='session')
def test_dataset(collections_root_dir, pipe_n_quantities):
    """A dataset ready to be used for topic modeling training. Depends on the input document sample size to take and resulting actual size"""
    pipe_handler = PipeHandler(collections_root_dir, 'posts', pipe_n_quantities[1])
    pipe_handler.pipeline = Pipeline.from_cfg(pipe_n_quantities[0])
    return pipe_handler.preprocess(TEST_COLLECTION, add_class_labels_to_vocab=True)


@pytest.fixture(scope='module')
def coherence_builder(collections_root_dir):
    return CoherenceFilesBuilder(os.path.join(collections_root_dir, TEST_COLLECTION))



class DatasetTransformationTester(object):

    #
    # @pytest.mark.parametrize("pipeline_cfg, sample, nb_docs, nb_bows, unique, vocab_file_len", [
    #     (TEST_PIPELINE_CFG, 100, 100, 1297, 833, 834)
    # ])
    def test_transform(self, pipe_n_quantities, test_dataset, coherence_builder):
        coherence_builder.create_files(cooc_window=10, min_tf=0, min_df=0, apply_zero_index=False)

        assert test_dataset.name == TEST_COLLECTION
        assert test_dataset.id == self._id(pipe_n_quantities[2], pipe_n_quantities[0])
        assert test_dataset._col_len == pipe_n_quantities[2]
        assert test_dataset.nb_bows == pipe_n_quantities[3]
        assert test_dataset.unique == pipe_n_quantities[4]

        assert self.file_len(test_dataset.bowf) == pipe_n_quantities[3] + 3
        assert self.file_len(test_dataset.words) == pipe_n_quantities[5]  # this hack is used because the currently used sample (which takes the first 500 documents) of 500 docs includes only one label
        assert self.file_len(test_dataset.vowpal) == pipe_n_quantities[2]

        assert self.file_len(self._file('cooc_0_tf.txt')) == pipe_n_quantities[6]
        assert self.file_len(self._file('cooc_0_df.txt')) == pipe_n_quantities[6]
        assert self.file_len(self._file('ppmi_0_tf.txt')) == pipe_n_quantities[6]
        assert self.file_len(self._file('ppmi_0_df.txt')) == pipe_n_quantities[6]
        assert os.path.isfile(self._file('{}.pkl'.format(test_dataset.id)))

    def _formatter(self, pipe_settings):
        return lambda x: '{}-{}'.format(x, pipe_settings[x] if x in ['lowercase', 'monospace', 'unicode',
                                                                     'deaccent'] else x)

    def _id(self, full_docs, cfg):
        config = ConfigParser.ConfigParser()
        config.read(cfg)
        d = OrderedDict([(var, value) for var, value in config.read('preprocessing').items()])
        formatter = self._formatter(d)
        b = '{}_{}'.format(full_docs, '_'.join(formatter(x) for i, x in enumerate(d.items()) if i < len(d) - 1))
        b += '_format1-uc1.format2-vowpal'
        return b

    @staticmethod
    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    @staticmethod
    def _file(name):
        return os.path.join(TEST_COLLECTIONS_ROOT_DIR_NAME, TEST_COLLECTION, name)
