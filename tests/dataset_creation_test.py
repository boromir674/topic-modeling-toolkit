import os
import sys
from configparser import ConfigParser
from collections import OrderedDict
import pytest


@pytest.fixture(scope='module')
def tokens_data():
    return {'vowpal-1st-line':'|@default_class action aftermath call compel congress controlled craft form gop gun legislation make political shooting show sign similar spectrum subtle take violence'}


class TestDatasetTransformation(object):

    def test_transform(self, test_collection_dir, pipe_n_quantities, test_dataset, tokens_data):
        assert test_dataset.name == os.path.basename(pipe_n_quantities['unittest-collection-dir'])
        assert test_dataset.id == self._id(pipe_n_quantities['resulting-nb-docs'], pipe_n_quantities['unittest-pipeline-cfg'])
        assert test_dataset._col_len == pipe_n_quantities['resulting-nb-docs']
        assert test_dataset.nb_bows == pipe_n_quantities['nb-bows']
        assert test_dataset._unique_words == pipe_n_quantities['word-vocabulary-length']

        assert self.file_len(test_dataset.bowf) == pipe_n_quantities['nb-bows'] + 3
        assert self.file_len(test_dataset.words) == pipe_n_quantities['nb-all-modalities-terms']  # this passes because the currently used sample (which takes the first 100 documents) includes only one label
        assert self.file_len(test_dataset.vowpal) == pipe_n_quantities['resulting-nb-docs']

        assert self.file_len(os.path.join(test_dataset.root_dir, 'cooc_0_tf.txt')) in pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert self.file_len(os.path.join(test_dataset.root_dir, 'cooc_0_df.txt')) in pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert self.file_len(os.path.join(test_dataset.root_dir, 'ppmi_0_tf.txt')) in pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert self.file_len(os.path.join(test_dataset.root_dir, 'ppmi_0_df.txt')) in pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert os.path.isfile(os.path.join(test_dataset.root_dir, '{}.pkl'.format(test_dataset.id)))
        with open(os.path.join(test_collection_dir, 'vowpal.{}.txt'.format(os.path.basename(test_collection_dir))), 'r') as f:
            assert tokens_data['vowpal-1st-line'] in f.readline()

        # assert pipe_handler.dict
        assert '|@default_class sign aftermath action call gop legislation spectrum take show subtle political make gun craft violence compel form similar shooting congress controlled'
    def _formatter(self, pipe_settings):
        return lambda x: x if x in ['lowercase', 'monospace', 'unicode', 'deaccent'] else '{}-{}'.format(x, pipe_settings[x])

    def _id(self, full_docs, cfg):
        config = ConfigParser()
        config.read(cfg)
        d = OrderedDict([(var, value) for var, value in config.items('preprocessing')])
        formatter = self._formatter(d)
        b = '{}_{}'.format(full_docs, '_'.join(formatter(x) for i, x in enumerate(d.keys()) if i < len(d) - 1))
        b += '_format1-uci.format2-vowpal'
        return b

    @staticmethod
    def file_len(file_path):
        with open(file_path, 'r') as f:

            for i, l in enumerate(f):
                pass
        return i + 1


    # def test_string_piping