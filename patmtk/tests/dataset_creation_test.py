import os
from configparser import ConfigParser
from collections import OrderedDict
import pytest


class TestDatasetTransformation(object):

    def test_transform(self, pipe_n_quantities, test_dataset):
        assert test_dataset.name == os.path.basename(pipe_n_quantities['unittest-collection-dir'])
        assert test_dataset.id == self._id(pipe_n_quantities['resulting-nb-docs'], pipe_n_quantities['unittest-pipeline-cfg'])
        assert test_dataset._col_len == pipe_n_quantities['resulting-nb-docs']
        assert test_dataset.nb_bows == pipe_n_quantities['nb-bows']
        assert test_dataset._unique_words == pipe_n_quantities['word-vocabulary-length']

        assert self.file_len(test_dataset.bowf) == pipe_n_quantities['nb-bows'] + 3
        assert self.file_len(test_dataset.words) == pipe_n_quantities['nb-all-modalities-terms']  # this passes because the currently used sample (which takes the first 100 documents) includes only one label
        assert self.file_len(test_dataset.vowpal) == pipe_n_quantities['resulting-nb-docs']

        assert self.file_len(os.path.join(test_dataset.root_dir, 'cooc_0_tf.txt')) == pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert self.file_len(os.path.join(test_dataset.root_dir, 'cooc_0_df.txt')) == pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert self.file_len(os.path.join(test_dataset.root_dir, 'ppmi_0_tf.txt')) == pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert self.file_len(os.path.join(test_dataset.root_dir, 'ppmi_0_df.txt')) == pipe_n_quantities['nb-lines-cooc-n-ppmi-files']
        assert os.path.isfile(os.path.join(test_dataset.root_dir, '{}.pkl'.format(test_dataset.id)))

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
        with open(file_path) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
