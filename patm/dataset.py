import os
import cPickle as pickle
from patm.definitions import collections_dir

split_tags = ('train', 'dev', 'test')


class TextDataset(object):
    def __init__(self, data_format, name, _id, nb_docs, unique_words, nb_words):
        """
        :param str data_format:
        :param str name:
        :param str _id:
        :param int nb_docs:
        :param int unique_words:
        :param int nb_words:
        """
        self._type = data_format
        self.name = name
        self.id = _id
        self._col_len = nb_docs
        self._unique_words = unique_words
        self._nb_bows = nb_words
        self.root_dir = os.path.join(collections_dir, self.name)
        assert os.path.isdir(self.root_dir)
        self.splits = None
        self.datapoints = dict([(tag, {}) for tag in split_tags])

    def __str__(self):
        return '{}: {}\ntype: {}, docs: {}\nunique: {}\nbows: {}'.format(self.name, self.id, self._type, self._col_len, self._unique_words, self._nb_bows)

    @property
    def unigue(self):
        return self._unique_words

    @property
    def nb_bows(self):
        return self._nb_bows

    @staticmethod
    def load(dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def save(self):
        name = self.id + '.pkl'
        try:
            with open(os.path.join(self.root_dir, name), 'wb') as pickled_dataset:
                pickle.dump(self, pickled_dataset, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved dataset in '{}' as '{}'".format(self.root_dir, name))
        except RuntimeError as e:
            print(e)
            print("Failed to save dataset wtih id '{}'".format(self.id))

class UciDataset(TextDataset):
    format = 'uci'
    def __init__(self, name, _id, nb_docs, unique_words, nb_words, weights_file, words_file):
        """
        :param str name: a name corresponding to the collection the dataset encodes/belongs to
        :param str _id: a unique identifier based on the prerpocessing steps executed to create this dataset
        :param int nb_docs:
        :param int unique_words:
        :param int nb_words:
        :param str weights_file: full path to a uci docwords (Bag of Words) file: */docword.name.txt
        :param str words_file: full path to the uci vocabulary file: */vocab.name.txt
        """
        super(UciDataset, self).__init__(self.format, name, _id, nb_docs, unique_words, nb_words)
        self.bowf = weights_file
        self.words = words_file
        assert os.path.isfile(self.bowf) and os.path.isfile(self.words)
        assert os.path.dirname(self.bowf) == os.path.dirname(self.words) == self.root_dir

class VowpalDataset(TextDataset):
    format = 'vowpal'
    def __init__(self, name, _id, nb_docs, unique_words, nb_words, vowpal_file):
        super(VowpalDataset, self).__init__(self.format, name, _id, nb_docs, unique_words, nb_words)
        self.vowpal = vowpal_file
        assert os.path.isfile(self.vowpal)
        assert os.path.dirname(self.vowpal) == self.root_dir
