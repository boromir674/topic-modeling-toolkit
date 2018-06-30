import os
import cPickle as pickle
from patm.definitions import collections_dir

split_tags = ('train', 'dev', 'test')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# print bcolors.HEADER + "Warning: No active frommets remain. Continue?"  #

class TextDataset(object):
    def __init__(self, name, _id, nb_docs, unique_words, nb_words, weights_file, words_file, vowpal_file):
        """
        :param str name: a name corresponding to the collection the dataset encodes/belongs to
        :param str _id: a unique identifier based on the prerpocessing steps executed to create this dataset
        :param int nb_docs:
        :param int unique_words:
        :param int nb_words:
        :param str weights_file: full path to a uci docwords (Bag of Words) file: */docword.name.txt
        :param str words_file: full path to the uci vocabulary file: */vocab.name.txt
        :param str vowpal_file: full path to a Vowpal formatted (Bag of Words) file: */vowpal.name.txt
        """
        self.name = name
        self.id = _id
        self._col_len = nb_docs
        self._unique_words = unique_words
        self._nb_bows = nb_words
        self.bowf = weights_file
        self.words = words_file
        self.root_dir = os.path.join(collections_dir, self.name)
        self.vowpal = vowpal_file
        assert os.path.isdir(self.root_dir)
        assert os.path.isfile(self.bowf) and os.path.isfile(self.words)
        assert os.path.dirname(self.bowf) == os.path.dirname(self.words) == self.root_dir
        assert os.path.isfile(self.vowpal)
        assert os.path.dirname(self.vowpal) == self.root_dir
        self.splits = None
        self.datapoints = dict([(tag, {}) for tag in split_tags])
# print bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC
    def __str__(self):
        return bcolors.OKBLUE + '{}:{} {}\ndocs: {}\nunique: {}\nbows: {}'.format(self.name, bcolors.ENDC, self.id, self._col_len, self._unique_words, self._nb_bows)

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
            dataset.path = dataset_path
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
