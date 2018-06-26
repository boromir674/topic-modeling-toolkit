import os
import cPickle as pickle

split_tags = ('train', 'dev', 'test')


class Dataset(object):
    def __init__(self, data_format, name, _id, unique_words=None, nb_words=None):
        """

        :param name:
        :param _id:
        :param collection_length:
        :param unique_words:
        :param nb_words:
        """
        self._type = data_format
        self.name = name
        self.id = _id
        self.col_len = collection_length
        self.splits = None
        self.datapoints = dict([(tag, {}) for tag in split_tags])
        self._unique_words = unique_words
        self._nb_bows = nb_words

    def __str__(self):
        return '{}: {}\ntype: {}\nunique: {}\nbows: {}'.format(self.name, self.id, self._type, self._unique_words, self._nb_bows)

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

    # def set_splits(self, splits):
    #     """
    #     :param splits: possible keys: {train, dev, test}
    #     :type splits: dict
    #     :return:
    #     """
    #     assert all(splits.keys()) in split_tags
    #     assert sum(splits.values()) == 1
    #     self.splits = splits


class UciDataset(Dataset):
    format = 'uci'
    def __init__(self, name, _id, weights, words):
        """

        :param str name: a name corresponding to the collection the dataset encodes/belongs to
        :param str _id: a unique identifier based on the prerpocessing steps executed to create this dataset
        :param str weights: full path to a uci docwords (Bag of Words) file: */docword.name.txt
        :param str words: full path to the uci vocabulary file: */vocab.name.txt
        """
        super(UciDataset, self).__init__(self.format, name, _id)
        self.bowf = weights
        self.words = words
        assert os.path.isfile(self.bowf)
        assert os.path.isfile(self.words)
        assert os.path.dirname(self.bowf) == os.path.dirname(self.words)
        self.root_dir = os.path.dirname(self.bowf)
        assert os.path.basename(self.root_dir) == name

    def save(self):
        name = self.id + '.pkl'
        try:
            with open(os.path.join(self.root_dir, name), 'wb') as pickled_dataset:
                pickle.dump(self, pickled_dataset, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved dataset in '{}' as '{}'".format(self.root_dir, name))
        except RuntimeError as e:
            print(e)
            print("Failed to save dataset wtih id '{}'".format(weedataset_id))
