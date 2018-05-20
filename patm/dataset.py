import os
import cPickle as pickle

split_tags = ('train', 'dev', 'test')


class Dataset(object):
    def __init__(self, name, _id, collection_length):
        self.name = name
        self.id = _id
        self.col_len = collection_length
        self.splits = None
        self.datapoints = dict([(tag, {}) for tag in split_tags])

    def add(self, other):
        pass
        # doc_id >= 1
        # datapoint[doc_id] = {class:'ney-york-times'}

    def set_splits(self, splits):
        """
        :param splits: possible keys: {train, dev, test}
        :type splits: dict
        :return:
        """
        assert all(splits.keys()) in split_tags
        assert sum(splits.values()) == 1
        self.splits = splits


class UciDataset(object):

    def __init__(self, name, _id, weights, words):
        """
        :param ids_n_weights: absolute path to the uci formated file that holds (doc_id, token id, weight) triplets
        :type ids_n_weights: str
        :param words: absolute path to the uci formated file that holds a word per line (no duplicates of course)
        :param words: str
        """
        self.name = name
        self.id = _id
        self.bowf = weights
        self.words = words
        assert os.path.isfile(self.bowf)
        assert os.path.isfile(self.words)
        assert os.path.dirname(self.bowf) == os.path.dirname(self.words)
        self.root_dir = os.path.dirname(self.bowf)

    def __str__(self):
        b = 'Uci Dataset\nid: {}\nname: {}\nroot-dir: {}\nuci-docwords: {}\nvocab: {}'.format(self.id, self.name, self.root_dir, os.path.basename(self.bowf), os.path.basename(self.words))
        return b

    def save(self):
        name = self.id + '.pkl'
        try:
            with open(os.path.join(self.root_dir, name), 'wb') as pickled_dataset:
                pickle.dump(self, pickled_dataset, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved dataset in '{}' as '{}'".format(self.root_dir, name))
        except RuntimeError as e:
            print(e)
            print("Failed to save dataset wtih id '{}'".format(weedataset_id))




def construct_dataset(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
        assert isinstance(weedataset, UciDataset)
    return dataset
