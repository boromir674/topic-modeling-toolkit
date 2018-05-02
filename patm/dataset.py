import os
import cPickle as pickle


class UciDataset(object):

    def __init__(self, _id, weights, words):
        """
        :param ids_n_weights: absolute path to the uci formated file that holds (doc_id, token id, weight) triplets
        :type ids_n_weights: str
        :param words: absolute path to the uci formated file that holds a word per line (no duplicates of course)
        :param words: str
        """
        self.id = _id
        self.bowf = weights
        self.words = words
        assert os.path.dirname(os.path.basename(self.bowf)) == os.path.dirname(os.path.basename(self.words))
        self.root_dir = os.path.dirname(os.path.basename(self.bowf))

    def save(self):
        name = self.id + '.pkl'
        try:
            with open(os.path.join(self.root_dir, name), 'wb') as pickled_dataset:
                pickle.dump(self, pickled_dataset, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved dataset in {} as '{}'".format(self.root_dir, name))
        except RuntimeError as e:
            print(e)
            print("Failed to save dataset wtih id '{}'".format(weedataset_id))


def construct_dataset(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
        assert isinstance(weedataset, UciDataset)
    return dataset

