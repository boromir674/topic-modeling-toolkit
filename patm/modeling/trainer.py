import os
import artm

from .model_factory import get_model_factory
from ..definitions import collections_dir
from ..evaluation.scorer_factory import ArtmScorerFactory


class ModelTrainer(object):
    def __init__(self, collection_dir):
        self.col_root = collection_dir
        self.batch_vectorizer = None
        self.dictionary = artm.Dictionary()
        self.observers = []

    def register(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)

    @property
    def model_factory(self):
        return get_model_factory(self.dictionary)

    def train(self, model, specs):
        """
        :param artm.ARTM model:
        :param patm.modeling.topic_model.TrainSpecs specs:
        """
        model.fit_offline(self.batch_vectorizer, num_collection_passes=specs['collection_passes'])
        self.update_observers(model, specs)


class TrainerFactory(object):

    def create_trainer(self, collection):
        """
        :param collection: the collection name which matches the root directory of all the files related to the collection
        :type collection: str
        :return: an artm model trainer
        :rtype: ModelTrainer
        """
        root_dir = os.path.join(collections_dir, collection)
        bin_dict = os.path.join(root_dir, 'mydic.dict')
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        batches = [_ for _ in os.listdir(root_dir) if '.batch' in _]
        mod_tr = ModelTrainer(collections_dir)

        if batches:
            mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=collection,
                                                           data_path=root_dir,
                                                           data_format='batches')
            print 'vectorizer initialized from \'batches\' file'
        else:
            mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=collection,
                                                           data_path=root_dir,
                                                           data_format='bow_uci',
                                                           target_folder=root_dir)
            print 'vectorizer initialized from \'uci\' file'
        if os.path.exists(bin_dict):
            mod_tr.dictionary.load(bin_dict)
            print 'loaded binary dictionary'
        else:
            mod_tr.dictionary.gather(data_path=root_dir, vocab_file_path=os.path.join(root_dir, 'vocab.'+collection+'.txt'))
            mod_tr.dictionary.save(bin_dict)
            print 'saved binary dictionary'
        return mod_tr
