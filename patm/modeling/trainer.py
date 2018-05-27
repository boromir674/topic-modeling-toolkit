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
        :param collection_passes: number of passes over the whole collection
        :type collection_passes: int
        :param nb_topics:
        :return:
        """

        # model = self.model_factory.create_model()
        model.fit_offline(self.batch_vectorizer, num_collection_passes=specs['collection_passes'])
        # print model.score_tracker['sp'].value
        # print model.score_tracker['my_fisrt_perplexity_score'].last_value
        # print model.score_tracker['sparsity_phi_score'].value  # .last_value
        # print model.score_tracker['sparsity_theta_score'].value  # .last_value
        # saved_top_tokens = model.score_tracker['top_tokens_score'].last_tokens
        # for topic_name in model.topic_names:
        #     print saved_top_tokens[topic_name]
        self.update_observers(model)


class TrainerFactory(object):

    def create_trainer(self, collection):
        """
        :param collection: the collection name which matches the root directory of all the files related to the collection
        :type collection: str
        :return: a model trainer
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


# if __name__ == '__main__':
    # tf = TrainerFactory()
    # trainer = tf.create_trainer('plsa', 'n1c')
    # md_factory = ModelFactory(trainer.dictionary)
    # model_inst = md_factory.create_model('/data/thesis/code/train.cfg')
    # trainer.train(model_inst, 50)
    # print model_inst.num_document_passes
