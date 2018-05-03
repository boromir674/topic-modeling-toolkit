import os
import artm

from ..definitions import collections_dir


class ModelTrainer(object):
    def __init__(self, model_type, collection):
        self.type = model_type
        self.col_name = collection
        self.batch_vectorizer = None
        self.dictionary = artm.Dictionary()

    @property
    def model_factory(self):
        return get_model_factory()

    def train(self, model, collection_passes):
        """
        :param collection_passes: number of passes over the whole collection
        :type collection_passes: int
        :param nb_topics:
        :return:
        """
        model.fit_offline(self.batch_vectorizer, num_collection_passes=collection_passes)
        # print model.score_tracker['sp'].value
        # print model.score_tracker['my_fisrt_perplexity_score'].last_value
        # print model.score_tracker['sparsity_phi_score'].value  # .last_value
        # print model.score_tracker['sparsity_theta_score'].value  # .last_value
        # saved_top_tokens = model.score_tracker['top_tokens_score'].last_tokens
        # for topic_name in model.topic_names:
        #     print saved_top_tokens[topic_name]


class TrainerFactory(object):

    def create_trainer(self, model_type, collection):
        collection_dir = os.path.join(collections_dir, collection)
        bin_dict = os.path.join(collection_dir, 'mydic.dict')
        if not os.path.exists(collection_dir):
            os.makedirs(collection_dir)
        batches = [_ for _ in os.listdir(collection_dir) if '.batch' in _]
        if model_type == 'plsa':
            mod_tr = ModelTrainer(model_type, collection)
            if batches:
                mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=collection,
                                                               data_path=collection_dir,
                                                               data_format='batches')
                print 'vectorizer initialized from \'batches\' file'
            else:
                mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=collection,
                                                               data_path=collection_dir,
                                                               data_format='bow_uci',
                                                               target_folder=collection_dir)
                print 'vectorizer initilized from \'uci\' file'
            if os.path.exists(bin_dict):
                mod_tr.dictionary.load(bin_dict)
                print 'loaded binary dictionary'
            else:
                mod_tr.dictionary.gather(data_path=collection_dir, vocab_file_path=os.path.join(collection_dir, 'vocab.'+collection+'.txt'))
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
