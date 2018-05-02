import os
import artm

from utils import ModelFactory
from definitions import collections_dir


class ModelTrainer(object):
    def __init__(self, model_type, collection):
        self.type = model_type
        self.col_name = collection
        self.batch_vectorizer = None
        self.dictionary = artm.Dictionary()

    def train(self, model, collection_passes):
        """
        :param collection_passes: number of passes over the whole collection
        :type collection_passes: int
        :param nb_topics:
        :return:
        """

        # model = artm.ARTM(num_topics=nb_topics, dictionary=self.dictionary)
        #
        # model.scores.add(artm.PerplexityScore(name='my_first_perplexity_score', dictionary=self.dictionary))
        # model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
        # model.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
        # model.scores.add(artm.TopTokensScore(name='top_tokens_score'))
        #
        # model.num_document_passes = document_passes
        #
        # model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer'))
        # model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer'))
        # model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'))
        # model.regularizers['sparse_phi_regularizer'].tau = -1.0
        # model.regularizers['sparse_theta_regularizer'].tau = -0.5
        # model.regularizers['decorrelator_phi_regularizer'].tau = 1e+5

        model.fit_offline(self.batch_vectorizer, num_collection_passes=collection_passes)

        print model.score_tracker['sp'].value
        # print model.score_tracker['my_fisrt_perplexity_score'].last_value
        # print model.score_tracker['sparsity_phi_score'].value  # .last_value
        # print model.score_tracker['sparsity_theta_score'].value  # .last_value
        # saved_top_tokens = model.score_tracker['top_tokens_score'].last_tokens
        # for topic_name in model.topic_names:
        #     print saved_top_tokens[topic_name]

        # return model


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


if __name__ == '__main__':
    tf = TrainerFactory()
    mt = tf.create_trainer('plsa', 'n1c')
    md_factory = ModelFactory(mt.dictionary)
    model_inst = md_factory.create_model('/data/thesis/code/train.cfg')
    mt.train(model_inst, 150)
    print model_inst.num_document_passes

