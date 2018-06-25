import os
import sys
import artm
from tqdm import tqdm

from patm.utils import cfg2model_settings, Spinner
from ..definitions import collections_dir
from .model_factory import get_model_factory
from ..evaluation.scorer_factory import ArtmScorerFactory
from .parameters import get_fit_iteration_chunks


class ModelTrainer(object):
    def __init__(self, collection_dir):
        self.col_root = collection_dir
        self.batch_vectorizer = None
        self.dictionary = artm.Dictionary()
        self.observers = []
        self.spinner = Spinner()
        self.bar = None

    def register(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)
            observer.set_dictionary(self.dictionary)

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

    def train(self, topic_model, specs):
        """
        :param patm.modeling.topic_model.TopicModel topic_model:
        :param patm.modeling.topic_model.TrainSpecs specs:
        """
        print specs.tau_trajectory_list
        trajectories_list = map(lambda x: x[1], specs.tau_trajectory_list)
        print trajectories_list
        if not trajectories_list:
            print 'Training without any trajectory...'
            self.spinner.start()
            topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=specs.collection_passes)
            self.spinner.stop()
            self.update_observers(topic_model, specs.collection_passes)

        else:
            steady_iter_chunks = get_fit_iteration_chunks(map(lambda x: x[1], specs.tau_trajectory_list))
            all_iter_chunks = steady_iter_chunks.to_training_chunks(specs.collection_passes)
            print 'Training train with steady_chunks', all_iter_chunks, '...'
            iter_sum = 0
            for chunk in tqdm(all_iter_chunks):
                # d = specs.to_taus_slice(iter_sum)
                # print d
                topic_model.set_parameters(specs.to_taus_slice(iter_sum))
                topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=chunk.span)
                self.update_observers(topic_model, chunk.span)
                iter_sum += chunk.span

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
        text_dict = os.path.join(root_dir, 'mydic.txt')
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
            print 'loaded binary dictionary', bin_dict
        else:
            mod_tr.dictionary.gather(data_path=root_dir, vocab_file_path=os.path.join(root_dir, 'vocab.'+collection+'.txt'))
            mod_tr.dictionary.save(bin_dict)
            mod_tr.dictionary.save_text(text_dict, encoding='utf-8')
            print 'saved binary dictionary as', bin_dict
            print 'saved textual dictionary as', text_dict
        return mod_tr


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
