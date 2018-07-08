import os
import re
import sys
import artm
from patm.modeling.parameters.trajectory import get_fit_iteration_chunks
from .model_factory import get_model_factory
from ..definitions import collections_dir, COOCURENCE_DICT_FILE_NAMES


class ModelTrainer(object):
    def __init__(self, collection_dir):
        self.col_root = collection_dir
        self.batch_vectorizer = None
        self.dictionary = artm.Dictionary()
        self.cooc_dicts = {}  # ppmi: positive pmi (Point-Mutual Information)
        self.observers = []
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
        return get_model_factory(self.dictionary, self.cooc_dicts)

    def train(self, topic_model, specs, effects=False):
        """
        :param patm.modeling.topic_model.TopicModel topic_model:
        :param patm.modeling.topic_model.TrainSpecs specs:
        """
        trajectories_list = map(lambda x: x[1], specs.tau_trajectory_list)
        if not trajectories_list:
            if effects:
                print 'Training without dynamic tau trajectories'
                from patm.utils import Spinner
                self.spinner.start()
            topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=specs.collection_passes)
            if effects:
                self.spinner.stop()
            self.update_observers(topic_model, specs.collection_passes)
        else:
            steady_iter_chunks = get_fit_iteration_chunks(map(lambda x: x[1], specs.tau_trajectory_list))
            all_iter_chunks = steady_iter_chunks.to_training_chunks(specs.collection_passes)

            iter_sum = 0
            gener = all_iter_chunks
            if effects:
                print 'Training with dynamic trajectories'
                # gener = tqdm(all_iter_chunks)
            for chunk in gener:
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
        vocab = os.path.join(root_dir, 'vocab.' + collection + '.txt')
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
            mod_tr.dictionary.gather(data_path=root_dir, vocab_file_path=vocab, symmetric_cooc_values=True)
            mod_tr.dictionary.save(bin_dict)
            mod_tr.dictionary.save_text(text_dict, encoding='utf-8')
            print 'saved binary dictionary as', bin_dict
            print 'saved textual dictionary as', text_dict

        # TODO replace with nested map/lambdas and regex
        for fname in os.listdir(root_dir):
            name = os.path.basename(fname)
            matc = re.match('^ppmi_(\w\w)_(\d*)', name)
            if matc:
                mod_tr.cooc_dicts[matc.group(1)] = {'obj': artm.Dictionary(), 'min': int(matc.group(2))}
                mod_tr.cooc_dicts[matc.group(1)]['obj'].gather(data_path=root_dir, cooc_file_path=os.path.join(root_dir, name), vocab_file_path=vocab, symmetric_cooc_values=True)
                print "Loaded positive pmi '{}' dictionary from '{}' text file".format(matc.group(1), name)
        return mod_tr
