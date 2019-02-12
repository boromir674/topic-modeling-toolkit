# -*- coding: utf-8 -*-
import os
import re
import abc
import artm

from .model_factory import get_model_factory
from ..definitions import COLLECTIONS_DIR_PATH, COOCURENCE_DICT_FILE_NAMES
from patm.modeling.parameters.trajectory import get_fit_iteration_chunks


class ModelTrainer(object):
    """
    This class is responsible for training models on a specific dataset/collection. It can perform the actual (machine)
    learning by fitting a topic model instance according to given train settings/specifications. It also notifies any
    observers/listeners on every training cycle updating them with new evaluation scores as the training progresses.
    """
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
        This method trains according to specifications the given topic model instance. Notifies accordingly any observer
        with the evalution metrics and potential dynamic parameters that change during training (i.e. a regularizer λ coefficient: tau value).\n
        :param patm.modeling.topic_model.TopicModel topic_model:
        :param patm.modeling.topic_model.TrainSpecs specs:
        :param bool effects:
        """
        trajectories_data = specs.tau_trajectory_list
        if not trajectories_data:
            # if effects:
            print 'Training with constant tau coefficients..'

            from patm.utils import Spinner
            spinner = Spinner(delay=0.2)
            spinner.start()
            try:
                topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=specs.collection_passes)
            except RuntimeError as e:
                spinner.stop()
                raise e
            spinner.stop()
            self.update_observers(topic_model, specs.collection_passes)
        else:
            steady_iter_chunks = get_fit_iteration_chunks(map(lambda x: x[1], trajectories_data))
            all_iter_chunks = steady_iter_chunks.to_training_chunks(specs.collection_passes)
            iter_sum = 0
            gener = all_iter_chunks
            if effects:
                print 'Will fit on {} chunks ({} steady) and train with tau trajectories for regs [{}]'.format(len(all_iter_chunks), len(steady_iter_chunks), ', '.join((_[0] for _ in trajectories_data)))
                from tqdm import tqdm
                gener = tqdm(all_iter_chunks, total=len(all_iter_chunks), unit='fit-operation')
            for chunk in gener:
                topic_model.set_parameters(specs.to_taus_slice(iter_sum))
                topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=chunk.span)
                self.update_observers(topic_model, chunk.span)
                iter_sum += chunk.span

class TrainerFactory(object):
    ideology_flag2data_format = {True: 'vowpal_wabbit', False: 'bow_uci'}
    ideology_flag2batches_dir_name = {True: 'vow-batches', False: 'uci-batches'}

    def create_trainer(self, collection, exploit_ideology_labels=False, force_new_batches=False):
        """
        Creates an object that can train any topic model on a specific dataset/collection.\n
        :param str collection: the collection name which matches the root directory of all the files related to the collection
        :param bool exploit_ideology_labels: whether to use the discretizd perspectives as labels for each document. If true then vowpal-formatted dataset has to be found (because it can encode label information per document) to create batches from
        :param bool force_new_batches: whether to force the creation of new batches regardless of fiding any existing ones. Potentially overrides old batches found
        :return: the ready-to-train created object
        :rtype: ModelTrainer
        """
        self._col = collection
        self._root_dir = os.path.join(COLLECTIONS_DIR_PATH, self._col)
        self._mod_tr = ModelTrainer(COLLECTIONS_DIR_PATH)
        self._batches_dir_name = self.ideology_flag2batches_dir_name[exploit_ideology_labels]
        self._batches_target_dir = os.path.join(self._root_dir, self._batches_dir_name)

        self._data_path_hash = {True: os.path.join(self._root_dir, 'vowpal.'+self._col+'.txt'), False: self._root_dir}
        for _ in [self._root_dir, self._batches_target_dir]:
            if not os.path.exists(_):
                os.makedirs(_)
        bin_dict = os.path.join(self._root_dir, 'mydic.dict')
        text_dict = os.path.join(self._root_dir, 'mydic.txt')
        vocab = os.path.join(self._root_dir, 'vocab.' + self._col + '.txt')

        self._existing_batches = [_ for _ in os.listdir(self._batches_target_dir) if '.batch' in _]
        if not force_new_batches and self._existing_batches:
            self.load_batches()
        else:
            self.create_batches(use_ideology_information=exploit_ideology_labels)

        if os.path.exists(bin_dict):
            self._mod_tr.dictionary.load(bin_dict)
            print 'Loaded binary dictionary', bin_dict
        else:
            self._mod_tr.dictionary.gather(data_path=self._root_dir, vocab_file_path=vocab, symmetric_cooc_values=True)
            self._mod_tr.dictionary.save(bin_dict)
            self._mod_tr.dictionary.save_text(text_dict, encoding='utf-8')
            print 'Saved binary dictionary as', bin_dict
            print 'Saved textual dictionary as', text_dict

        # TODO replace with nested map/lambdas and regex
        for fname in os.listdir(self._root_dir):
            name = os.path.basename(fname)
            matc = re.match('^ppmi_(\w\w)_(\d*)', name)
            if matc:
                self._mod_tr.cooc_dicts[matc.group(1)] = {'obj': artm.Dictionary(), 'min': int(matc.group(2))}
                self._mod_tr.cooc_dicts[matc.group(1)]['obj'].gather(data_path=self._root_dir, cooc_file_path=os.path.join(self._root_dir, name), vocab_file_path=vocab, symmetric_cooc_values=True)
                print "Loaded positive pmi '{}' dictionary from '{}' text file".format(matc.group(1), name)
        return self._mod_tr

    def create_batches(self, use_ideology_information=False):
        self._mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=self._col,
                                                               data_path=self._data_path_hash[use_ideology_information],
                                                               data_format=self.ideology_flag2data_format[use_ideology_information],
                                                               target_folder=self._batches_target_dir)
        print "Vectorizer initialized from '{}' file".format(self.ideology_flag2data_format[use_ideology_information])

    def load_batches(self):
        self._mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=self._col,
                                                       data_path=self._batches_target_dir,
                                                       data_format='batches')
        print "Vectorizer initialized from [{}] 'batch' files found in '{}'".format(', '.join(self._existing_batches), self._batches_target_dir)


if __name__ == '__main__':
    trainer_factory = TrainerFactory()
    model_trainer = trainer_factory.create_trainer('articles', exploit_ideology_labels=False, force_new_batches=False)
