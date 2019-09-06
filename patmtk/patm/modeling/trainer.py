# -*- coding: utf-8 -*-
import os
import re

import artm

from .regularization.trajectory import get_fit_iteration_chunks
from .model_factory import ModelFactory


class ModelTrainer(object):
    """
    This class is responsible for training models on a specific dataset/collection. It can perform the actual (machine)
    learning by fitting a topic model instance according to given train settings/specifications. It also notifies any
    observers/listeners on every training cycle updating them with new evaluation scores as the training progresses.
    """
    def __init__(self):
        self.batch_vectorizer = None
        # self.dictionary = artm.Dictionary()
        self._pmi_key = ''
        self.ppmi_dicts = {}  # ppmi: positive pmi (Point-Mutual Information)
        self.observers = []

    @property
    def dictionary(self):
        """The dictionary used to compute all coherence related metric"""
        return self.ppmi_dicts[self._pmi_key]['obj']

    @dictionary.setter
    def dictionary(self, dictionary_obj):
        """The dictionary used to compute all coherence related metric"""
        self._pmi_key = self.__search(dictionary_obj)

    def __search(self, dictionary_obj):
        for k, v in self.ppmi_dicts.items():
            if v['obj'] == dictionary_obj:
                return k
        raise RuntimeError("Dict {} not found in {}".format(dictionary_obj, self.ppmi_dicts))

    def register(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)
            observer.dictionary = self.dictionary

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
        return ModelFactory(self.dictionary)

    def train(self, topic_model, specs, effects=False, cache_theta=False):
        """
        This method trains according to specifications the given topic model instance. Notifies accordingly any observer
        with the evalution metrics and potential dynamic parameters that change during training (i.e. a regularizer λ coefficient: tau value).\n
        :param patm.modeling.topic_model.TopicModel topic_model:
        :param patm.modeling.topic_model.TrainSpecs specs:
        :param bool effects:
        """
        trajectories_data = specs.tau_trajectory_list
        topic_model.artm_model.cache_theta = cache_theta
        if not trajectories_data:
            if effects:
                print("Training, keeping regularizers' coefficients constant..")
                from patm.utils import Spinner
                spinner = Spinner(delay=0.2)
                spinner.start()
                try:
                    topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=specs.collection_passes)
                except (Exception, KeyboardInterrupt) as e:
                    spinner.stop()
                    raise e
                spinner.stop()
            else:
                topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=specs.collection_passes)
            self.update_observers(topic_model, specs.collection_passes)
        else:
            steady_iter_chunks = get_fit_iteration_chunks(map(lambda x: x[1], trajectories_data))
            all_iter_chunks = steady_iter_chunks.to_training_chunks(specs.collection_passes)
            iter_sum = 0
            gener = all_iter_chunks
            if effects:
                print('Will fit on {} chunks ({} steady) and train with tau trajectories for regs [{}]'.format(len(all_iter_chunks), len(steady_iter_chunks), ', '.join((_[0] for _ in trajectories_data))))
                from tqdm import tqdm
                gener = tqdm(all_iter_chunks, total=len(all_iter_chunks), unit='fit-operation')
            for chunk in gener:
                topic_model.set_parameters(specs.to_taus_slice(iter_sum))
                topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=chunk.span)
                self.update_observers(topic_model, chunk.span)
                iter_sum += chunk.span


class TrainerFactory(object):
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(TrainerFactory, cls).__new__(cls)
        if 'collections_root_dir' in kwargs:
            cls.__instance._collections_root = kwargs['collections_root_dir']
        return cls.__instance

    ideology_flag2data_format = {True: 'vowpal_wabbit', False: 'bow_uci'}
    ideology_flag2batches_dir_name = {True: 'vow-batches', False: 'uci-batches'}

    def create_trainer(self, collection, exploit_ideology_labels=True, force_new_batches=False):
        """
        Creates an object that can train any topic model on a specific dataset/collection.\n
        :param str collection: the collection dir path which matches the root directory of all the files related to the collection
        :param bool exploit_ideology_labels: whether to use the calss labels for each document. If true then vowpal-formatted dataset has to be found (because it can encode label information per document) to create batches from
        :param bool force_new_batches: whether to force the creation of new batches regardless of fiding any existing ones. Potentially overrides old batches found
        :return: the ready-to-train created object
        :rtype: ModelTrainer
        """
        self._col = os.path.basename(collection)
        self._root_dir = collection
        self._mod_tr = ModelTrainer()
        self._batches_dir_name = self.ideology_flag2batches_dir_name[exploit_ideology_labels]
        self._batches_target_dir = os.path.join(self._root_dir, self._batches_dir_name)

        self._data_path_hash = {True: os.path.join(self._root_dir, 'vowpal.'+self._col+'.txt'), False: self._root_dir}
        for _ in [self._root_dir, self._batches_target_dir]:
            if not os.path.exists(_):
                os.makedirs(_)
        bin_dict = os.path.join(self._root_dir, 'mydic.dict')
        text_dict = os.path.join(self._root_dir, 'mydic.txt')
        vocab = os.path.join(self._root_dir, 'vocab.' + self._col + '.txt')

        existing_batches = [_ for _ in os.listdir(self._batches_target_dir) if '.batch' in _]
        if not force_new_batches and existing_batches:
            self.load_batches(existing_batches)
        else:
            self.create_batches(use_ideology_information=exploit_ideology_labels)

        # TODO replace with nested map/lambdas and regex
        for fname in os.listdir(self._root_dir):
            name = os.path.basename(fname)
            matc = re.match(r'^ppmi_(\d+)_([td]f)\.txt$', name)
            if matc:
                self._mod_tr.ppmi_dicts[matc.group(2)] = {'obj': artm.Dictionary(name=name), 'min': int(matc.group(1))}
                self._mod_tr.ppmi_dicts[matc.group(2)]['obj'].gather(data_path=self._root_dir,
                                                                     cooc_file_path=os.path.join(self._root_dir, name),
                                                                     vocab_file_path=vocab,
                                                                     symmetric_cooc_values=True)
                print("Loaded positive pmi dictionary with min_{} = {} from '{}' text file".format(matc.group(2), matc.group(1), name))
        if not 'tf' in self._mod_tr.ppmi_dicts:
            raise RuntimeError("Key 'tf' should be in the coocurrences dictionaries hash: Instead these is the hash: [{}]".format('{}: {}'.format(k, v) for k,v in self._mod_tr.ppmi_dicts.items()))

        ##### DECIDE HOW TO COMPUTE PPMI
        self._mod_tr.dictionary = self._mod_tr.ppmi_dicts['tf']['obj']

        # if os.path.exists(bin_dict):
        #     self._mod_tr.dictionary.load(bin_dict)
        #     print 'Loaded binary dictionary', bin_dict
        # else:
        #     self._mod_tr.dictionary.save(bin_dict)
        #     self._mod_tr.dictionary.save_text(text_dict, encoding='utf-8')
        #     print 'Saved binary dictionary as', bin_dict
        #     print 'Saved textual dictionary as', text_dict

        return self._mod_tr

    def create_batches(self, use_ideology_information=False):
        self._mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=self._col,
                                                               data_path=self._data_path_hash[use_ideology_information],
                                                               data_format=self.ideology_flag2data_format[use_ideology_information],
                                                               target_folder=self._batches_target_dir)
        print("Vectorizer initialized from '{}' file".format(self.ideology_flag2data_format[use_ideology_information]))

    def load_batches(self, batch_files_list):
        self._mod_tr.batch_vectorizer = artm.BatchVectorizer(collection_name=self._col,
                                                       data_path=self._batches_target_dir,
                                                       data_format='batches')
        print("Vectorizer initialized from [{}] 'batch' files found in '{}'".format(', '.join(batch_files_list), self._batches_target_dir))


if __name__ == '__main__':
    trainer_factory = TrainerFactory()
    model_trainer = trainer_factory.create_trainer('articles', exploit_ideology_labels=False, force_new_batches=False)
