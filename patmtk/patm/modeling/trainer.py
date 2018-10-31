# -*- coding: utf-8 -*-
import os
import re
import abc
import artm
from .model_factory import get_model_factory
from ..definitions import COLLECTIONS_DIR, COOCURENCE_DICT_FILE_NAMES
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
            if effects:
                print 'Training with constant taus'
                from patm.utils import Spinner
                self.spinner.start()
            topic_model.artm_model.fit_offline(self.batch_vectorizer, num_collection_passes=specs.collection_passes)
            if effects:
                self.spinner.stop()
            self.update_observers(topic_model, specs.collection_passes)
        else:
            steady_iter_chunks = get_fit_iteration_chunks(map(lambda x: x[1], trajectories_data))
            all_iter_chunks = steady_iter_chunks.to_training_chunks(specs.collection_passes)

            iter_sum = 0
            gener = all_iter_chunks
            if effects:
                print 'Will fit on {} chunks and train with tau trajectories for regs [{}]'.format(len(all_iter_chunks), ', '.join((_[0] for _ in trajectories_data)))
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
        self._root_dir = os.path.join(COLLECTIONS_DIR, self._col)
        self._mod_tr = ModelTrainer(COLLECTIONS_DIR)
        self._batches_dir_name = self.ideology_flag2batches_dir_name[exploit_ideology_labels]
        self._batches_target_dir = os.path.join(self._root_dir, self._batches_dir_name)

        self._data_path_hash = {True: os.path.join(self._root_dir, 'vowpal.'+self._col+'.txt'), False: self._root_dir}
        for _ in [self._root_dir, self._batches_target_dir]:
            if not os.path.exists(_):
                os.makedirs(_)
        bin_dict = os.path.join(self._root_dir, 'mydic.dict')
        text_dict = os.path.join(self._root_dir, 'mydic.txt')
        vocab = os.path.join(self._root_dir, 'vocab.' + self._col + '.txt')

        batches = [_ for _ in os.listdir(self._batches_target_dir) if '.batch' in _]
        print 'batches', batches
        if not force_new_batches and batches:
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
        print "Vectorizer initialized from 'batches' found in '{}'".format(self._batches_target_dir)


class DegenerationChecker(object):
    def __init__(self, reference_keys):
        self._keys = sorted(reference_keys)

    def _initialize(self):
        self._iter = 0
        self._degen_info = {key: [] for key in self._keys}
        self._building = {k: False for k in self._keys}
        self._li = {k: 0 for k in self._keys}
        self._ri = {k: 0 for k in self._keys}

    def __str__(self):
        return '[{}]'.format(', '.join(self._keys))
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self))
    @property
    def keys(self):
        return self.keys
    @keys.setter
    def keys(self, keys):
        self._keys = sorted(keys)

    def get_degeneration_info(self, dict_list):
        self._initialize()
        return dict(map(lambda x: (x, self.get_degenerated_tuples(x, self._get_struct(dict_list))), self._keys))

    def get_degen_info_0(self, dict_list):
        self.build(dict_list)
        return self._degen_info

    def build(self, dict_list):
        self._initialize()
        for k in self._keys:
            self._build_degen_info(k, self._get_struct(dict_list))
            self._add_final(k, self._degen_info[k])

    def get_degenerated_tuples(self, key, struct):
        """
        :param str key:
        :param list struct: output of self._get_struct(dict_list)
        :return: a list of tuples indicating starting and finishing train iterations when the information has been degenerated (missing)
        """
        _ = filter(None, map(lambda x: self._build_tuple(key, x[1], x[0]), enumerate(struct)))
        self._add_final(key, _)
        return _

    def _add_final(self, key, info):
        if self._building[key]:
            info.append((self._li[key], self._ri[key]))

    def _build_tuple(self, key, struct_element, iter_count):
        """

        :param str key:
        :param list struct_element:
        :param int iter_count:
        :return:
        """
        r = None
        if key in struct_element:  # if has lost its information (has degenerated) for iteration/collection_pass i
            if self._building[key]:
                self._ri[key] += 1
            else:
                self._li[key] = iter_count
                self._ri[key] = iter_count + 1
                self._building[key] = True
        else:
            if self._building[key]:
                r = (self._li[key], self._ri[key])
                self._building[key] = False
        return r

    def _build_degen_info(self, key, struct):
        for i, el in enumerate(struct):
            if key in el:  # if has lost its information (has degenerated) for iteration/collection_pass i
                if self._building[key]:
                    self._ri[key] += 1
                else:
                    self._li[key] = i
                    self._ri[key] = i + 1
                    self._building[key] = True
            else:
                if self._building[key]:
                    self._degen_info[key].append((self._li[key], self._ri[key]))
                    self._building[key] = False

    def _get_struct(self, dict_list):
        return map(lambda x: self._get_degen_keys(self._keys, x), dict_list)

    def _get_degen_keys(self, key_list, a_dict):
        return filter(None, map(lambda x: x if self._has_degenerated(x, a_dict) else None, key_list))

    def _has_degenerated(self, key, a_dict):
        if key not in a_dict or len(a_dict[key]) == 0 or not a_dict[key]: return True
        return False


class DegenChecker(object):
    __metaclass__ = abc.ABCMeta
    # def __init__(self):
    #     pass
    @abc.abstractmethod
    def has_degenerated(self, entity):
        raise NotImplemented

class DictDegenChecker(DegenChecker):
    def __init__(self, keys):
        self._keys = keys
        # super(DictDegenChecker, self).__init__()

    def has_degenerated(self, entity):
        pass

class ListDegenChecker(DegenChecker):
    def __init__(self, key):
        self._key = key

    def has_degenerated(self, entity):
        if key not in entity or len(entity[key]): return True
        return False


if __name__ == '__main__':
    trainer_factory = TrainerFactory()
    model_trainer = trainer_factory.create_trainer('articles', exploit_ideology_labels=False, force_new_batches=False)
