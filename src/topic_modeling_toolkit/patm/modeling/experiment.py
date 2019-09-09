from .model_factory import ModelFactory
from .persistence import ResultsWL, ModelWL
from .regularization.regularizers_factory import REGULARIZER_TYPE_2_DYNAMIC_PARAMETERS_HASH as DYN_COEFS


class Experiment:
    """
    This class encapsulates experimental related activities such as tracking of various quantities during model training
    and also has capabilities of persisting these tracked quantities as experimental results.\n
    Experimental results include:
    - number of topics: the number of topics inffered
    - document passes: the inner iterations over each document. Phi matrix gets updated 'document_passes' x 'nb_docs' times during one path through the whole collection
    - root dir: the base directory of the colletion, on which experiments were conducted
    - model label: the unique identifier of the model used for the experiments
    - trackable "evaluation" metrics
    - regularization parameters
    - _evaluator_name2definition
    """


    MAX_DECIMALS = 2
    def __init__(self, dataset_dir):
        """
        Encapsulates experimentation by doing topic modeling on a dataset/'collection' in the given patm_root_dir. A 'collection' is a proccessed document collection into BoW format and possibly split into 'train' and 'test' splits.\n
        :param str dataset_dir: the full path to a dataset/collection specific directory
        :param str cooc_dict: the full path to a dataset/collection specific directory
        """
        self._dir = dataset_dir
        # self.cooc_dict = cooc_dict
        self._loaded_dictionary = None # artm.Dictionary object. Data population happens uppon artm.Artm object creation in model_factory; dictionary.load(bin_dict_path) is called there
        self._topic_model = None
        self.collection_passes = []
        self.trackables = None
        # self.train_results_handler = ResultsWL.from_experiment(self)
        # self.phi_matrix_handler = ModelWL.from_experiment(self)
        self.train_results_handler = ResultsWL(self)
        self.phi_matrix_handler = ModelWL(self)
        self.failed_top_tokens_coherence = {}
        self._total_passes = 0
        self.regularizers_dynamic_parameters = None
        self._last_tokens = {}

    def init_empty_trackables(self, model):
        self._topic_model = model
        self.trackables = {}
        self.failed_top_tokens_coherence = {}
        self._total_passes = 0
        for evaluator_definition in model.evaluator_definitions:
            if self._strip_parameters(evaluator_definition) in ('perplexity', 'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio'):
                self.trackables[Experiment._assert_max_decimals(evaluator_definition)] = []
            elif evaluator_definition.startswith('topic-kernel-'):
                self.trackables[Experiment._assert_max_decimals(evaluator_definition)] = [[], [], [], [], {t_name: {'coherence': [], 'contrast': [], 'purity': [], 'size': []} for t_name in model.domain_topics}]
            elif evaluator_definition.startswith('top-tokens-'):
                self.trackables[evaluator_definition] = [[], {t_name: [] for t_name in model.domain_topics}]
                self.failed_top_tokens_coherence[evaluator_definition] = {t_name: [] for t_name in model.domain_topics}
        self.collection_passes = []
        if not all(x in DYN_COEFS for _, x in model.long_types_n_types):
            raise KeyError("One of the [{}] 'reg_type' (not unique types!) should be added in the REGULARIZER_TYPE_2_DYNAMIC_PARAMETERS_HASH in the regularizers_factory module".format(', '.join("'{}'".format(x) for _, x in model.long_types_n_types)))
        self.regularizers_dynamic_parameters = {unique_type: {attr: [] for attr in DYN_COEFS[reg_type]} for unique_type, reg_type in model.long_types_n_types}

    @property
    def dataset_iterations(self):
        return sum(self.collection_passes)

    @property
    def model_factory(self):
        return ModelFactory(self.dictionary)

    @property
    def topic_model(self):
        """
        :rtype: patm.modeling.topic_model.TopicModel
        """
        return self._topic_model

    @staticmethod
    def _strip_parameters(score_definition):
        tokens = []
        for el in score_definition.split('-'):
            try:
                _ = float(el)
            except ValueError:
                if el[0] != '@':
                    tokens.append(el)
        return '-'.join(tokens)

    @classmethod
    def _assert_max_decimals(cls, definition):
        """Converts like:\n
        - 'topic-kernel-0.6'            -> 'topic-kernel-0.60'\n
        - 'topic-kernel-0.871'          -> 'topic-kernel-0.87'\n
        - 'background-tokens-ratio-0.3' -> 'background-tokens-ratio-0.30'
        """
        s = definition.split('-')
        sl = s[-1]
        try:
            _ = float(sl)
            if len(sl) < 4:
                sl = '{}{}'.format(sl, '0'*(2 + cls.MAX_DECIMALS- len(sl)))
            else:
                sl = sl[:4]
            return '-'.join(s[:-1] + [sl])
        except ValueError:
            return definition

    @property
    def dictionary(self):
        """This dictionary is passed to Perplexity artm scorer"""
        return self._loaded_dictionary

    @dictionary.setter
    def dictionary(self, artm_dictionary):
        self._loaded_dictionary = artm_dictionary

    # TODO refactor this; remove dubious exceptions
    def update(self, topic_model, span):
        self.collection_passes.append(span) # iterations performed on the train set for the current 'steady' chunk

        for unique_reg_type, reg_settings in list(topic_model.get_regs_param_dict().items()):
            for param_name, param_value in list(reg_settings.items()):
                self.regularizers_dynamic_parameters[unique_reg_type][param_name].extend([param_value] * span)
        for evaluator_name, evaluator_definition in zip(topic_model.evaluator_names, topic_model.evaluator_definitions):
            reportable_to_results = topic_model.get_evaluator(evaluator_name).evaluate(topic_model.artm_model)
            definition_with_max_decimals = Experiment._assert_max_decimals(evaluator_definition)

            if self._strip_parameters(evaluator_definition) in ('perplexity', 'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio'):
                self.trackables[definition_with_max_decimals].extend(reportable_to_results['value'][-span:])
            elif evaluator_definition.startswith('topic-kernel-'):
                self.trackables[definition_with_max_decimals][0].extend(reportable_to_results['average_coherence'][-span:])
                self.trackables[definition_with_max_decimals][1].extend(reportable_to_results['average_contrast'][-span:])
                self.trackables[definition_with_max_decimals][2].extend(reportable_to_results['average_purity'][-span:])
                self.trackables[definition_with_max_decimals][3].extend(reportable_to_results['average_size'][-span:])
                for topic_name, topic_metrics in list(self.trackables[definition_with_max_decimals][4].items()):
                    topic_metrics['coherence'].extend([x[topic_name] for x in reportable_to_results['coherence'][-span:]])
                    topic_metrics['contrast'].extend([x[topic_name] for x in reportable_to_results['contrast'][-span:]])
                    topic_metrics['purity'].extend([x[topic_name] for x in reportable_to_results['purity'][-span:]])
                    topic_metrics['size'].extend([x[topic_name] for x in reportable_to_results['size'][-span:]])
            elif evaluator_definition.startswith('top-tokens-'):
                self.trackables[evaluator_definition][0].extend(reportable_to_results['average_coherence'][-span:])

                assert 'coherence' in reportable_to_results
                for topic_name, topic_metrics in list(self.trackables[evaluator_definition][1].items()):
                    try:
                        topic_metrics.extend([x[topic_name] for x in reportable_to_results['coherence'][-span:]])
                    except KeyError:
                        if len(self.failed_top_tokens_coherence[evaluator_definition][topic_name]) == 0:
                            self.failed_top_tokens_coherence[evaluator_definition][topic_name].append((self._total_passes, span))
                        else:
                            if span + self.failed_top_tokens_coherence[evaluator_definition][topic_name][-1][0] == self._total_passes:
                                perv_tuple = self.failed_top_tokens_coherence[evaluator_definition][topic_name][-1]
                                self.failed_top_tokens_coherence[evaluator_definition][topic_name][-1] = (self._total_passes, perv_tuple[1]+span)
                self._total_passes += span
        self.final_tokens = {
            'topic-kernel': {eval_def: self._get_final_tokens(eval_def) for eval_def in
                             self.topic_model.evaluator_definitions if eval_def.startswith('topic-kernel-')},
            'top-tokens': {eval_def: self._get_final_tokens(eval_def) for eval_def in
                           self.topic_model.evaluator_definitions if eval_def.startswith('top-tokens-')},
            'background-tokens': self.topic_model.background_tokens
        }

    @property
    def current_root_dir(self):
        return self._dir

    def save_experiment(self, save_phi=True):
        """Dumps the dictionary-type accumulated experimental results with the given file name. The file is saved in the directory specified by the latest train specifications (TrainSpecs)"""
        if not self.collection_passes:
            raise DidNotReceiveTrainSignalException('Model probably hasn\'t been fitted since len(self.collection_passes) = {}'.format(len(self.collection_passes)))
        # asserts that the number of observations recorded per tracked metric variables is equal to the number of "collections pass"; training iterations over the document dataset
        # artm.ARTM.scores satisfy this
        # print 'Saving model \'{}\', train set iterations: {}'.format(self.topic_model.label, self.collection_passes)
        # assert all(map(lambda x: len(x) == sum(self.collection_passes), [values_list for eval2scoresdict in self.trackables.values() for values_list in eval2scoresdict.values()]))
        # the above will fail when metrics outside the artm library start to get tracked, because these will be able to only capture the last state of the metric trajectory due to fitting by "chunks"
        self.train_results_handler.save(self._topic_model.label)
        if save_phi:
            self.phi_matrix_handler.save(self._topic_model.label)

    def load_experiment(self, model_label):
        """
        Given a unigue model label, restores the state of the experiment from disk. Loads all tracked values of the experimental results
        and the state of the TopicModel inferred so far: namely the phi p_wt matrix.
        In details loads settings:\n
        - doc collection fit iteration steady_chunks\n
        - eval metrics/measures trackes per iteration\n
        - regularization parameters\n
        - document passes\n
        :param str model_label: a unigue identifier of a topic model
        :return: the latest train specification used in the experiment
        :rtype: patm.modeling.topic_model.TopicModel
        """
        results = self.train_results_handler.load(model_label)
        self._topic_model = self.phi_matrix_handler.load(model_label, results)
        # self._topic_model.scores = {v.name: topic_model.artm_model.score_tracker[v.name] for v in topic_model._definition2evaluator.values()}
        self.failed_top_tokens_coherence = {}
        self.collection_passes = [item for sublist in results.tracked.collection_passes for item in sublist]
        # [x for x in ] [sublist for sublist in results.tracked.collection_passes in x for x in sublist for sublist in results.tracked.collection_passes]
        self._total_passes = sum(sum(_) for _ in results.tracked.collection_passes)
        try:
            self.regularizers_dynamic_parameters = dict(results.tracked.regularization_dynamic_parameters)
        except KeyError as e:
            print(e)
            raise RuntimeError("Tracked: {}, dynamic reg params: {}".format(results.tracked, results.tracked.regularization_dynamic_parameters))
        self.trackables = self._get_trackables(results)
        self.final_tokens = {
            'topic-kernel': {kernel_def: {t_name: list(tokens) for t_name, tokens in topics_tokens} for kernel_def, topics_tokens in list(results.final.kernel_hash.items())},
            'top-tokens': {top_tokens_def: {t_name: list(tokens) for t_name, tokens in topics_tokens} for top_tokens_def, topics_tokens in list(results.final.top_hash.items())},
            'background-tokens': list(results.final.background_tokens),
        }
        return self._topic_model

    def _get_final_tokens(self, evaluation_definition):
        return self.topic_model.artm_model.score_tracker[self.topic_model.definition2evaluator_name[evaluation_definition]].tokens[-1]


    def _get_trackables(self, results):
        """
        :param results.experimental_results.ExperimentalResults results:
        :return:
        :rtype: dict
        """
        trackables = {}
        for k in (_.replace('_', '-') for _ in dir(results.tracked) if _ not in ('tau_trajectories', 'regularization_dynamic_parameters')):
            if self._strip_parameters(k) in ('perplexity', 'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio'):
                trackables[Experiment._assert_max_decimals(k)] = results.tracked[k].all
            elif k.startswith('topic-kernel-'):
                trackables[Experiment._assert_max_decimals(k)] = [
                    results.tracked[k].average.coherence.all,
                    results.tracked[k].average.contrast.all,
                    results.tracked[k].average.purity.all,
                    results.tracked[k].average.size.all,
                    {t_name: {'coherence': getattr(results.tracked[k], t_name).coherence.all,
                              'contrast': getattr(results.tracked[k], t_name).contrast.all,
                              'purity': getattr(results.tracked[k], t_name).purity.all,
                              'size': getattr(results.tracked[k], t_name).size.all} for t_name in results.scalars.domain_topics}
                ]
            elif k.startswith('top-tokens-'):
                trackables[k] = [results.tracked[k].average_coherence.all,
                                      {t_name: getattr(results.tracked[k], t_name).all for t_name in results.scalars.domain_topics}]
        return trackables


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
        return dict([(x, self.get_degenerated_tuples(x, self._get_struct(dict_list))) for x in self._keys])

    # def get_degen_info_0(self, dict_list):
    #     self.build(dict_list)
    #     return self._degen_info
    #
    # def build(self, dict_list):
    #     self._prepare_storing()
    #     for k in self._keys:
    #         self._build_degen_info(k, self._get_struct(dict_list))
    #         self._add_final(k, self._degen_info[k])

    def get_degenerated_tuples(self, key, struct):
        """
        :param str key:
        :param list struct: output of self._get_struct(dict_list)
        :return: a list of tuples indicating starting and finishing train iterations when the information has been degenerated (missing)
        """
        _ = [_f for _f in [self._build_tuple(key, x[1], x[0]) for x in enumerate(struct)] if _f]
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
        return [self._get_degen_keys(self._keys, x) for x in dict_list]

    def _get_degen_keys(self, key_list, a_dict):
        return [_f for _f in [x if self._has_degenerated(x, a_dict) else None for x in key_list] if _f]

    def _has_degenerated(self, key, a_dict):
        if key not in a_dict or len(a_dict[key]) == 0 or not a_dict[key]: return True
        return False


class EvaluationOutputLoadingException(Exception): pass
class DidNotReceiveTrainSignalException(Exception): pass
