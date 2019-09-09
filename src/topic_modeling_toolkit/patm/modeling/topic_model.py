import pprint

import warnings
from collections import Counter

from .regularization.regularizers_factory import REGULARIZER_TYPE_2_DYNAMIC_PARAMETERS_HASH


class TopicModel(object):
    """
    An instance of this class encapsulates the behaviour of a topic_model.
    - LIMITATION:
        Does not support dynamic changing of the number of topics between different training cycles.
    """
    def __init__(self, label, artm_model, evaluators):
        """
        Creates a topic model object. The model is ready to add regularizers to it.\n
        :param str label: a unique identifier for the model; model name
        :param artm.ARTM artm_model: a reference to an artm model object
        :param dict evaluators: mapping from evaluator-definitions strings to patm.evaluation.base_evaluator.ArtmEvaluator objects
         ; i.e. {'perplexity': obj0, 'sparsity-phi-\@dc': obj1, 'sparsity-phi-\@ic': obj2, 'topic-kernel-0.6': obj3,
         'topic-kernel-0.8': obj4, 'top-tokens-10': obj5}
        """
        self.label = label
        self.artm_model = artm_model

        self._definition2evaluator = evaluators

        self.definition2evaluator_name = {k: v.name for k, v in evaluators.items()} # {'sparsity-phi': 'spd_p', 'top-tokens-10': 'tt10'}
        self._evaluator_name2definition = {v.name: k for k, v in evaluators.items()} # {'sp_p': 'sparsity-phi', 'tt10': 'top-tokens-10'}
        self._reg_longtype2name = {}  # ie {'smooth-theta': 'smb_t', 'sparse-phi': 'spd_p'}
        self._reg_name2wrapper = {}

    def add_regularizer_wrapper(self, reg_wrapper):
        self._reg_name2wrapper[reg_wrapper.name] = reg_wrapper
        self._reg_longtype2name[reg_wrapper.long_type] = reg_wrapper.name
        self.artm_model.regularizers.add(reg_wrapper.artm_regularizer)

    def add_regularizer_wrappers(self, reg_wrappers):
        for _ in reg_wrappers:
            self.add_regularizer_wrapper(_)

    @property
    def pformat_regularizers(self):
        def _filter(reg_name):
            d = self._reg_name2wrapper[reg_name].static_parameters
            if 'topic_names' in d:
                del d['topic_names']
            return d
        return pprint.pformat({
            reg_long_type: dict(_filter(reg_name),
                                **{k:v for k,v in {'target topics': (lambda x: 'all' if len(x) == 0 else '[{}]'.
                                                                               format(', '.join(x)))(self.get_reg_obj(reg_name).topic_names),
                                                   'mods': getattr(self.get_reg_obj(reg_name), 'class_ids', None)}.items()}
                                )
            for reg_long_type, reg_name in self._reg_longtype2name.items()
        })

    @property
    def pformat_modalities(self):
        return pprint.pformat(self.modalities_dictionary)

    def initialize_regularizers(self, collection_passes, document_passes):
        """
        Call this method before starting of training to build some regularizers settings from run-time parameters.\n
        :param int collection_passes:
        :param int document_passes:
        """
        self._trajs = {}
        for reg_name, wrapper in self._reg_name2wrapper.items():
            self._trajs[reg_name] = wrapper.get_tau_trajectory(collection_passes)
            wrapper.set_alpha_iters_trajectory(document_passes)

    def get_reg_wrapper(self, reg_name):
        return self._reg_name2wrapper.get(reg_name, None)

    @property
    def regularizer_names(self):
        return sorted(_ for _ in self.artm_model.regularizers.data)

    @property
    def regularizer_wrappers(self):
        return list(map(lambda x: self._reg_name2wrapper[x], self.regularizer_names))

    @property
    def regularizer_types(self):
        return map(lambda x: self._reg_name2wrapper[x].type, self.regularizer_names)

    @property
    def regularizer_unique_types(self):
        return map(lambda x: self._reg_name2wrapper[x].long_type, self.regularizer_names)

    @property
    def long_types_n_types(self):
        return map(lambda x: (self._reg_name2wrapper[x].long_type, self._reg_name2wrapper[x].type), self.regularizer_names)

    @property
    def evaluator_names(self):
        return sorted(_ for _ in self.artm_model.scores.data)

    @property
    def evaluator_definitions(self):
        return [self._evaluator_name2definition[eval_name] for eval_name in self.evaluator_names]

    @property
    def evaluators(self):
        return [self._definition2evaluator[ev_def] for ev_def in self.evaluator_definitions]

    @property
    def tau_trajectories(self):
        return filter(lambda x: x[1] is not None, self._trajs.items())

    @property
    def nb_topics(self):
        return self.artm_model.num_topics

    @property
    def topic_names(self):
        return self.artm_model.topic_names

    @property
    def domain_topics(self):
        """Returns the mostly agreed list of topic names found in all evaluators"""
        c = Counter()
        for definition, evaluator in self._definition2evaluator.items():
            tn = getattr(evaluator, 'topic_names', None)
            if tn:
            # if hasattr(evaluator.artm_score, 'topic_names'):
                # print definition, evaluator.artm_score.topic_names
                # tn = evaluator.artm_score.topic_names
                # if tn:
                c['+'.join(tn)] += 1
        if len(c) > 2:
            warnings.warn("There exist {} different subsets of all the topic names targeted by evaluators".format(len(c)))
        # print c.most_common()
        try:
            return c.most_common(1)[0][0].split('+')
        except IndexError:
            raise IndexError("Failed to compute domain topic names. Please enable at least one score with topics=domain_names argument. Scores: {}".
                             format(['{}: {}'.format(x.name, x.settings) for x in self.evaluators]))

    @property
    def background_topics(self):
        return [topic_name for topic_name in self.artm_model.topic_names if topic_name not in self.domain_topics]

    @property
    def background_tokens(self):
        background_tokens_eval_name = ''
        for eval_def, eval_name in self.definition2evaluator_name.items():
            if eval_def.startswith('background-tokens-ratio-'):
                background_tokens_eval_name = eval_name
        if background_tokens_eval_name:
            res = self.artm_model.score_tracker[background_tokens_eval_name].tokens
            return list(res[-1])

    @property
    def modalities_dictionary(self):
        return self.artm_model.class_ids

    @property
    def document_passes(self):
        return self.artm_model.num_document_passes

    def get_reg_obj(self, reg_name):
        return self.artm_model.regularizers[reg_name]

    def get_reg_name(self, reg_type):
        return self._reg_longtype2name[reg_type]
        # try:
        #     return self._reg_longtype2name[reg_type]
        # except KeyError:
        #     print '{} is not found in {}'.format(reg_type, self._reg_longtype2name)
        #     import sys
        #     sys.exit(1)

    def get_evaluator(self, eval_name):
        return self._definition2evaluator[self._evaluator_name2definition[eval_name]]

    # def get_evaluator_by_def(self, definition):
    #     return self._definition2evaluator[definition]
    #
    # def get_scorer_by_name(self, eval_name):
    #     return self.artm_model.scores[eval_name]

    # def get_targeted_topics_per_evaluator(self):
    #     tps = []
    #     for evaluator in self.artm_model.scores.data:
    #         if hasattr(self.artm_model.scores[evaluator], 'topic_names'):
    #             tps.append((evaluator, self.artm_model.scores[evaluator].topic_names))
    #     return tps

    # def get_formated_topic_names_per_evaluator(self):
    #     return 'MODEL topic names:\n{}'.format('\n'.join(map(lambda x: ' {}: [{}]'.format(x[0], ', '.join(x[1])), self.get_targeted_topics_per_evaluator())))

    @document_passes.setter
    def document_passes(self, iterations):
        self.artm_model.num_document_passes = iterations



    def set_parameter(self, reg_name, reg_param, value):
        if reg_name in self.artm_model.regularizers.data:
            if hasattr(self.artm_model.regularizers[reg_name], reg_param):
                try:
                    # self.artm_model.regularizers[reg_name].__setattr__(reg_param, parameter_name2encoder[reg_param](value))
                    self.artm_model.regularizers[reg_name].__setattr__(reg_param, value)
                except (AttributeError, TypeError) as e:
                    print(e)
            else:
                raise ParameterNameNotFoundException("Regularizer '{}' with name '{}' has no attribute (parameter) '{}'".format(type(self.artm_model.regularizers[reg_name]).__name__, reg_name, reg_param))
        else:
            raise RegularizerNameNotFoundException("Did not find a regularizer in the artm_model with name '{}'".format(reg_name))

    def set_parameters(self, reg_name2param_settings):
        for reg_name, settings in reg_name2param_settings.items():
            for param, value in settings.items():
                self.set_parameter(reg_name, param, value)

    def get_regs_param_dict(self):
        """
        Returns a mapping between the model's regularizers unique string definition (ie shown in train.cfg: eg 'smooth-phi', 'sparse-theta',
        'label-regulariation-phi-cls', 'decorrelate-phi-dom-def') and their corresponding parameters that can be dynamically changed during training (eg 'tau', 'gamma').\n
        See patm.modeling.regularization.regularizers.REGULARIZER_TYPE_2_DYNAMIC_PARAMETERS_HASH\n
        :return: the regularizer type (str) to parameters (list of strings) mapping (str => list)
        :rtype: dict
        """
        d = {}
        for unique_type, reg_type in self.long_types_n_types:
            d[unique_type] = {}
            cur_reg_obj = self.artm_model.regularizers[self._reg_longtype2name[unique_type]]
            for attribute_name in REGULARIZER_TYPE_2_DYNAMIC_PARAMETERS_HASH[reg_type]:  # ie for _ in ('tau', 'gamma')
                d[unique_type][attribute_name] = getattr(cur_reg_obj, attribute_name)
        return d
    #
    # def _get_header(self, max_lens, topic_names):
    #     assert len(max_lens) == len(topic_names)
    #     return ' - '.join(map(lambda x: '{}{}'.format(x[1], ' ' * (max_lens[x[0]] - len(name))), (j, name in enumerate(topic_names))))
    #
    # def _get_rows(self, topic_name2tokens):
    #     max_token_lens = [max(map(lambda x: len(x), topic_name2tokens[name])) for name in self.artm_model.topic_names]
    #     b = ''
    #     for i in range(len(topic_name2tokens.values()[0])):
    #         b += ' | '.join('{} {}'.format(topic_name2tokens[name][i], (max_token_lens[j] - len(topic_name2tokens[name][i])) * ' ') for j, name in enumerate(self.artm_model.topic_names)) + '\n'
    #     return b, max_token_lens


class TrainSpecs(object):
    def __init__(self, collection_passes, reg_names, tau_trajectories):
        self._col_iter = collection_passes
        assert len(reg_names) == len(tau_trajectories)
        self._reg_name2tau_trajectory = dict(zip(reg_names, tau_trajectories))
        # print "TRAIN SPECS", self._col_iter, map(lambda x: len(x), self._reg_name2tau_trajectory.values())
        assert all(map(lambda x: len(x) == self._col_iter, self._reg_name2tau_trajectory.values()))

    def tau_trajectory(self, reg_name):
        return self._reg_name2tau_trajectory.get(reg_name, None)

    @property
    def tau_trajectory_list(self):
        """Returns the list of (reg_name, tau trajectory) pairs, sorted alphabetically by regularizer name"""
        return sorted(self._reg_name2tau_trajectory.items(), key=lambda x: x[0])

    @property
    def collection_passes(self):
        return self._col_iter

    def to_taus_slice(self, iter_count):
        return {reg_name: {'tau': trajectory[iter_count]} for reg_name, trajectory in self._reg_name2tau_trajectory.items()}
        # return dict(zip(self._reg_name2tau_trajectory.keys(), map(lambda x: x[iter_count], self._reg_name2tau_trajectory.values())))


class RegularizerNameNotFoundException(Exception):
    def __init__(self, msg):
        super(RegularizerNameNotFoundException, self).__init__(msg)

class ParameterNameNotFoundException(Exception):
    def __init__(self, msg):
        super(ParameterNameNotFoundException, self).__init__(msg)
