import warnings
from collections import Counter

from patm.modeling.regularization.regularizers import regularizer2parameters, parameter_name2encoder


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
        self._definition2evaluator_name = {k: v.name for k, v in evaluators.items()} # {'sparsity-phi': 'spd_p', 'top-tokens-10': 'tt10'}
        self._evaluator_name2definition = {v.name: k for k, v in evaluators.items()} # {'sp_p': 'sparsity-phi', 'tt10': 'top-tokens-10'}
        self._reg_type2name = {}  # ie {'smooth-theta': 'smb_t', 'sparse-phi': 'spd_p'}
        self._reg_name2wrapper = {}

    # def add_regularizer(self, reg_object, reg_type):
    #     """
    #     Adds a regularizer component to the optimization likelihood function. Adds the given object to the underlying artm model.\n
    #     :param artm.BaseRegularizer reg_object: the object to add
    #     :param str reg_type:
    #     """
    #     self._reg_type2name[reg_type] = reg_object.name
    #     self.artm_model.regularizers.add(reg_object)

    def add_regularizer_wrapper(self, reg_wrapper):
        self._reg_name2wrapper[reg_wrapper.name] = reg_wrapper
        self._reg_type2name[reg_wrapper.type] = reg_wrapper.name
        self.artm_model.regularizers.add(reg_wrapper.artm_regularizer)

    def add_regularizer_wrappers(self, reg_wrappers):
        for _ in reg_wrappers:
            self.add_regularizer_wrapper(_)

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
    def tau_trajectories(self):
        return filter(lambda x: x[1] is not None, self._trajs.items())

    @property
    def regularizer_names(self):
        return sorted(_ for _ in self.artm_model.regularizers.data)

    @property
    def regularizer_types(self):
        return map(lambda x: self._reg_name2wrapper[x].type, self.regularizer_names)

    @property
    def evaluator_names(self):
        return sorted(self._evaluator_name2definition.keys())

    @property
    def evaluator_definitions(self):
        return [self._evaluator_name2definition[eval_name] for eval_name in self.evaluator_names]

    @property
    def topic_names(self):
        return self.artm_model.topic_names

    @property
    def nb_topics(self):
        return self.artm_model.num_topics

    @property
    def domain_topics(self):
        """Returns the mostly agreed list of topic names found in all evaluators"""
        c = Counter()
        for evaluator in self._definition2evaluator.values():
            if hasattr(evaluator.artm_score, 'topic_names'):
                tn = evaluator.artm_score.topic_names
                if tn:
                    c['+'.join(tn)] += 1
        if len(c) > 1:
            warnings.warn("There exist evaluator objects that target different (domain) topics to score")
        # print c.most_common()
        return c.most_common(1)[0][0].split('+')

    @property
    def background_topics(self):
        return list(filter(None, map(lambda x: x if x not in self.domain_topics else None, self.artm_model.topic_names)))

    @property
    def modalities_dictionary(self):
        return self.artm_model.class_ids

    @property
    def document_passes(self):
        return self.artm_model.num_document_passes

    def get_reg_obj(self, reg_name):
        return self.artm_model.regularizers[reg_name]

    def get_reg_name(self, reg_type):
        return self._reg_type2name.get(reg_type, None)

    def get_evaluator(self, eval_name):
        return self._definition2evaluator[self._evaluator_name2definition[eval_name]]

    def get_scorer_by_name(self, eval_name):
        return self.artm_model.scores[eval_name]

    def get_scorer_by_type(self, eval_type):
        return self.artm_model.scores[self._definition2evaluator_name[eval_type]]

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
                    print e
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
        :return:
        :rtype dict
        """
        d = {}
        for reg_type in self.regularizer_types:
            d[reg_type] = {}
            cur_reg_obj = self.artm_model.regularizers[self._reg_type2name[reg_type]]
            d[reg_type]['name'] = cur_reg_obj.name
            for key in regularizer2parameters[reg_type]:
                d[reg_type][key] = cur_reg_obj.__getattribute__(key)
        return d

    # def get_top_tokens(self, topic_names='all', nb_tokens=10):
    #     if topic_names == 'all':
    #         topic_names = self.artm_model.topic_names
    #     toks = self.artm_model.score_tracker[self._evaluator_name2definition['top-tokens'].name].last_value
    #     return [toks[topic_name] for topic_name in topic_names]
    #
    # def print_topics(self, topic_names='all'):
    #     if topic_names == 'all':
    #         topic_names = self.artm_model.topic_names
    #     toks = self.artm_model.score_tracker[self._evaluator_name2definition['top-tokens'].name].last_value
    #     body, max_lens = self._get_rows(toks)
    #     header = self._get_header(max_lens, topic_names)
    #     print(header + body)

    def _get_header(self, max_lens, topic_names):
        assert len(max_lens) == len(topic_names)
        return ' - '.join(map(lambda x: '{}{}'.format(x[1], ' ' * (max_lens[x[0]] - len(name))), (j, name in enumerate(topic_names))))

    def _get_rows(self, topic_name2tokens):
        max_token_lens = [max(map(lambda x: len(x), topic_name2tokens[name])) for name in self.artm_model.topic_names]
        b = ''
        for i in range(len(topic_name2tokens.values()[0])):
            b += ' | '.join('{} {}'.format(topic_name2tokens[name][i], (max_token_lens[j] - len(topic_name2tokens[name][i])) * ' ') for j, name in enumerate(self.artm_model.topic_names)) + '\n'
        return b, max_token_lens


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
