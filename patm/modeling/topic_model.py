from collections import Mapping
from patm.utils import cfg2model_settings
from .regularizers import regularizer2parameters, parameter_name2encoder
from regularizers import regularizer_class_string2reg_type


class TopicModel(object):
    """
    An instance of this class encapsulates the behaviour of a _topic_model.
    - LIMITATION:
        Does not support dynamic changing of the number of topics between different training cycles.
    """
    def __init__(self, label, artm_model, evaluators):
        """
        Creates a topic model object\n. The model is ready to add regularizers to it.
        :param str label: a unique identifier for the model; model name
        :param artm.ARTM artm_model: a reference to an artm model object
        :param dict evaluators: mapping from evaluator_types to evaluator_names; i.e. {'sparsity-phi': 'sp', 'top-tokens': 'tt'}
        """
        self.observers = []
        self.label = label
        self.artm_model = artm_model
        self._eval_type2eval_wrapper_obj = evaluators
        self._eval_name2eval_type = {v.name: k for k, v in evaluators.items()} # {'sp': 'sparsity-phi', 'tt10': 'top-tokens10'}
        self._reg_type2name = {}  # ie {'smooth-sparse-theta': 'sst', 'smooth-sparse-phi': 'ssp'}

    def add_regularizer(self, reg_object):
        """
        Adds a regularizer component to the optimization likelihood function. Adds the given object to the underlying artm model.\n
        :param artm.BaseRegularizer reg_object: the object to add
        """
        self._reg_type2name[regularizer_class_string2reg_type[type(reg_object).__name__]] = reg_object.name
        self.artm_model.regularizers.add(reg_object)

    @property
    def regularizer_names(self):
        return sorted(_ for _ in self.artm_model.regularizers.data)

    @property
    def regularizer_types(self):
        return [regularizer_class_string2reg_type[type(self.artm_model.regularizers[name]).__name__] for name in self.regularizer_names]

    def get_reg_obj(self, reg_name):
        return self.artm_model.regularizers[reg_name]

    def get_reg_name(self, reg_type):
        return self._reg_type2name.get(reg_type, None)

    @property
    def evaluator_names(self):
        return sorted(self._eval_name2eval_type.keys())

    @property
    def evaluator_types(self):
        return [self._eval_name2eval_type[eval_name] for eval_name in self.evaluator_names]

    def get_scorer_wrapper(self, eval_name):
        return self._eval_type2eval_wrapper_obj[self._eval_name2eval_type[eval_name]]

    def get_scorer(self, eval_name):
        return self.artm_model.scorer[eval_name]

    @property
    def topic_names(self):
        return self.artm_model.topic_names

    @property
    def nb_topics(self):
        return self.artm_model.num_topics

    @property
    def document_passes(self):
        return self.artm_model.num_document_passes
    def set_document_passes(self, iterations):
        self.artm_model.num_document_passes = iterations

    def set_parameter(self, reg_name, reg_param, value):
        if reg_name in self.artm_model.regularizers.data:
            if hasattr(self.artm_model.regularizers[reg_name], reg_param):
                try:
                    self.artm_model.regularizers[reg_name].__setattr__(reg_param, parameter_name2encoder[reg_param](value))
                except (AttributeError, TypeError) as e:
                    print e
            else:
                raise ParameterNameNotFoundException("Regularizer '{}' with name '{}' has no attribute (parameter) '{}'".format(type(self.artm_model.regularizers[reg_name]).__name__, reg_name, reg_param))
        else:
            raise RegularizerNameNotFoundException("Did not find a regularizer set in the artm_model with name '{}'".format(reg_name))

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
            # print ' reg {}: {}'.format(score_type, cur_reg_obj.name)
            d[reg_type]['name'] = cur_reg_obj.name
            for key in regularizer2parameters[reg_type]:
                # print '  recording param:', key
                d[reg_type][key] = cur_reg_obj.__getattribute__(key)
        return d

    def get_top_tokens(self, topic_names='all', nb_tokens=10):
        if topic_names == 'all':
            topic_names = self.artm_model.topic_names
        toks = self.artm_model.score_tracker[self._eval_name2eval_type['top-tokens'].name].last_value
        return [toks[topic_name] for topic_name in topic_names]

    def print_topics(self, topic_names='all'):
        if topic_names == 'all':
            topic_names = self.artm_model.topic_names
        toks = self.artm_model.score_tracker[self._eval_name2eval_type['top-tokens'].name].last_value
        body, max_lens = self._get_rows(toks)
        header = self._get_header(max_lens, topic_names)
        print(header + body)

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

# def get_simple_train_specs(iterations):
#     return TrainSpecs(iterations, [], [])

class RegularizerNameNotFoundException(Exception):
    def __init__(self, msg):
        super(RegularizerNameNotFoundException, self).__init__(msg)

class ParameterNameNotFoundException(Exception):
    def __init__(self, msg):
        super(ParameterNameNotFoundException, self).__init__(msg)
