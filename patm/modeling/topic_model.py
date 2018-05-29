from collections import Mapping
from patm.utils import cfg2model_settings
from .regularizers import regularizer2parameters, parameter_name2encoder
from regularizers import regularizer_class_string2regularizer_type as reg_obj_class_str2reg_type


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
        self.label = label
        self.artm_model = artm_model
        self.evaluators = evaluators
        self.regularizers_set = set()
        self.observers = []
        self.reg_hash = {}  # regularizer_type => regularizer_object

    def add_regularizer(self, reg_object):
        """
        Adds a regularizer component to the optimization likelihood function. Adds the given object to the underlying artm model.\n
        :param artm.BaseRegularizer reg_object: the object to add
        """
        self.reg_hash[reg_obj_class_str2reg_type[type(reg_object).__name__]] = reg_object
        self.artm_model.regularizers.add(reg_object)
        self.regularizers_set.add(reg_object.name)

    @property
    def regularizer_types(self):
        return sorted(self.reg_hash.keys())

    @property
    def regularizer_names(self):
        return sorted(list(self.regularizers_set))

    @property
    def evaluator_types(self):
        return sorted(self.evaluators.keys())

    @property
    def evaluator_names(self):
        return sorted(self.evaluators.values())

    @property
    def topic_names(self):
        return self.artm_model.topic_names

    @property
    def nb_topics(self):
        return self.artm_model.num_topics

    @property
    def document_passes(self):
        return self.artm_model.num_document_passes

    def set_parameter(self, reg_name, reg_param, value):
        if reg_name in self.regularizers_set:
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
        d = {}
        for reg_type in self.regularizer_types:
            d[reg_type] = {}
            print 'REGTYPE', reg_type
            for key in regularizer2parameters[reg_type]:
                print ' ', key
                d[reg_type][key] = self.reg_hash[reg_type].__getattribute__(key)
        return d

    def get_top_tokens(self, topic_names='all', nb_tokens=10):
        if topic_names == 'all':
            topic_names = self.artm_model.topic_names
        toks = self.artm_model.score_tracker[self.evaluators['top-tokens'].name].last_value
        return [toks[topic_name] for topic_name in topic_names]

    def print_topics(self, topic_names='all'):
        if topic_names == 'all':
            topic_names = self.artm_model.topic_names
        toks = self.artm_model.score_tracker[self.evaluators['top-tokens'].name].last_value
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

    def get_top_tokens(self, topic_names='all', nb_tokens=10):
        if topic_names == 'all':
            topic_names = self.model.topic_names
        toks = self.model.score_tracker[self.evaluators['top-tokens'].name].last_value
        return [toks[topic_name] for topic_name in topic_names]


class TrainSpecs(Mapping):
    def __init__(self, *args, **kwargs):
        self._storage = dict(*args, **kwargs)
        assert 'collection_passes' in self._storage

    def __str__(self):
        # return 'Mapping([{}])'.format(', '.join())
        return str(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def fct_method(self, cfg_file):
        return TrainSpecs(collection_passes=cfg2model_settings(cfg_file)['learning']['collection_passes'], output_dir=self['output_dir'])


class RegularizerNameNotFoundException(Exception):
    def __init__(self, msg):
        super(RegularizerNameNotFoundException, self).__init__(msg)

class ParameterNameNotFoundException(Exception):
    def __init__(self, msg):
        super(ParameterNameNotFoundException, self).__init__(msg)
