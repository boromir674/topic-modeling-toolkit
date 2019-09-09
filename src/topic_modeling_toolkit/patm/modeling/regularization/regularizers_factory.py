from os import path
import artm
from collections import OrderedDict
from configparser import ConfigParser

from topic_modeling_toolkit.patm.utils import cfg2model_settings
from topic_modeling_toolkit.patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME  # this is the name of the default modality. it is irrelevant to class lebels or document lcassification

from .regularizers import ArtmRegularizerWrapper

import logging
logger = logging.getLogger(__name__)

import attr

my_dir = path.dirname(path.realpath(__file__))
REGULARIZERS_CFG = path.join(my_dir, './regularizers.cfg')


set1 = ('tau', 'gamma', 'class_ids', 'topic_names')
set2 = ('tau', 'topic_names', 'alpha_iter', 'doc_titles', 'doc_topic_coef')
decorrelation = ('tau', 'gamma', 'class_ids', 'topic_names', 'topic_pairs')
regularizer2parameters = {
    'smooth-phi': set1,
    'sparse-phi': set1,
    'smooth-theta': set2,
    'sparse-theta': set2,
    'decorrelate-phi': decorrelation,
    # 'decorrelate-phi-def': decorrelation,  # default token modality
    # 'decorrelate-phi-class': decorrelation,  # doc class modality
    # 'decorrelate-phi-domain': decorrelation,  # all modalities
    # 'decorrelate-phi-background': decorrelation,  # all modalities
    'label-regularization-phi': set1,
    # 'kl-function-info': ('function_type', 'power_value'),
    # 'specified-sparse-phi': ('tau', 'gamma', 'topic_names', 'class_id', 'num_max_elements', 'probability_threshold', 'sparse_by_column'),
    'improve-coherence': ('tau', 'gamma', 'class_ids', 'topic_names'),
    # 'smooth-ptdw': ('tau', 'topic_names', 'alpha_iter'),
    # 'topic-selection': ('tau', 'topic_names', 'alpha_iter')
    # the reasonable parameters to consider experiment with setting to different values and to also change between training cycles
}


# parameters that can be dynamically updated during training ie tau coefficients for sparsing regularizers should increase in absolute values (start below zero and decrease )
supported_dynamic_parameters = ('tau', 'gamma')

REGULARIZER_TYPE_2_DYNAMIC_PARAMETERS_HASH = dict([(k, [_ for _ in v if _ in supported_dynamic_parameters]) for k, v in regularizer2parameters.items()])


parameter_name2encoder = {
    'tau': str, # the coefficient of regularization for this regularizer
    'start': str, # the number of iterations to initially keep the regularizer turned off
    'gamma': float, # the coefficient of relative regularization for this regularizer
    'alpha_iter': str, # list of additional coefficients of regularization on each iteration over document. Should have length equal to model.num_document_passes (of an artm.ARTM model object)
    # 'class_ids': list, # list of class_ids or single class_id to regularize, will regularize all classes if empty or None [ONLY for ImproveCoherencePhiRegularizer: dictionary should contain pairwise tokens co-occurrence info]
    # 'topic_names': list, # specific topics to target for applying regularization, Targets all if None
    # 'topic_pairs': dict, # key=topic_name, value=dict: pairwise topic decorrelation coefficients, if None all values equal to 1.0
    # 'function_type': str, # the type of function, 'log' (logarithm) or 'pol' (polynomial)
    # 'power_value': float, # the float power of polynomial, ignored if 'function_type' = 'log'
    # 'doc_titles': list, # of strings: list of titles of documents to be processed by this regularizer. Default empty value means processing of all documents. User should guarantee the existence and correctness of document titles in batches (e.g. in src files with data, like WV).
    # 'doc_topic_coef': list, # (list of floats or list of list of floats): of floats len=nb_topics or of lists of floats len=len(doc_titles) and len(inner_list)=_nb_topics: Two cases: 1) list of floats with length equal to num of topics. Means additional multiplier in M-step formula besides alpha and tau, unique for each topic, but general for all processing documents. 2) list of lists of floats with outer list length equal to length of doc_titles, and each inner list length equal to num of topics. Means case 1 with unique list of additional multipliers for each document from doc_titles. Other documents will not be regularized according to description of doc_titles parameter. Note, that doc_topic_coef and topic_names are both using.
    # 'num_max_elements': int, # number of elements to save in row/column for the artm.SpecifiedSparsePhiRegularizer
    # 'probability_threshold': float, # if m elements in row/column sum into value >= probability_threshold, m < n => only these elements would be saved. Value should be in (0, 1), default=None
    # 'sparse_by_columns': bool, # find max elements in column or row
}


class RegularizersFactory:
    """Supports construction of Smoothing, Sparsing, Label, Decorrelating, Coherence-improving regularizers. Currently, it
        resorts to supplying the internal artm.Dictionary to the regularizers' constructors whenever possible."""

    def __init__(self, dictionary):
        """
        :param artm.Dictionary dictionary: this object shall be passed as regularizers' constructors argument when possible
        """
        # :param str regularizers_initialization_parameters: this file contains parameters and values to use as defaults for when initializing regularizer object
        self._dictionary = dictionary
        self._regs_data = None
        self._reg_settings = {}
        self._back_t, self._domain_t = [], []
        self._regularizer_type2constructor = \
            {'smooth-phi': lambda x: ArtmRegularizerWrapper.create('smooth-phi', x, self._back_t, [DEFAULT_CLASS_NAME]),
             'smooth-theta': lambda x: ArtmRegularizerWrapper.create('smooth-theta', x, self._back_t),
             'sparse-phi': lambda x: ArtmRegularizerWrapper.create('sparse-phi', x, self._domain_t, [DEFAULT_CLASS_NAME]),
             'sparse-theta': lambda x: ArtmRegularizerWrapper.create('sparse-theta', x, self._domain_t),
             'smooth-phi-dom-cls': lambda x: ArtmRegularizerWrapper.create('smooth-phi', x, self._domain_t, [IDEOLOGY_CLASS_NAME]),
             'smooth-phi-bac-cls': lambda x: ArtmRegularizerWrapper.create('smooth-phi', x, self._back_t, [IDEOLOGY_CLASS_NAME]),
             'smooth-phi-cls': lambda x: ArtmRegularizerWrapper.create('smooth-phi', x, self._back_t + self._domain_t, [IDEOLOGY_CLASS_NAME]),
             'label-regularization-phi-dom-cls': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                 self._domain_t,
                                                                                 dictionary=self._dictionary,
                                                                                 class_ids=IDEOLOGY_CLASS_NAME),
             'decorrelate-phi-dom-def': lambda x: ArtmRegularizerWrapper.create('decorrelate-phi', x, self._domain_t,
                                                                                class_ids=DEFAULT_CLASS_NAME),
             'label-regularization-phi-dom-all': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x, self._domain_t,
                                                                                 dictionary=self._dictionary,
                                                                                 class_ids=None), # targets all classes, since no CLASS_LABELS list is given
             'label-regularization-phi-bac-all': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                         self._back_t,
                                                                                         dictionary=self._dictionary,
                                                                                         class_ids=None),
             'label-regularization-phi-dom-def': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                 self._domain_t,
                                                                                 dictionary=self._dictionary,
                                                                                 class_ids=[DEFAULT_CLASS_NAME]),
             'label-regularization-phi-bac-def': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                         self._back_t,
                                                                                         dictionary=self._dictionary,
                                                                                         class_ids=DEFAULT_CLASS_NAME),

             'label-regularization-phi-bac-cls': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                         self._back_t,
                                                                                         dictionary=self._dictionary,
                                                                                         class_ids=IDEOLOGY_CLASS_NAME),
             'label-regularization-phi-all': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                 self._domain_t + self._back_t,
                                                                                 dictionary=self._dictionary,
                                                                                 class_ids=None),
             'label-regularization-phi-def': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                     self._domain_t + self._back_t,
                                                                                     dictionary=self._dictionary,
                                                                                     class_ids=DEFAULT_CLASS_NAME),
             'label-regularization-phi-cls': lambda x: ArtmRegularizerWrapper.create('label-regularization-phi', x,
                                                                                     self._domain_t + self._back_t,
                                                                                     dictionary=self._dictionary,
                                                                                     class_ids=IDEOLOGY_CLASS_NAME),
             'decorrelate-phi-def': lambda x: ArtmRegularizerWrapper.create('decorrelate-phi-def', x, self._domain_t, class_ids=DEFAULT_CLASS_NAME),
             'decorrelate-phi-class': lambda x: ArtmRegularizerWrapper.create('decorrelate-phi-class', x, self._domain_t,
                                                                            class_ids=IDEOLOGY_CLASS_NAME),
             'decorrelate-phi-background': lambda x: ArtmRegularizerWrapper.create('decorrelate-phi', x, self._back_t,
                                                                               class_ids=None),
             'improve-coherence': lambda x: ArtmRegularizerWrapper.create('improve-coherence', x, self._domain_t, self._dictionary,
                                                                          class_ids=DEFAULT_CLASS_NAME)}
        logger.info("Initialized RegularizersFactory with artm.Dictionary '{}'".format(self._dictionary.name))

    @property
    def regs_data(self):
        return self._regs_data

    @regs_data.setter
    def regs_data(self, regs_data):
        """
        :param RegularizersData regs_data:
        """
        self._regs_data = regs_data
        self._back_t, self._domain_t = regs_data.background_topics, regs_data.domain_topics
        self._reg_settings = regs_data.regularizers_parameters

    def create_reg_wrappers(self, reg_type2name, background_topics, domain_topics, reg_cfg=None):
        """
        Creates a dict: each key is a regularizer type (identical to one of the '_regularizers_section_name2constructor' hash'\n
        :param str or list or dict reg_type2name: indicates which regularizers should be active; eg keys of the 'regularizers' section of the train.cfg
        - If type(reg_type2name) == str: reg_type2name is a file path to a cfg formated file that has a 'regularizers' section indicating the active regularization components.\n
        - If type(reg_type2name) == list: reg_type2name is a list of tuples with each 1st element being the regularizer type (eg 'smooth-phi', 'decorrelate-phi-domain') and each 2nd element being the regularizer unique name.\n
        - If type(reg_type2name) == dict: reg_type2name maps regularizer types to names. regularizer types and regularizer names
        :param list background_topics: a list of the 'background' topic names. Can be empty.
        :param list domain_topics: a list of the 'domain' topic names. Can be empty.
        :param str or dict reg_cfg: contains the values for initializing the regularizers' parameters (eg tau) with. If None then the default file is used
        - If type(reg_cfg) == str: reg_cfg is a file path to a cfg formated file that has as sections regularizer_types with their keys being initialization parameters\n
        - If type(reg_cfg) == dict: reg_cfg maps regularizer types to parameters dict.
        :rtype: RegularizersFactory
        """
        if not reg_cfg:
            reg_cfg = REGULARIZERS_CFG
        self._regs_data = RegularizersData(background_topics, domain_topics, reg_type2name, reg_cfg)
        self._back_t, self._domain_t = background_topics, domain_topics
        self._reg_settings = self._regs_data.regularizers_parameters
        logger.info("Active regs: {}".format(self._regs_data.regs_hash))
        logger.info("Regs default inits cfg file: {}".format(reg_cfg))
        logger.info("Reg settings: {}".format(self._regs_data.regularizers_parameters))
        return [self.construct_reg_wrapper(reg, params) for reg, params in sorted(self._regs_data.regularizers_parameters.items(), key=lambda x: x[0])]

    # def _create_reg_wrappers(self):
    #     """
    #     Call this method to create all possible regularization components for the model.\n
    #      keys; each value is a dictionry of the corresponding's regularizer's initialization parameter; ie:
    #       {\n
    #       'sparse-phi': {'name': 'spp', 'start': 5, 'tau': 'linear_-0.5_-5'},\n
    #       'smooth-phi': {'name': 'smp', 'tau': 1},\n
    #       'sparse-theta': {'name': 'spt', 'start': 3, 'tau': linear_-0.2_-6, 'alpha_iter': 1},\n
    #       'smooth-theta': {'name': 'smt', 'tau': 1, 'alpha_iter': linear_0.5_1}\n
    #       }
    #     :return: the constructed regularizers; objects of type ArtmRegularizerWrapper
    #     :rtype: list
    #     """
    #
        # return list(filter(None, map(lambda x: self.construct_reg_wrapper(x[0], x[1]), sorted(self._reg_settings.items(), key=lambda y: y[0]))))

    def construct_reg_wrapper(self, reg_type, settings):
        """
        :param str reg_type: the regularizer's unique definition, based on reg_type, topics targeted, modality targeted
        :param dict settings: key, values pairs to initialize the regularizer parameters. Must contain 'name' key
        :return: the regularizer's wrapper object reference
        :rtype: ArtmRegularizerWrapper
        """
        if reg_type not in self._regularizer_type2constructor:
            raise RuntimeError("Requested to create '{}' regularizer, which is not supported".format(reg_type))
        if (self._back_t is None or len(self._back_t) == 0) and reg_type.startswith('smooth'):
            logger.warning("Requested to create '{}' regularizer, which normally targets 'bakground' topicts, but there are "
                          "not distinct 'background' topics defined. The constructed regularizer will target all topics instead.".format(reg_type))
        # manually insert the 'long_type' string in the settings hash to use it as the truly unique 'type' of a regularizer
        return self._regularizer_type2constructor[reg_type](dict(settings, **{'long-type': reg_type}))


def _parse_active_regs(regs):
    reg_settings = {'OrderedDict': lambda x: x.items(),
                                                'dict': lambda x: x.items(),
                                                'str': lambda x: cfg2model_settings(x)['regularizers'],
                                                'list': lambda x: dict(_abbreviation(element) for element in x)}
    return dict(reg_settings[type(regs).__name__](regs))  # reg-def, reg-name tuples in a list

def _abbreviation(reg_type):
    if type(reg_type) == str:
        r = reg_type.split('-')
        if len(r) == 1:
            return reg_type, reg_type
        if len(r) == 2:
            return reg_type, ''.join(x[:2] for x in r)
        return reg_type, ''.join(x[0] for x in r)
    elif type(reg_type) == tuple:
        return reg_type
    else:
        raise ValueError("Either input a string representing a regularizer type (immplying that the name should be inferred as an abbreviation) or a tuple holding both the type and the name")


def cfg2regularizer_settings(cfg_file):
    config = ConfigParser()
    config.read(u'{}'.format(cfg_file))
    return OrderedDict([(str(section),
                         OrderedDict([(str(setting_name),
                                       parameter_name2encoder[str(setting_name)](value)) for setting_name, value in config.items(section) if value])
                         ) for section in config.sections()])


def _parse_reg_cfg(regs_config):
    reg_initialization_type2_dict = {'dict': lambda x: x,
                                     'OrderedDict': lambda x: x,
                                     'str': lambda x: cfg2regularizer_settings(x)}
    return reg_initialization_type2_dict[type(regs_config).__name__](regs_config)


def _create_reg_settings(self):
    regs_init_params = {}
    _settings = _parse_reg_cfg(self.reg_cfg)
    for reg_unique_type, reg_name in self.regs_hash.items():
        try:
            regs_init_params[reg_unique_type] = dict(_settings[reg_unique_type], **{'name': reg_name})
        except KeyError:
            raise KeyError("'reg_cfg' {} resulting in settings {}, does not have key '{}'. Probably you forgot to add the corresponding entry in 'train.cfg' and/or 'regularizers.cfg'".format(self.reg_cfg, _settings, reg_unique_type))
                # "Keys in regs_cfg: [{}], keys requested as active regularizers: [{}], current key: {}, name: {}. "
                # "Probably you forgot to add the corresponding entry in 'train.cfg' and/or 'regularizers.cfg'".format(
                #     ', '.join(sorted(self.reg_cfg.keys())),
                #     ', '.join(sorted(self.reg_cfg.keys())),
                #     reg_unique_type, reg_name))
    return regs_init_params


@attr.s(cmp=True, hash=True, slots=False)
class RegularizersData(object):
    background_topics = attr.ib(init=True, converter=list, repr=True, cmp=True)
    domain_topics = attr.ib(init=True, converter=list, repr=True, cmp=True)
    regs_hash = attr.ib(init=True, converter=_parse_active_regs)
    reg_cfg = attr.ib(init=True)
    regularizers_parameters = attr.ib(default=attr.Factory(lambda self: _create_reg_settings(self), takes_self=True), repr=True, cmp=True, hash=True, init=False)
