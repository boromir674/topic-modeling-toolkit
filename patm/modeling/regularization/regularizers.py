import re
import sys
import artm
from collections import OrderedDict
from configparser import ConfigParser

from patm.utils import cfg2model_settings
from patm.definitions import REGULARIZERS_CFG, DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME
from patm.modeling.parameters.trajectory import trajectory_builder
from patm.modeling.parameters.trajectory import ParameterTrajectory

from base_regularizer import *

def cfg2regularizer_settings(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)
    return OrderedDict([(str(section), OrderedDict([(str(setting_name), parameter_name2encoder[str(setting_name)](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])


smooth_sparse_phi = ('tau', 'gamma', 'class_ids', 'topic_names')
smooth_sparse_theta = ('tau', 'topic_names', 'alpha_iter', 'doc_titles', 'doc_topic_coef')
decorrelation = ('tau', 'gamma', 'class_ids', 'topic_names', 'topic_pairs')

regularizer2parameters = {
    'smooth-phi': smooth_sparse_phi,
    'sparse-phi': smooth_sparse_phi,
    'smooth-theta': smooth_sparse_theta,
    'sparse-theta': smooth_sparse_theta,
    'decorrelate-phi-domain': decorrelation,
    'decorrelate-phi-background': decorrelation,
    'label-regularization-phi': ('tau', 'gamma', 'class_ids', 'topic_names'),
    # 'kl-function-info': ('function_type', 'power_value'),
    # 'specified-sparse-phi': ('tau', 'gamma', 'topic_names', 'class_id', 'num_max_elements', 'probability_threshold', 'sparse_by_column'),
    # 'improve-coherence-phi': ('tau', 'gamma', 'class_ids', 'topic_names'),
    # 'smooth-ptdw': ('tau', 'topic_names', 'alpha_iter'),
    # 'topic-selection': ('tau', 'topic_names', 'alpha_iter')
    # the reasonable parameters to consider experiment with setting to different values and to also change between training cycles
}

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
    # 'doc_topic_coef': list, # (list of floats or list of list of floats): of floats len=nb_topics or of lists of floats len=len(doc_titles) and len(inner_list)=nb_topics: Two cases: 1) list of floats with length equal to num of topics. Means additional multiplier in M-step formula besides alpha and tau, unique for each topic, but general for all processing documents. 2) list of lists of floats with outer list length equal to length of doc_titles, and each inner list length equal to num of topics. Means case 1 with unique list of additional multipliers for each document from doc_titles. Other documents will not be regularized according to description of doc_titles parameter. Note, that doc_topic_coef and topic_names are both using.
    # 'num_max_elements': int, # number of elements to save in row/column for the artm.SpecifiedSparsePhiRegularizer
    # 'probability_threshold': float, # if m elements in row/column sum into value >= probability_threshold, m < n => only these elements would be saved. Value should be in (0, 1), default=None
    # 'sparse_by_columns': bool, # find max elements in column or row
}


class RegularizersFactory:
    active_regularizers_type2tuples_enlister = {'OrderedDict': lambda x: x.items(),
                                                'dict': lambda x: x.items(),
                                                  'str': lambda x: cfg2model_settings(x)['regularizers'],
                                                'list': lambda x: x}
    reg_initialization_type2_enlister = {'dict': lambda x: x,
                                         'OrderedDict': lambda x: x,
                                         'str': lambda x: cfg2regularizer_settings(x)}

    def __init__(self):
        self._registry = {}
        self._regularizers_cfg = REGULARIZERS_CFG
        self._reg_settings = cfg2regularizer_settings(self._regularizers_cfg)
        self._reg_defs = {}
        self._wrappers = []
        self._col_passes = 0
        self._back_t, self._domain_t = [], []
        self._regularizer_type2constructor = {'sparse-phi': lambda x: SparsePhiRegularizerWrapper(x['name'], x, self._domain_t, [DEFAULT_CLASS_NAME]),
                                              'smooth-phi': lambda x: SmoothPhiRegularizerWrapper(x['name'], x, self._back_t, [DEFAULT_CLASS_NAME]),
                                              'sparse-theta': lambda x: SparseThetaRegularizerWrapper(x['name'], x, self._domain_t),
                                              'smooth-theta': lambda x: SmoothThetaRegularizerWrapper(x['name'], x, self._back_t)}

    @property
    def collection_passes(self):
        return self._col_passes

    @collection_passes.setter
    def collection_passes(self, collection_passes):
        self._col_passes = collection_passes

    def set_regularizers_definitions(self, train_cfg, background_topics, domain_topics, reg_cfg=None):
        """
        Creates a dict: each key is a regularizer type (identical to one of the '_regularizers_section_name2constructor' hash'\n
        :param str or list or dict train_cfg: indicates which regularizers should be active.
        - If type(train_cfg) == str: train_cfg is a file path to a cfg formated file that has a 'regularizers' section indicating the active regularization components.\n
        - If type(train_cfg) == list: train_cfg is a list of tuples with each 1st element being the regularizer type (eg 'smooth-phi', 'decorrelate-phi-domain') and each 2nd element being the regularizer unique name.\n
        - If type(train_cfg) == dict: train_cfg maps regularizer types to names. regularizer types and regularizer names
        :param str or dict reg_cfg: contains the values for initializing the regularizers with. If None then the default file is used
        - If type(reg_cfg) == str: reg)cfg is a file path to a cfg formated file that has as sections regularizer_types with their keys being initialization parameters\n
        - If type(reg_cfg) == dict: reg_cfg maps regularizer types to parameters dict.
        :rtype: RegularizersFactory
        """
        self._back_t, self._domain_t = background_topics, domain_topics
        reg_types_n_names = self.active_regularizers_type2tuples_enlister[type(train_cfg).__name__](train_cfg)
        if reg_cfg is not None:
            reg_settings_dict =self.reg_initialization_type2_enlister[type(reg_cfg).__name__](reg_cfg)
        else:
            reg_settings_dict = self._reg_settings
        self._reg_defs = {}
        for reg_type, reg_name in reg_types_n_names:
            self._reg_defs[reg_type] = dict(reg_settings_dict[reg_type], **{'name': reg_name})
        return self

    def create_reg_wrappers(self):
        """
        Call this method to create all possible regularization components for the model.\n
         keys; each value is a dictionry of the corresponding's regularizer's initialization parameter; ie:
          {\n
          'sparse-phi': {'name': 'spp', 'start': 5, 'tau': 'linear_-0.5_-5'},\n
          'smooth-phi': {'name': 'smp', 'tau': 1},\n
          'sparse-theta': {'name': 'spt', 'start': 3, 'tau': linear_-0.2_-6, 'alpha_iter': 1},\n
          'smooth-theta': {'name': 'smt', 'tau': 1, 'alpha_iter': linear_0.5_1}\n
          }
        :return: the constructed regularizers; objects of type ArtmRegularizerWrapper
        :rtype: list
        """
        return list(map(lambda x: self._regularizer_type2constructor[x[0]](x[1]), sorted(self._reg_defs.items())))

    # def construct_reg_pool_from_latest_results(self, results):
    #     return [ArtmRegularizerWrapper(reg_type, dict([(attr_name, reg_settings_dict[attr_name]) for attr_name in regularizer2parameters[reg_type]])) for reg_type, reg_settings_dict in results['reg_parameters'][-1][1].items()]

    def construct_reg_wrapper(self, reg_type, settings):
        """
        :param str reg_type: the regularizer's type
        :param dict settings: key, values pairs to initialize the regularizer parameters. Must contain 'name' key
        :return: the regularizer's wrapper object reference
        :rtype: ArtmRegularizerWrapper
        """
        return ArtmRegularizerWrapper(reg_type, settings)


regularizers_factory = RegularizersFactory()


class RegularizerPoolBuilder:
    def __init__(self):
        self._topics = []
        self._nb_topics = 0
        self._pool = []
        self.background_topics_ratio = 0
        self._background_topics = []
        self._domain_topics = []

    def init_pool(self, topics_labels):
        self._topics = topics_labels
        # self._nb_topics = len(selpatm / modeling / regularization / regularizers.pyf._topics)
        self._pool = []
        self._background_topics = []
        self._domain_topics = []
        return self

    def separate_background(self, background_topics_ratio):
        self._pool = []
        self.background_topics_ratio = background_topics_ratio
        _ = int(self.background_topics_ratio * self._nb_topics)
        self._background_topics = self._topics[:_]
        self._domain_topics = self._topics[_:]
        return self

    def add_regularizers(self, reg_type2reg_attributes):
        for reg_type, reg_settings in sorted(reg_type2reg_attributes.items(), key=lambda x: x[1][
            'name']):  # sort by regularizer custom name following same ordering as TopicModel
            self._pool.append(construct_regularizer(reg_type, reg_settings['name'], reg_settings['attributes']))

    # def smooth_background(self, matrix, tau): # ('phi', 'theta'}
    #     self._pool.append(construct_regularizer('smooth-' + matrix, 'sb' + matrix[0], {'topic_names': self._background_topics, 'tau': tau}))
    #     return self
    # def sparse_domain(self, matrix):
    #     # expects to receive a tau trajectory as a paramter later
    #     self._pool.append(construct_regularizer('sparse' + matrix, 'sd' + matrix[0], {'topic_names': self._domain_topics, 'tau': -0.1}))
    #     return self
    # def decorrelate(self):
    #     self._pool.append(construct_regularizer('decorrelate-phi', 'ddt', {'topic_names': self._domain_topics, 'tau': 2e+5}))
    #     return self

    def build_pool(self):
        return self._pool

regularizer_pool_builder = RegularizerPoolBuilder()


# def construct_regularizer(reg_type, name, reg_settings):
#     """
#     Constructs a new artm regularizer object given its type, a name and optional initialization values for its attributes.\n
#     :param str reg_type: the regularizer's type
#     :param str name: the regularizers name
#     :param dict reg_settings: key, values pairs to initialize the regularizer parameters
#     :return: the regularizer's object reference
#     :rtype: artm.regularizers.BaseRegularizer
#     """
#     reg_parameters = {k: v for k, v in reg_settings.items() if v}
#     if reg_type != 'kl-function-info':
#         reg_parameters = dict(reg_parameters, **{'name': name})
#     elif 'name' in reg_settings:
#         reg_parameters = {k: v for k, v in reg_settings.items() if k != 'name'}
#     artm_reg = _regularizers_section_name2constructor[reg_type](**reg_parameters)
#     found = ', '.join(map(lambda x: '{}={}'.format(x[0], x[1]), reg_parameters.items()))
#     not_found = ', '.join(map(lambda x: '{}={}'.format(x[0], x[1]), {k: v for k, v in reg_settings.items() if not v}.items()))
#     print 'Constructed reg: {}: set params: ({}); using defaults: ({})'.format(reg_type+'.'+name, found, not_found)
#     return artm_reg
