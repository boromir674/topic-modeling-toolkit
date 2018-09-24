import re
import sys
import artm
from collections import OrderedDict
from configparser import ConfigParser

from ..utils import cfg2model_settings
from ..definitions import REGULARIZERS_CFG


smooth_sparse_phi = ('tau', 'gamma', 'class_ids', 'topic_names')
smooth_sparse_theta = ('tau', 'topic_names', 'alpha_iter', 'doc_titles', 'doc_topic_coef')

regularizer2parameters = {
    'kl-function-info': ('function_type', 'power_value'),
    'smooth-phi': smooth_sparse_phi,
    'sparse-phi': smooth_sparse_phi,
    'smooth-theta': smooth_sparse_theta,
    'sparse-theta': smooth_sparse_theta,
    'decorrelator-phi': ('tau', 'gamma', 'class_ids', 'topic_names', 'topic_pairs'),
    'label-regularization-phi': ('tau', 'gamma', 'class_ids', 'topic_names'),
    'specified-sparse-phi': ('tau', 'gamma', 'topic_names', 'class_id', 'num_max_elements', 'probability_threshold', 'sparse_by_column'),
    'improve-coherence-phi': ('tau', 'gamma', 'class_ids', 'topic_names'),
    'smooth-ptdw': ('tau', 'topic_names', 'alpha_iter'),
    'topic-selection': ('tau', 'topic_names', 'alpha_iter')
    # the reasonable parameters to consider experiment with setting to different values and to also change between training cycles
}

parameter_name2encoder = {
    'tau': float, # the coefficient of regularization for this regularizer
    'gamma': float, # the coefficient of relative regularization for this regularizer
    'class_ids': list, # list of class_ids or single class_id to regularize, will regularize all classes if empty or None [ONLY for ImproveCoherencePhiRegularizer: dictionary should contain pairwise tokens co-occurrence info]
    'topic_names': list, # specific topics to target for applying regularization, Targets all if None
    'topic_pairs': dict, # key=topic_name, value=dict: pairwise topic decorrelation coefficients, if None all values equal to 1.0
    'function_type': str, # the type of function, 'log' (logarithm) or 'pol' (polynomial)
    'power_value': float, # the float power of polynomial, ignored if 'function_type' = 'log'
    'alpha_iter': list, # list of additional coefficients of regularization on each iteration over document. Should have length equal to model.num_document_passes (of an artm.ARTM model object)
    'doc_titles': list, # of strings: list of titles of documents to be processed by this regularizer. Default empty value means processing of all documents. User should guarantee the existence and correctness of document titles in batches (e.g. in src files with data, like WV).
    'doc_topic_coef': list, # (list of floats or list of list of floats): of floats len=nb_topics or of lists of floats len=len(doc_titles) and len(inner_list)=nb_topics: Two cases: 1) list of floats with length equal to num of topics. Means additional multiplier in M-step formula besides alpha and tau, unique for each topic, but general for all processing documents. 2) list of lists of floats with outer list length equal to length of doc_titles, and each inner list length equal to num of topics. Means case 1 with unique list of additional multipliers for each document from doc_titles. Other documents will not be regularized according to description of doc_titles parameter. Note, that doc_topic_coef and topic_names are both using.
    'num_max_elements': int, # number of elements to save in row/column for the artm.SpecifiedSparsePhiRegularizer
    'probability_threshold': float, # if m elements in row/column sum into value >= probability_threshold, m < n => only these elements would be saved. Value should be in (0, 1), default=None
    'sparse_by_columns': bool, # find max elements in column or row
}

reg_type2constructor = {
    'kl-function-info': artm.KlFunctionInfo,
    'sparse-phi': artm.SmoothSparsePhiRegularizer,
    'smooth-phi': artm.SmoothSparsePhiRegularizer,
    'sparse-theta': artm.SmoothSparseThetaRegularizer,
    'smooth-theta': artm.SmoothSparseThetaRegularizer,
    'decorrelator-phi': artm.DecorrelatorPhiRegularizer,
    'label-regularization-phi': artm.LabelRegularizationPhiRegularizer,
    'specified-sparse-phi': artm.SpecifiedSparsePhiRegularizer,
    'improve-coherence-phi': artm.ImproveCoherencePhiRegularizer,
    'smooth-ptdw': artm.SmoothPtdwRegularizer,
    'topic-selection': artm.TopicSelectionThetaRegularizer
}
# regularizer_class_string2reg_type = {re.match("^<class 'artm\.regularizers\.(\w+)'>$", str(constructor)).group(1): reg_type for reg_type, constructor in reg_type2constructor.items()}


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
        self._nb_topics = len(self._topics)
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
        for reg_type, reg_settings in sorted(reg_type2reg_attributes.items(), key=lambda x: x[1]['name']): #sort by regularizer custom name following same ordering as TopicModel
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


class RegularizersFactory:
    def __init__(self, regularizers_cfg=REGULARIZERS_CFG):
        self._registry = {}
        self._regularizers_cfg = regularizers_cfg
        self._reg_settings = cfg2regularizer_settings(self._regularizers_cfg)

    def get_type(self, regularizer_obj):
        return self._registry.get(regularizer_obj, None)

    def construct_reg_obj(self, reg_type, name, settings):
        """
        :param str reg_type: the regularizer's type
        :param str name: the regularizers name
        :param dict settings: key, values pairs to initialize the regularizer parameters
        :return: the regularizer's object reference
        :rtype: artm.regularizers.BaseRegularizer
        """
        _ = construct_regularizer(reg_type, name, settings)
        self._registry[_] = reg_type
        return _

    def construct_pool_from_file(self, type_names_list, reg_config=None):
        """
        Initializes a set of regularizers given the type of regularizers desired to create, a custom name to use and a static configuration file containing initialization settings\n
        :param list of tuples type_names_list: the desired regularizers to initialize. Elements should correspond to pairs of 'sections' in the input cfg file and names; eg: [('smooth-theta', 'sst'), ('decorrelator-phi', 'dp')]
        :param str reg_config: a config file containing values to initialize the regularizer parameters; it has as sections the type of regularizers supported. Each section's attributes in the file can be optionally set to values (if unset inits with defaults)
        :return: a list of instantiated regularizer objects,
        :rtype: list
        """
        self._set_reg_settings(reg_config)
        return [self.construct_reg_obj(reg_type, name, self.reg_settings[reg_type]) for reg_type, name in type_names_list]

    def construct_reg_pool_from_files(self, train_cfg, reg_cfg=self._regularizers_cfg):
        train_settings = cfg2model_settings(train_cfg)
        self._set_reg_settings(reg_cfg)
        return [self.construct_reg_obj(reg_type, name, self.reg_settings[reg_type]) for reg_type, name in train_settings['regularizers'].items()]

    def get_reg_pool_from_latest_results(self, results):
        return [self.construct_reg_obj(reg_type, reg_settings_dict['name'], dict([(attr_name, reg_settings_dict[attr_name]) for attr_name in regularizer2parameters[reg_type]])) for reg_type, reg_settings_dict in results['reg_parameters'][-1][1].items()]

    def _set_reg_settings(self, a_train_cfg):
        if a_train_cfg:
            self._reg_settings = cfg2regularizer_settings(a_train_cfg)

regularizers_factory = RegularizersFactory()


def construct_regularizer(reg_type, name, reg_settings):
    """
    Constructs a new artm regularizer object given its type, a name and optional initialization values for its attributes.\n
    :param str reg_type: the regularizer's type
    :param str name: the regularizers name
    :param dict reg_settings: key, values pairs to initialize the regularizer parameters
    :return: the regularizer's object reference
    :rtype: artm.regularizers.BaseRegularizer
    """
    reg_parameters = {k: v for k, v in reg_settings.items() if v}
    if reg_type != 'kl-function-info':
        reg_parameters = dict(reg_parameters, **{'name': name})
    elif 'name' in reg_settings:
        reg_parameters = {k: v for k, v in reg_settings.items() if k != 'name'}
    artm_reg = reg_type2constructor[reg_type](**reg_parameters)
    # found = ', '.join(map(lambda x: '{}={}'.format(x[0], x[1]), reg_parameters.items()))
    # not_found = ', '.join(map(lambda x: '{}={}'.format(x[0], x[1]), {k: v for k, v in reg_settings.items() if not v}.items()))
    # print 'Constructed reg: {}: set params: ({}); using defaults: ({})'.format(reg_type+'.'+name, found, not_found)
    return artm_reg


def cfg2regularizer_settings(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)
    return OrderedDict([(str(section), OrderedDict([(str(setting_name), parameter_name2encoder[str(setting_name)](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])


if __name__ == '__main__':
    pass
