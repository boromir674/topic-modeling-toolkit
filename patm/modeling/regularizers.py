import sys
from collections import OrderedDict
from configparser import ConfigParser
import artm


regularizer2parameters = {
    'kl-function-info': ('function_type', 'power_value'),
    'smooth-sparse-phi': ('tau', 'gamma', 'class_ids', 'topic_names'),
    'smooth-sparse-theta': ('tau', 'topic_names', 'alpha_iter', 'doc_titles', 'doc_topic_coef'),
    'decorrelator-phi': ('tau', 'gamma', 'class_ids', 'topic_names', 'topic_pairs'),
    'label-regularization-phi': ('tau', 'gamma', 'class_ids', 'topic_names'),
    'specified-sparse-phi': ('tau', 'gamma', 'topic_names', 'class_id', 'num_max_elements', 'probability_threshold', 'sparse_by_column'),
    'improve-coherence-phi': ('tau', 'gamma', 'class_ids', 'topic_names'),
    'smooth-ptdw': ('tau', 'topic_names', 'alpha_iter'),
    'topic-selection': ('tau', 'topic_names', 'alpha_iter')
    # the reasonable parameters to consider experiment with setting to different values and to also change between training cycles
}

reg_type2constructor = {
        'kl-function-info': artm.KlFunctionInfo,
        'smooth-sparse-phi': artm.SmoothSparsePhiRegularizer,
        'smooth-sparse-theta': artm.SmoothSparseThetaRegularizer,
        'decorrelator-phi': artm.DecorrelatorPhiRegularizer,
        'label-regularization-phi': artm.LabelRegularizationPhiRegularizer,
        'specified-sparse-phi': artm.SpecifiedSparsePhiRegularizer,
        'improve-coherence-phi': artm.ImproveCoherencePhiRegularizer,
        'smooth-ptdw': artm.SmoothPtdwRegularizer,
        'topic-selection': artm.TopicSelectionThetaRegularizer
    # There are a few more regularizers implemented in the BigARTM library
    # artm.BitermsPhiRegularizer(name=None, tau=1.0, gamma=None, class_ids=None, topic_names=None, dictionary=None, config=None)
    }

reg_class_string2reg_type = {constructor.__name__: reg_type for reg_type, constructor in reg_type2constructor.items()}


parameter_name2encoder = {
    'tau': float, # the coefficient of regularization for this regularizer
    'gamma': float, # the coefficient of relative regularization for this regularizer
    'class_ids': list, # list of class_ids or single class_id to regularize, will regularize all classes if empty or None [ONLY for ImproveCoherencePhiRegularizer: dictionary should contain pairwise tokens co-occurrence info]
    'topic_names': list, # specific topics to target for applying regularization, Targets all if None
    'topic_pairs': dict, # key=topic_name, value=dict: pairwise topic decorrelation coefficients, if None all values equal to 1.0
    'function_type': str, # the type of function, 'log' (logarithm) or 'pol' (polynomial)
    'power_value': float, # the float power of polynomial, ignored if type = 'log'
    'alpha_iter': list, # list of additional coefficients of regularization on each iteration over document. Should have length equal to model.num_document_passes (of an artm.ARTM model object)
    'doc_titles': list, # of strings: list of titles of documents to be processed by this regularizer. Default empty value means processing of all documents. User should guarantee the existence and correctness of document titles in batches (e.g. in src files with data, like WV).
    'doc_topic_coef': list, # (list of floats or list of list of floats): of floats len=nb_topics or of lists of floats len=len(doc_titles) and len(inner_list)=nb_topics: Two cases: 1) list of floats with length equal to num of topics. Means additional multiplier in M-step formula besides alpha and tau, unique for each topic, but general for all processing documents. 2) list of lists of floats with outer list length equal to length of doc_titles, and each inner list length equal to num of topics. Means case 1 with unique list of additional multipliers for each document from doc_titles. Other documents will not be regularized according to description of doc_titles parameter. Note, that doc_topic_coef and topic_names are both using.
    'num_max_elements': int, # number of elements to save in row/column for the artm.SpecifiedSparsePhiRegularizer
    'probability_threshold': float, # if m elements in row/column sum into value >= probability_threshold, m < n => only these elements would be saved. Value should be in (0, 1), default=None
    'sparse_by_columns': bool, # find max elements in column or row
}


def _construct_regularizer(reg_type, name, reg_settings):
    if reg_type == 'kl-function-info':
        return reg_type2constructor[reg_type](**reg_settings)
    # print 'Adding regularizer', reg_type, '\n'
    d = dict(reg_settings, **{'name': name})
    return reg_type2constructor[reg_type](**d)


def init_from_file(type_names_list, reg_config):
    """
    Initializes a set of regularizers.\n
    :param type_names_list: the desired regularizers to initialize. Elements should correspond to pairs of 'sections' in the input cfg file and names; eg: [('smooth-sparse-theta', 'sst'), ('decorrelator-phi', 'dp')]
    :type type_names_list: list of tuples
    :param reg_config: a cfg file having as sections the type of regularizers supported. Each section attributes can be optionally set to values (if unset inits with defaults)
    :type reg_config: str
    :return: a list of instantiated regularizer objects,
    :rtype: list
    """
    reg_settings = cfg2regularizer_settings(reg_config)
    regs = []
    for reg_type, name in type_names_list:
        try:
            # print "reg settings:", reg_settings[reg_type]
            reg = _construct_regularizer(reg_type, name, reg_settings[reg_type])
            print 'constructed regularizer of type', type(reg), 'with name', reg.name
            regs.append(reg)
        except RuntimeError as e:
            print '\n', reg_type, '\n'
            print e
    return regs

def init_from_latest_results(results):
    regs = []
    for reg_type, reg_settings_dict in results['reg_parameters'][-1]:
        regs.append(_construct_regularizer(reg_type,
                                           reg_settings_dict['name'],
                                           dict([(attr_name, reg_settings_dict[attr_name]) for attr_name in regularizer2parameters[reg_type]])))

def cfg2regularizer_settings(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)
    for sec in config.sections():
        print str(sec)
        for par in config[sec].items():
            print '  {}: {}'.format(str(par[0]), str(par[1]))
    return OrderedDict([(str(section), OrderedDict([(str(setting_name), parameter_name2encoder[str(setting_name)](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])


class RegularizerW(object):
    def __init__(self, instance):
        self.instance = instance
        self._global_wealth = 10.0
        self._observers = []

    # def __getattr__(self, name):
    #     print 'debug:', name
    #     try:
    #         return getattr(self.instance, name)
    #     except KeyError:
    #         msg = "'{0}' object has no attribute '{1}'"
    #         sys.exit(1)
    #         raise AttributeError(msg.format(type(self).__name__, name))

    # def __setattr__(self, key, value):
    #     self.instance.__setattr__(key, value)

    @property
    def global_wealth(self):
        return self._global_wealth

    @global_wealth.setter
    def global_wealth(self, value):
        self._global_wealth = value
        for callback in self._observers:
            print('announcing change')
            callback(self._global_wealth)

    def bind_to(self, callback):
        print('bound')
        self._observers.append(callback)


if __name__ == '__main__':
    reg_inst = artm.SmoothSparsePhiRegularizer(name='gav', tau=0.5)
    print reg_inst.name
    print reg_inst.tau
    print reg_inst.gamma

    res = init_from_file([('smooth-sparse-phi', 'ssp'), ('smooth-sparse-theta', 'sst')], '/data/thesis/code/regularizers.cfg')

    print res, '\n'

    print 'nb regularizers initialized:', len(res), '\n'

    regw1 = RegularizerW(res[0])
    print regw1.instance, type(regw1.instance)

    print regw1.instance.name
    regw1.instance.__setattr__('tau', -1)
    print regw1.instance.name