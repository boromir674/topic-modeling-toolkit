import sys
import warnings
from collections import OrderedDict, Counter
from abc import ABCMeta, abstractmethod, abstractproperty


class MetaParameterMixture(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def static_parameters(self):
        raise NotImplemented

    @abstractproperty
    def explorable_parameters(self):
        raise NotImplemented

    def __str__(self):
        return type(self).__name__


class StaticAndExplorableMixture(MetaParameterMixture):

    def __init__(self, parameters):
        assert len(parameters) > 0
        self._static, self._explorable = OrderedDict(), OrderedDict()
        self._nb_combinations = 1
        for k, v in parameters.items():
            self._parse(k, v)

    def _parse(self, k, v):
        # print 'K', k, 'TYPE:', type(v)
        if isinstance(v, StaticAndExplorableMixture):
            self._assimilate_mixture(v)
        elif type(v) in (str, int, float):
            self._static[k] = v
        elif type(v) in (list, tuple):
            assert len(v) > 0
            if len(v) == 1:
                self._static[k] = v[0]
            else:  # len(v) > 1
                self._explorable[k] = list(v)
                self._nb_combinations *= len(v)
        else:
            raise ValueError("Parsing of type '{}' is not supported".format(type(v)))

    def __len__(self):
        return self._nb_combinations

    def _assimilate_mixture(self, mixture):
        for i, v in mixture.static_parameters.items():
            self._static[mixture.name + '.' + i] = v
        for i, v in mixture.explorable_parameters.items():
            self._explorable[mixture.name + '.' + i] = v
        self._nb_combinations *= len(mixture)

    def valid_trajectory_defs(self):
        """
        Call this method to get a list of strings (ie ['sparse_phi', 'sparse_theta']) corresponding to the valid tau
        coefficient trajectory definitions found withing the static and explorable parameters
        """
        res = OrderedDict()
        for k, v in self._static.items() + self._explorable.items():
            if k.startswith('sparse_'):
                nk = k.split('.')[0]
                if nk not in res:
                    res[nk] = OrderedDict()
                res[nk][k.split('.')[1]] = v
        return OrderedDict([(k, v) for k, v in res.items() if all(map(lambda x: x in v, ('deactivate', 'kind', 'start', 'end')))])

    @property
    def parameter_spans(self):
        return self._explorable.values()

    @property
    def static_parameters(self):
        return self._static

    @property
    def explorable_parameters(self):
        return self._explorable

    def __getattr__(self, item):
        if item in self._static:
            return self._static[item]
        if item in self._explorable:
            return self._explorable[item]
        if item == 'default_class_weight':
            return 1.0
        raise AttributeError("Attribute '{}' is not found in object".format(item))

class TunerDefinition(StaticAndExplorableMixture):

    def __init__(self, parameters):
        super(TunerDefinition, self).__init__(parameters)


class SparseTauTrajectoryDefinition(StaticAndExplorableMixture):
    def __init__(self, params, name):
        super(SparseTauTrajectoryDefinition, self).__init__(params)
        self.name = name


class ParameterMixtureBuilder(object):

    def __init__(self):
        self._params = OrderedDict()

    def initialize(self, *args):
        self._params = OrderedDict()

    def _set_return(self, key, value):
        self._params[key] = value
        return self


class TunerDefinitionBuilder(ParameterMixtureBuilder):
    def __init__(self):
        super(TunerDefinitionBuilder, self).__init__()
        self._trajectory_builder = TrajectoryParametersBuilder(self)

    def initialize(self):
        super(TunerDefinitionBuilder, self).initialize()
        return self

    def nb_topics(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def collection_passes(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def document_passes(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def background_topics_pct(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def default_class_weight(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def ideology_class_weight(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def decorrelate_all_tau(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)
    def decorrelate_def_tau(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)
    def decorrelate_class_tau(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def sparse_phi(self):
        k = sys._getframe().f_code.co_name
        self._trajectory_builder.initialize(k)
        return self._trajectory_builder

    def sparse_theta(self):
        k = sys._getframe().f_code.co_name
        self._trajectory_builder.initialize(k)
        return self._trajectory_builder

    def build(self):
        return TunerDefinition(OrderedDict(self._params))


class TrajectoryParametersBuilder(ParameterMixtureBuilder):

    def __init__(self, master_builder):
        """
        :param TunerDefinitionBuilder master_builder:
        """
        super(TrajectoryParametersBuilder, self).__init__()
        self._master_builder = master_builder

    def initialize(self, param_set_key):
        self._reg_name = param_set_key
        super(TrajectoryParametersBuilder, self).initialize()

    def deactivate(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def kind(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def start(self, *value):
        return self._set_return(sys._getframe().f_code.co_name, value)

    def end(self, *value):
        self._params[sys._getframe().f_code.co_name] = value
        self.build()
        return self._master_builder

    def build(self):
        self._master_builder._params[self._reg_name] = SparseTauTrajectoryDefinition(self._params, self._reg_name)

tuner_definition_builder = TunerDefinitionBuilder()


class RegularizersActivationDefinitionBuilder(object):
    def __init__(self, tuner=None):
        self._tuner = tuner
        self._reg_types = []
        self._key = None

    @property
    def activate(self):
        self._reg_types = []
        self._key = None
        return self

    @property
    def smoothing(self):
        self._key = 'smooth'
        return self

    @property
    def sparsing(self):
        self._key = 'sparse'
        return self

    @property
    def label_regularization(self):
        """DxWxTxC"""
        self._reg_types.append('label-regularization-phi')
        return self

    @property
    def phi(self):
        self._reg_types.append(self._key + '-phi')
        return self

    @property
    def theta(self):
        self._reg_types.append(self._key + '-theta')
        return self

    @property
    def decorrelate_phi_all(self):  # target all modalities
        self._reg_types.append('decorrelate-phi')
        return self

    @property
    def decorrelate_phi_def(self):  # target @default_class modality only
        self._reg_types.append('decorrelate-phi-def')
        return self

    @property
    def decorrelate_phi_class(self):  # target @ideology_class modality only
        self._reg_types.append('decorrelate-phi-class')
        return self

    @property
    def improve_coherence_phi(self):
        self._reg_types.append('improve-coherence')
        return self

    def build(self):
        if len(set(self._reg_types)) != len(self._reg_types):
            warnings.warn("Duplicate regularizers [{}] reguested for activation. Will activate once the uniques.".format(', '.join([item for item, count in Counter(self._reg_types).items() if count > 1])))
        seen = set()
        return [x for x in self._reg_types if x not in seen and not seen.add(x)]

    def done(self):
        if self._tuner:
            self._tuner.active_regularizers = self.build()


if __name__ == '__main__':
    tdb = TunerDefinitionBuilder()

    radb = RegularizersActivationDefinitionBuilder()
    activated_regularizer_types = radb.activate.smoothing.theta.phi.sparsing.phi.build()

    print 'REGS', activated_regularizer_types
    #
    #
    # d1 = tdb.initialize()._nb_topics(20).collection_passes(100).document_passes(1).background_topics_pct(0.1).\
    #     ideology_class_weight(5).sparse_phi().deactivate(10).kind('quadratic').start(-1).end(-10).build()
    #
    # tuner_definition = tdb.initialize()._nb_topics([20, 40]).collection_passes(100).document_passes(1).background_topics_pct(0.1). \
    #     ideology_class_weight(5).sparse_phi().deactivate(10).kind(['quadratic', 'cubic']).start(-1).end([-10, -20, -30]).\
    #     sparse_theta().deactivate(10).kind(['quadratic', 'cubic']).start([-1, -2, -3]).end(-10).build()
    #
    # print 'LENS', len(d1), len(tuner_definition)
    # print 'D1 static', d1.static_parameters
    # print 'D2 static', tuner_definition.static_parameters
    # print 'D1 expl', d1.explorable_parameters
    # print 'D2 expl', tuner_definition.explorable_parameters
