import sys
from abc import ABCMeta
from collections import defaultdict
from functools import reduce


class FitnessValue(object):
    def __init__(self, value):
        self.value = value
    def __abs__(self):
        return abs(self.value)
    def __eq__(self, other):
        return self.value == other.value
    def __ne__(self, other):
        return not self.__ne__(other)
    def __str__(self):
        return "{:.2f}".format(self.value)
    def __repr__(self):
        return "{:.2f}".format(self.value)
class NaturalFitnessValue(FitnessValue):
    def __init__(self, value):
        super(NaturalFitnessValue, self).__init__(value)
    def __lt__(self, other):
        return self.value < other.value
    def __le__(self, other):
        return self.value <= other.value
    def __gt__(self, other):
        return self.value > other.value
    def __ge__(self, other):
        return self.value >= other.value
class ReversedFitnessValue(FitnessValue):
    def __init__(self, value):
        super(ReversedFitnessValue, self).__init__(value)
    def __lt__(self, other):
        return self.value > other.value
    def __le__(self, other):
        return self.value >= other.value
    def __gt__(self, other):
        return self.value < other.value
    def __ge__(self, other):
        return self.value <= other.value


_ORDERING_HASH = defaultdict(lambda: 'natural', perplexity='reversed')
_INITIAL_BEST = defaultdict(lambda: 0, perplexity=float('inf'))
_FITNESS_VALUE_CONSTRUCTORS_HASH = {'natural': NaturalFitnessValue, 'reversed': ReversedFitnessValue}


class FitnessFunctionBuilder(object):
    def __init__(self):
        self._column_definitions = []
        self._extractors = []
        self._coeff_values = []
        self._order = ''
        self._names = []

    def _create_extractor(self, column_definition):
        return lambda x: x[self._column_definitions.index(column_definition)]

    def start(self, column_definitions, ordering='natural'):
        self._column_definitions = column_definitions
        self._extractors = []
        self._coeff_values = []
        self._order = ordering
        self._names = []
        return self

    def coefficient(self, name, value):
        assert name in self._column_definitions
        self._extractors.append(self._create_extractor(name))
        self._coeff_values.append(value)
        self._names.append(name)
        return self

    def build(self):
        return FitnessFunction(self._extractors, self._coeff_values, self._names, ordering=self._order)

function_builder = FitnessFunctionBuilder()


class FitnessFunction(object):
    def __init__(self, extractors, coefficients, names, ordering='natural'):
        self._extr = extractors
        self._coeff = coefficients
        self._names = names
        self._order = ordering
        assert ordering in ('natural', 'reversed')
        assert abs(sum(map(abs, self._coeff))) - 1 < 1e-6

    @classmethod
    def single_metric(cls, metric_definition):
        return function_builder.start([metric_definition], ordering=_ORDERING_HASH[metric_definition]).coefficient(metric_definition, 1).build()

    @property
    def ordering(self):
        return self._order

    def compute(self, individual):
        """
        :param list individual: list of values for metrics [prpl, top-tokens-coh-10, ..]
        :return:
        """
        try:
            c = [x(individual) for x in self._extr]
        except IndexError as e:
            raise IndexError("Error: {}. Current builder column definitions: [{}], Input: [{}]".format(e, ', '.join(str(_) for _ in sorted(function_builder._column_definitions)),
                                                                                                       ', '.join(str(_) for _ in sorted(individual))))
        c1 = [self._wrap(x) for x in c]
        c2 = [self._coeff[i] * x for i,x in enumerate(c1)]
        c3 = reduce(lambda i,j: i+j, c2)
        c4 = _FITNESS_VALUE_CONSTRUCTORS_HASH[self._order]
        r = c4(c3)
        return r
        # return _FITNESS_VALUE_CONSTRUCTORS_HASH[self._order](
        #     reduce(lambda i, j: i + j, [x[0] * self._wrap(x[1](individual)) for x in zip(self._coeff, self._extr)]))

    def _wrap(self, value):
        if value is None:
            return {'natural': 0, 'reversed': float('inf')}[self._order]
        return value

    def __str__(self):
        return ' + '.join(['{}*{}'.format(x[0], x[1]) for x in zip(self._names, self._coeff)])


class FitnessCalculator(object):
    def __new__(cls, *args, **kwargs):
        x = super(FitnessCalculator, cls).__new__(cls)
        x._func = None
        x._column_defs, x._best = [], []
        x._highlightable_columns = []
        return x

    def __init__(self, single_metric=None, column_definitions=None):
        if type(single_metric) == str and type(column_definitions) == list:
        # FitnessFunctionBuilder.start(column_definitions).coefficient(single_metric, 1).build()
            self._func = function_builder.start(column_definitions, ordering=_ORDERING_HASH[single_metric]).coefficient(single_metric, 1).build()
            assert isinstance(self._func, FitnessFunction)
            self._column_defs = column_definitions

    # def initialize(self, fitness_function, column_definitions):
    #     """
    #     :param FitnessFunction fitness_function:
    #     :param list column_definitions: i.e. ['perplexity', 'kernel-coherence-0.8', 'kernel-contrast-0.8', 'top-tokens-coherence-10', 'top-tokens-coherence-100']
    #     """
    #     assert isinstance(fitness_function, FitnessFunction)
    #     self.function = fitness_function
    #     self._column_defs = column_definitions

    @property
    def highlightable_columns(self):
        return self._highlightable_columns

    @highlightable_columns.setter
    def highlightable_columns(self, column_definitions):
        self._highlightable_columns = column_definitions
        self._best = dict([(x, _INITIAL_BEST[FitnessCalculator._get_column_key(x)]) for x in column_definitions])

    @property
    def best(self):
        return self._best

    @property
    def function(self):
        return self._func

    # @function.setter
    # def function(self, a_fitness_function):
    #     self._func = a_fitness_function

    def pass_vector(self, values_vector):
        self._update_best(values_vector)
        return values_vector

    def compute_fitness(self, values_vector):
        self._update_best(values_vector)
        return self._func.compute(values_vector)

    def _update_best(self, values_vector):
        self._best.update([(column_def, value) for column_key, column_def, value in
                           [(FitnessCalculator._get_column_key(x[0]), x[0], x[1]) for x in zip(self._column_defs, values_vector)]
                           if column_def in self._best and
                           FitnessCalculator._fitness(column_key, value) > FitnessCalculator._fitness(column_key, self._best[column_def])])

    def __call__(self, *args, **kwargs):
        return self.compute_fitness(args[0])

    @staticmethod
    def _fitness(column_key, value):
        if value is None:
            return _FITNESS_VALUE_CONSTRUCTORS_HASH[_ORDERING_HASH[column_key]](_INITIAL_BEST[column_key])
        return _FITNESS_VALUE_CONSTRUCTORS_HASH[_ORDERING_HASH[column_key]](value)

    @staticmethod
    def _get_column_key(column_definition):
        return '-'.join([_f for _f in map(FitnessCalculator._get_token, column_definition.split('-')) if _f])
        # return '-'.join([_f for _f in [FitnessCalculator._get_token(x) for x in column_definition.split('-')] if _f])

    @staticmethod
    def _get_token(definition_element):
        try:
            _ = float(definition_element)
            return None
        except ValueError:
            if definition_element[0] == '@' or len(definition_element) == 1:
                return None
            return definition_element

    # def _query_vector(self, values_vector, column_key):
    #     """Call this method to get a list of tuples; 1st elements are the  vactor values, 2nd elements are the corresponding column definitions"""
    #     return map(lambda x: (values_vector[x[0]], x[1]), [_ for _ in enumerate(self._column_defs) if _[1].startswith(column_key)])
