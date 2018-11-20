import abc
from collections import defaultdict

class FitnessValue(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, value):
        self.value = value
    def __abs__(self):
        return abs(self.value)
    def __eq__(self, other):
        return self.value == other.value
    def __ne__(self, other):
        return not self.__ne__(other)
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

_FITNESS_VALUE_CONSTRUCTORS_HASH = {'natural': NaturalFitnessValue, 'reversed': ReversedFitnessValue}


class FitnessFunction:
    def __init__(self, extractors, coefficients, names, ordering='natural'):
        self._extr = extractors
        self._coeff = coefficients
        self._names = names
        self._order = ordering
        assert ordering in ('natural', 'reversed')
        assert sum(map(lambda x: abs(x), self._coeff)) - 1 < abs(1e-6)

    @property
    def ordering(self):
        return self._order

    def compute(self, individual):
        return _ORDERING_HASH[self._order](reduce(lambda i, j: i + j, map(lambda x: x[0] * x[1](individual), zip(self._coeff, self._extr))))

    def __str__(self):
        return ' + '.join(map(lambda x: '{}*{}'.format(x[0], x[1]), zip(self._names, self._coeff)))


class FitnessFunctionBuilder:
    def __init__(self):
        self._column_definitions = []
        self._extractors = []
        self._coeff_values = []
        self._order = ''
        self._names = []

    def _create_extractor(self, column_definition):
        return lambda x: x[self._column_definitions.index(column_definition)]

    def new(self, column_definitions, ordering='natural'):
        self._column_definitions = column_definitions
        self._extractors = []
        self._coeff_values = []
        self._order = ordering
        self._names = []
        return self
    def coefficient(self, name, value):
        self._extractors.append(self._create_extractor(name))
        self._coeff_values.append(value)
        self._names.append(name)
        return self
    def build(self):
        return FitnessFunction(self._extractors, self._coeff_values, self._names, ordering=self._order)


class FitnessFunctionFactory(object):
    def __init__(self):
        self._fitness_function_builder = FitnessFunctionBuilder()

    def get_single_metric_function(self, column_definitions, name):
        return self._fitness_function_builder.new(column_definitions, ordering=_ORDERING_HASH[_get_column_key(name)]).coefficient(name, 1).build()

    # def linear_combination_function(self, function_expression):
    #     pass
        # FUTURE WORK to support custom linnear combination of evaluation metrics


################################

class FitnessCalculator(object):
    def __init__(self):
        self._func = None
        self._column_defs, self._best = [], []
        self._initial_best = defaultdict(lambda: 0, perplexity=float('inf'))
        self._highlightable_columns = []

    def initialize(self, fitness_function, column_definitions):
        """
        :param FitnessFunction fitness_function:
        :param list column_definitions: i.e. ['perplexity', 'kernel-coherence-0.8', 'kernel-contrast-0.8', 'top-tokens-coherence-10', 'top-tokens-coherence-100']
        """
        assert isinstance(fitness_function, FitnessFunction)
        self.function = fitness_function
        self._column_defs = column_definitions

    @property
    def highlightable_columns(self):
        return self._highlightable_columns

    @highlightable_columns.setter
    def highlightable_columns(self, column_definitions):
        self._highlightable_columns = column_definitions
        self._best = dict(map(lambda x: (x, self._initial_best[_get_column_key(x)]), column_definitions))

    @property
    def best(self):
        return self._best

    @property
    def function(self):
        return self._func

    @function.setter
    def function(self, a_fitness_function):
        self._func = a_fitness_function

    def pass_vector(self, values_vector):
        self._update_best(values_vector)
        return values_vector

    def compute_fitness(self, values_vector):
        self._update_best(values_vector)
        return self._func.compute(values_vector)

    def _update_best(self, values_vector):
        self._best.update([(column_def, value) for column_key, column_def, value in
                           map(lambda x: (_get_column_key(x[0]), x[0], x[1]), zip(self._column_defs, values_vector))
                           if self._get_value(column_key, value) > self._get_value(column_key, self._best[column_def])])

    @staticmethod
    def _get_value(column_key, value):
        return _FITNESS_VALUE_CONSTRUCTORS_HASH[_ORDERING_HASH[column_key]](value)

    # def _query_vector(self, values_vector, column_key):
    #     """Call this method to get a list of tuples; 1st elements are the  vactor values, 2nd elements are the corresponding column definitions"""
    #     return map(lambda x: (values_vector[x[0]], x[1]), [_ for _ in enumerate(self._column_defs) if _[1].startswith(column_key)])

def _get_column_key(column_definition):
    return '-'.join(filter(None, map(lambda x: _get_token(x), column_definition.split('-'))))


def _get_token(definition_element):
    try:
        _ = float(definition_element)
        return None
    except ValueError:
        if definition_element[0] == '@' or len(definition_element) == 1:
            return None
        return definition_element