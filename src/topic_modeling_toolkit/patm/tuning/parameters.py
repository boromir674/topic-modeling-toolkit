import abc
from functools import reduce
from builtins import object  # python 2-3 cross support


class Parameter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def value(self):
        raise NotImplemented

class AbstractImmutableParameter(Parameter):
    # __slots__ = ['value']

    def value(self):
        raise NotImplementedError


class ImmutableParameter(object):
    __slots__ = ['value']

    def __init__(self, value):
        super(ImmutableParameter, self).__setattr__('value', value)

    def __setattr__(self, name, value):
        raise AttributeError('{} is immutable'.format(self.__class__))


# Python 2 'def __iter__ ' returning self, 'def next', client as obj.next()
# Python 3  Define an __iter__() method which returns an object with a __next__() method. If the class defines __next__(), then __iter__() can just return self:

class ParameterSpan(object):

    def __init__(self, list_span):
        self.span = list_span
        self._index = -1

    def __iter__(self):
        return self

    def __getitem__(self, item):
        return self.span[item]

    def __len__(self):
        return len(self.span)

    def __next__(self):  # this works on python 3 because of futures. On python 3 it is the correct syntax already
        self._index = (self._index + 1) % len(self.span)
        return self.span[self._index]

    def __str__(self):
        return str(self.span)

class ParameterGrid(object):
    def __init__(self, parameter_spans):
        self._spans = []
        self._filtered_out_indices = []
        for sp in parameter_spans:
            if type(sp) == list:
                self._spans.append(ParameterSpan(sp))
            elif type(sp) == ParameterSpan:
                self._spans.append(sp)
            else:
                raise InvalidSpanException("Object {} of type {} is not a valid list-like parameter span.".format(sp, type(sp).__name__))

    def remove_filter(self):
        self._filtered_out_indices = []

    @property
    def ommited_indices(self):
        return self._filtered_out_indices

    @ommited_indices.setter
    def ommited_indices(self, indices_to_ommit):
        self._filtered_out_indices = indices_to_ommit

    def __len__(self):
        return len([_ for _ in self._get_filtered_generator()])

    def _generate_spans(self):
        for i, sp in enumerate(self._spans):
            if i not in self._filtered_out_indices:
                yield sp

    def _init(self):
        self._span_lens = [len(_) for _ in self._spans]
        self._inds = [0] * len(self._spans)
        self._to_increment = len(self._spans) - 1
        self._parameter_vector = [next(x) for x in self._spans]

    def _increment(self):
        """Should change the self._parameter_vector to the next state to yield in self.__iter__"""
        cur_value = next(self._spans[self._to_increment])
        self._parameter_vector[self._to_increment] = cur_value
        while cur_value == self._spans[self._to_increment][0] and self._to_increment > 0:
            self._to_increment -= 1
            cur_value = next(self._spans[self._to_increment])
            self._parameter_vector[self._to_increment] = cur_value
        self._to_increment = len(self._spans) - 1

    def _spit(self):
        return [_ for _ in self._parameter_vector]

    def __iter__(self):
        for _ in self._get_filtered_generator():
            yield _

    def _get_filtered_generator(self):
        return (vector for i, vector in enumerate(self._raw_generate()) if i not in self._filtered_out_indices)

    def _raw_generate(self):
        # TODO include logic here to ommit filtered out indices
        self._init()
        yield self._spit()
        for i in range(reduce(lambda x, y: x*y, self._span_lens) - 1):
            self._increment()
            yield self._spit()


class InvalidSpanException(Exception): pass
