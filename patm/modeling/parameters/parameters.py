import abc

class Parameter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def value(self):
        raise NotImplemented

class AbstractImmutableParameter(Parameter):
    __slots__ = ['value']

    def value(self):
        raise NotImplementedError


class ImmutableParameter(object):
    __slots__ = ['value']

    def __init__(self, value):
        super(ImmutableParameter, self).__setattr__('value', value)

    def __setattr__(self, name, value):
        """"""
        msg = '{} is immutable'.format(self.__class__)
        raise AttributeError(msg)


class ParameterSpan(object):
    def __init__(self, list_span):
        self.span = list_span
        self._index = -1
    def __iter__(self):
        for _ in self.span:
            yield _
    def __getitem__(self, item):
        return self.span[item]
    def __len__(self):
        return len(self.span)
    def next(self):
        self._index = (self._index + 1) % len(self.span)
        return self.span[self._index]
    def __str__(self):
        return str(self.span)

class ParameterGrid(object):
    def __init__(self, parameter_spans):
        self._spans = []
        for sp in parameter_spans:
            if type(sp) == list:
                self._spans.append(ParameterSpan(sp))
            elif type(sp) == ParameterSpan:
                self._spans.append(sp)
            else:
                print sp
                raise InvalidSpanException("Invalid span object given of type " + type(sp).__name__)

    def generate_with_filter(self, inds):
        return (v for v, i in enumerate(self) if i not in inds)

    def __len__(self):
        if len(self._spans) == 1:
            return len(self._spans[0])
        return reduce(lambda x,y: x*y, map(len, self._spans))

    def _init(self):
        self._span_lens = [len(_) for _ in self._spans]
        self._inds = [0] * len(self._spans)
        self._to_increment = len(self._spans) - 1
        self._parameter_vector = map(lambda x: x.next(), self._spans)

    def _increment(self):
        """Should change the self._parameter_vector to the next state to yield in self.__iter__"""
        cur_value = self._spans[self._to_increment].next()
        self._parameter_vector[self._to_increment] = cur_value
        while cur_value == self._spans[self._to_increment][0] and self._to_increment > 0:
            self._to_increment -= 1
            cur_value = self._spans[self._to_increment].next()
            self._parameter_vector[self._to_increment] = cur_value
        self._to_increment = len(self._spans) - 1

    def _spit(self):
        return [_ for _ in self._parameter_vector]

    def __iter__(self):
        self._init()
        yield self._spit()
        for i in range(reduce(lambda x, y: x*y, self._span_lens) - 1):
            self._increment()
            yield self._spit()


def test():
    sp1 = ParameterSpan(sparse_tau_start[:3])
    sp2 = ParameterSpan(deactivation_period)
    sp3 = ParameterSpan(doc_iterations[:2])
    pg = ParameterGrid([sp1, sp2, sp3])
    res = []
    for prs in pg:
        res.append(prs)
    assert res == [[-0.1, 10, 1],
                   [-0.1, 10, 5],
                   [-0.1, 20, 1],
                   [-0.1, 20, 5],
                   [-0.1, 30, 1],
                   [-0.1, 30, 5],
                   [-0.2, 10, 1],
                   [-0.2, 10, 5],
                   [-0.2, 20, 1],
                   [-0.2, 20, 5],
                   [-0.2, 30, 1],
                   [-0.2, 30, 5],
                   [-0.3, 10, 1],
                   [-0.3, 10, 5],
                   [-0.3, 20, 1],
                   [-0.3, 20, 5],
                   [-0.3, 30, 1],
                   [-0.3, 30, 5],
                   ]

class InvalidSpanException(Exception):
    def __init__(self, msg):
        super(InvalidSpanException, self).__init__(msg)



# class ParameterExplorer(object):
#     def get_tau_start_span(self):
#         return sparse_tau_start
#     def get_tau_deactivation_period_span(self):
#         return deactivation_period
#     def get_tau_deactivation_period_pct_span(self):
#         return deactivation_period_pct
#     def get_spline_types(self):
#         return splines
#
# parameter_explorer = ParameterExplorer()

if __name__ == '__main__':
    test()