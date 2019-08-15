import numpy as np
from functools import reduce
import attr


def _group_iterations(self):
    """
    This [0, 0, 0, 0, 1, 1, 0.8, 0.6, 0.4, 0.2, 0.2, 0.1] leads to this [4, 2, 1, 1, 1, 2, 1].\n
    :return: steady_chunks of fit calls that have to be made to satisfy the dynamic change of the parameter value
    :rtype: list
    """
    iters = []
    pv = self._values[0]
    to_put = 1
    for cv in self._values[1:]:
        if cv == pv:
            to_put += 1
        else:
            iters.append(to_put)
            to_put = 1
        pv = cv
    iters.append(to_put)
    return iters


def _steady_iteration_ranges(iterations_groups):
    """
    This [5, 2, 1, 1, 1, 1] leads to this [[1,5], [6,7]]\n
    This [5, 2, 1, 1, 4, 1] leads to this [[1,5], [6,7], [10,13]].\n
    :param list iterations_groups:
    :return:
    :rtype: list indeces start from 1 (not 0) !
    """
    res = []
    accumulated_iters = 1
    for iter_chunk in iterations_groups:
        left_iter_count = accumulated_iters
        if iter_chunk > 1:
            right_iter_count = left_iter_count + iter_chunk - 1
            res.append([left_iter_count, right_iter_count])
            accumulated_iters += iter_chunk
        else:
            accumulated_iters += 1
    return res


@attr.s
class ParameterTrajectory(object):
    """
    This class encapsulates a parameter's value trajectory. This is basically the value the parameter shall have in
    every consecutive train iteration (pass through the whole collection.\n
    For example we may want to deactivate a regularizer for the first 5 iterations and then activating it while downgrading
     its coefficient from 1 to 0.2 over another 4 iterations. Then we need the tau value to follow trajectory:\n
      [0, 0, 0, 0, 0, 1, 0.8, 0.6, 0.4, 0.2]. \nThis class encapsulates this bevaviour.
    """
    _param = attr.ib(init=True, converter=str, repr=True)
    _values = attr.ib(init=True, converter=list, repr=True)
    group_iterations = attr.ib(init=True, default=attr.Factory(lambda self: _group_iterations(self), takes_self=True), repr=True)
    steady_chunks = attr.ib(init=False, default=attr.Factory(lambda self: IterationChunks([IterDuo(x) for x in _steady_iteration_ranges(self.group_iterations)]), takes_self=True))

    def __getattr__(self, item):
        if item == self._param:
            return self._values
        elif item == 'last_' + self._param:
            return self._values[-1]

    def __len__(self):
        return len(self._values)

    def __str__(self):
        return str(self._values)

    def __getitem__(self, item):
        return self._values[item]

    def __iter__(self):
        for v in self._values:
            yield v


class IterationChunks(object):
    def __init__(self, chunks_ref_list):
        if chunks_ref_list and (type(chunks_ref_list[0]) == list or chunks_ref_list[0] == 1):
            self.chunks = [IterSingle() if x == 1 else IterDuo(x) for x in chunks_ref_list]
        else:
            self.chunks = chunks_ref_list
        self._res = []
        self._ref = []
        self._toput_left = 0
        self._cond = None
        self._done = False

    def to_training_chunks(self, collection_passes):
        if not self.chunks:
            return IterationChunks([IterSingle() for _ in range(collection_passes)])
        covered = 0
        res = []
        for ind, ch in enumerate(self.chunks):
            to_add = [IterSingle() for _ in range(ch.left-covered-1)]
            res.extend(to_add)
            res.append(ch)
            covered = ch.right
        to_add = [IterSingle() for _ in range(collection_passes-covered)]
        res.extend(to_add)
        covered += len(to_add)
        assert covered == collection_passes
        return IterationChunks(res)

    def __str__(self):
        return '[{}]'.format(', '.join((str(_) for _ in self.chunks)))
        # return '[{}]'.format(', '.join(map(lambda x: '[{},{}]'.format(x.left, x.right), self.chunks)))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, item):
        return self.chunks[item]

    def __iter__(self):
        for v in self.chunks:
            yield v

    def __eq__(self, other):
        if len(other) != len(self.chunks):
            return False
        for ext_el, self_el in zip(other, self.chunks):
            if ext_el != self_el:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _overlap(self, duo, duo_list):
        self._res = []
        self._ref = duo
        self._done = False
        ind = 0
        ll = [_f for _f in [x if x.right > self._ref.left else None for x in duo_list] if _f]
        while not self._done and ind < len(ll):
            cand = ll[ind]
            if self._cand_left_smaller_than_ref_left(cand) and self._cond(cand) or self._cand_left_equal_to_ref_left(cand):
                self._insert_iterations(cand)
            elif self._cand_left_bigger_than_ref_left(cand):
                if self._cond(cand):
                    self._insert_iterations(cand)
                else:
                    self._done = True
            else:
                raise NoneConditionSatisfiedException("Candidate/external tuple {} to consider against the 'ref' tuple {}, was not found to be neither small, nor equal, nor bigger than 'ref'".format(cand, self._ref))
            ind += 1
        return self._res

    def _check_end(self, cand):
        if cand.left == self._ref.right - 1:
            self._done = True

    def _cand_left_smaller_than_ref_left(self, cand):
        self._cond = lambda x: self._ref.left < x.right  # since cand.left < ref_left then if also ref.left < cand.right then there is overlap
        self._toput_left = self._ref.left
        return cand.left < self._ref.left

    def _cand_left_equal_to_ref_left(self, cand): # since ref.left = cand.left, there is overlap by default. no need for secondary condition
        self._toput_left = self._ref.left
        return cand.left == self._ref.left

    def _cand_left_bigger_than_ref_left(self, cand):
        self._cond = lambda x: x.left < self._ref.right  # since ref_left < cand.left then if also can.left < ref.right then there is overlap
        self._toput_left = cand.left
        return self._ref.left < cand.left

    def _insert_iterations(self, cand):
        if self._ref.right < cand.right:
            self._res.append([self._toput_left, self._ref.right])
        else:
            self._res.append([self._toput_left, cand.right])
        self._check_end(cand)

    def common_chunks(self, chunks):
        """
        :param IterationChunks chunks:
        :return:
        :rtype: IterationChunks
        """
        return IterationChunks([IterDuo(item) for sublist in map(lambda x: self._overlap(x, chunks), self.chunks) for item in sublist])

class NoneConditionSatisfiedException(Exception): pass


class IterChunk(object):
    def __init__(self, data):
        self._data = data

    @property
    def left(self):
        return self._data[0]
    @property
    def right(self):
        return self._data[1]
    @property
    def span(self):
        return self._data[1] - self._data[0] + 1

    def __str__(self):
        return str(self._data)

    def __eq__(self, other):
        if self._data == other:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class IterDuo(IterChunk):
    def __init__(self, iters_tuple):
        super(IterDuo, self).__init__(iters_tuple)
        assert len(self._data) == 2
        assert self._data[0] < self._data[1]
        assert type(self._data[0]) == type(self._data[1]) == int

    def __str__(self):
        return '[{}, {}]'.format(self._data[0], self._data[1])

class IterSingle(IterChunk):
    def __init__(self):
        super(IterSingle, self).__init__(1)
    @property
    def left(self):
        return self._data
    @property
    def right(self):
        return self._data
    @property
    def span(self):
        return self._data


class TrajectoryBuilder(object):
    _interpolation_kind2interpolant = {'linear': {'preprocess': lambda x: (x[2], x[0], x[1]),
                                                 'process': lambda x: list(np.interp(*x))},
                                      'quadratic': {'preprocess': lambda x: (np.polyfit(x[0], x[1], 2), x[2]),
                                                    'process': lambda x: np.polyval(*x)},
                                      'cubic': {'preprocess': lambda x: (np.polyfit(x[0], x[1], 3), x[2]),
                                                 'process': lambda x: np.polyval(*x)}}

    def __init__(self):
        self._values = []
        self._name = ''
        self._interpolant = None

    @property
    def name(self):
        return self._name

    def steady_prev(self, iters):
        """Use regularizer using the latest tau used and keeping it constant for 'iters' train cycles through the collection"""
        self._values.extend([self._values[-1]]*iters)
        return self

    def steady_new(self, iters, value):
        """Use regularizer with a constant tau for 'iters' train cycles through the collection"""
        self._values.extend([value]*iters)
        return self

    def deactivate(self, iters):
        """Keep regularizer inactive for 'iters' train cycles through the collection"""
        self._values.extend([0]*iters)
        return self

    def interpolate_to(self, iters, value, interpolation='linear', start=None):
        """Use regularizer with tau gradually increasing or descreasing up to a value. Interpolates from the latest value
        used, (or from 'start' value if specified) to the specified 'value' using 'iters' steps. Each step is a train
        cycle through the collection\n
        Supports linear interpolation"""
        assert iters > 1
        prev_iter = len(self._values)
        iter_inds = range(prev_iter, prev_iter + iters)
        if start is None:
            start = self._values[-1]
            iter_inds = range(prev_iter + 1, prev_iter + iters + 1)
        self._interpolate(prev_iter, iter_inds, start, value, interpolation=interpolation)
        return self

    def _interpolate(self, prev_iter, iter_inds, start_y, end_y, interpolation='linear'):
        xs = [prev_iter, iter_inds[-1]]
        ys = [start_y, end_y]
        prods = self._interpolation_kind2interpolant[interpolation]['preprocess']((xs, ys, iter_inds))
        vals = self._interpolation_kind2interpolant[interpolation]['process'](prods)
        self._values.extend(vals)

    def begin_trajectory(self, name):
        self._name = name
        self._values = []
        return self

    def create(self):
        return ParameterTrajectory(self._name, self._values)

    def create_tau_trajectory(self, values_list):
        """Typical factory method"""
        return ParameterTrajectory('tau', values_list)

def get_fit_iteration_chunks(parameter_trajectories):
    """
    Given a list of parameter trajectories along the iteration count "dimension", returns a list of iterations tuples/steady_chunks.
    This steady_chunks indicate slices of the train iterations that the parameters stay constant and therefore fit can be called
    "consecutively" for these steady_chunks. It creates an object that encapsulates these steady_chunks\n
    :param list parameter_trajectories: the list of ParameterTrajectory objects
    :return: the newly created object
    :rtype: IterationChunks
    """
    return reduce(lambda x, y: x.common_chunks(y), map(lambda x: x.steady_chunks, parameter_trajectories))



if __name__ == '__main__':
    tr_builder = TrajectoryBuilder()
    _test(tr_builder)
