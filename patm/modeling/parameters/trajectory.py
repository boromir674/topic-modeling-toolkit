import numpy as np


class ParameterTrajectory(object):
    """
    This class encapsulates a parameter's value trajectory. This is basically the value the parameter shall have in
    every consecutive train iteration (pass through the whole collection.\n
    For example we may want to deactivate a regularizer for the first 5 iterations and then activating it while downgrading
     its coefficient from 1 to 0.2 over another 4 iterations. Then we need the tau value to follow trajectory:\n
      [0, 0, 0, 0, 0, 1, 0.8, 0.6, 0.4, 0.2]. \nThis class encapsulates this bevaviour.
    """
    def __init__(self, param_name, values_list):
        self._param = param_name
        self._values = values_list
        self.steady_chunks = IterationChunks(map(lambda x: IterDuo(x), self._steady_iteration_ranges(self.group_iterations())))

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

    def group_iterations(self):
        """
        This [0, 0, 0, 0, 0, 1, 1, 0.8, 0.6, 0.4, 0.2] leads to this [5, 2, 1, 1, 1, 1].\n
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

    def _steady_iteration_ranges(self, iterations_groups):
        """
        This [5, 2, 1, 1, 1, 1] leads to this [[1,5], [6,7]]\n
        This [5, 2, 1, 1, 4, 1] leads to this [[1,5], [6,7], [10,13]].\n
        :param list iterations_groups:
        :return:
        :rtype: list
        """
        res = []
        accumulated_iters = 1
        for iter_chunk in iterations_groups:
            left_iter_count = accumulated_iters
            if iter_chunk > 1:
                try:
                    right_iter_count = left_iter_count + iter_chunk - 1
                except TypeError as e:
                    print 'ERROR'
                    print left_iter_count
                    print iter_chunk
                    import sys
                    sys.exit(1)
                res.append([left_iter_count, right_iter_count])
                accumulated_iters += iter_chunk
            else:
                accumulated_iters += 1
        return res


class IterationChunks(object):
    def __init__(self, chunks_ref_list):
        if chunks_ref_list and (type(chunks_ref_list[0]) == list or chunks_ref_list[0] == 1):
            self.chunks = map(lambda x: IterSingle() if x == 1 else IterDuo(x), chunks_ref_list)
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
        ll = filter(None, map(lambda x: x if x.right > self._ref.left else None, duo_list))
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

class NoneConditionSatisfiedException(Exception):
    def __init__(self, msg):
        super(NoneConditionSatisfiedException, self).__init__(msg)

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
    """
    This class acts as a ParameterTrajectory factory.
    """
    interpolation_kind2interpolant = {'linear': {'preprocess': lambda x: (x[2], x[0], x[1]),
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
        prods = self.interpolation_kind2interpolant[interpolation]['preprocess']((xs, ys, iter_inds))
        vals = self.interpolation_kind2interpolant[interpolation]['process'](prods)
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


trajectory_builder = TrajectoryBuilder()


def _test(builder):
    tr1 = builder.begin_trajectory('tau').deactivate(2).steady_prev(3).interpolate_to(3, 1.2).steady_prev(
        2).interpolate_to(2, 1).create()
    tr2 = builder.begin_trajectory('tau').deactivate(2).steady_new(3, 0.8).interpolate_to(4, 1.2).steady_prev(
        2).interpolate_to(2, 1).create()
    tr3 = builder.begin_trajectory('tau').steady_new(3, 4).interpolate_to(3, 0.5).steady_prev(2).create()

    tr4 = builder.begin_trajectory('tau').interpolate_to(4, 0.5, start=1).steady_new(2, 0.2).create()

    assert IterChunk([1, 2]) == IterChunk([1, 2])
    assert IterChunk([1, 2]) != IterChunk([1, 4])

    assert IterationChunks([[1, 5], [8, 10]]) == IterationChunks([[1, 5], [8, 10]])
    assert IterationChunks([[1, 5], [8, 10]]) != IterationChunks([[1, 5], [8, 11]])
    assert IterationChunks([[1, 5], [8, 10]]) != IterationChunks([[1, 5], [8, 10], [11, 14]])

    g = IterationChunks([[1, 5], [8, 10]])
    d1 = IterChunk([1, 2])
    d2 = IterChunk([1, 3])

    # print tr1
    # print tr1.steady_chunks
    # print [str(_) for _ in tr1.steady_chunks]
    # print tr1.steady_chunks, IterationChunks([[1, 5], [8, 10]])

    assert tr1.steady_chunks == g
    assert tr1.group_iterations() == [5, 1, 1, 3, 1, 1]
    assert tr2.group_iterations() == [2, 3, 1, 1, 1, 3, 1, 1]
    assert tr2.steady_chunks == IterationChunks([[1, 2], [3, 5], [9, 11]])
    assert tr3.group_iterations() == [3, 1, 1, 3]
    assert tr3.steady_chunks == IterationChunks([[1, 3], [6, 8]])
    assert tr4.group_iterations() == [1, 1, 1, 1, 2]
    assert tr4.steady_chunks == IterationChunks([[5, 6]])

    # a = tr1.steady_chunks.common_chunks(tr2.steady_chunks)
    # print a
    # print [[1, 2], [3, 5], [9, 10]]
    # assert a == [[1, 2], [3, 5], [9, 10]]
    # print list(tr2.steady_chunks.common_chunks(tr1.steady_chunks))
    e1 = tr2.steady_chunks.common_chunks(tr1.steady_chunks)
    assert tr2.steady_chunks.common_chunks(tr1.steady_chunks) == IterationChunks([[1, 2], [3, 5], [9, 10]])
    assert tr1.steady_chunks.common_chunks(tr3.steady_chunks) == IterationChunks([[1, 3]])
    assert tr3.steady_chunks.common_chunks(tr1.steady_chunks) == IterationChunks([[1, 3]])
    assert tr1.steady_chunks.common_chunks(tr4.steady_chunks) == IterationChunks([])
    assert tr4.steady_chunks.common_chunks(tr1.steady_chunks) == IterationChunks([])

    # print '\n', tr1, '\n', tr1.steady_chunks
    # print tr2, '\n', tr2.steady_chunks
    res = get_fit_iteration_chunks([tr1, tr2])
    assert res == IterationChunks([[1, 2], [3, 5], [9, 10]])
    assert res.to_training_chunks(14) == IterationChunks([[1, 2], [3, 5], 1, 1, 1, [9, 10], 1, 1, 1, 1])
    # print [str(_) for _ in res]
    # print 'G', [_.span for _ in res.to_training_chunks(14)]
    res = get_fit_iteration_chunks([tr1, tr2, tr3])
    assert res == IterationChunks([[1, 2]])
    assert res.to_training_chunks(6) == IterationChunks([[1,2], 1, 1, 1, 1])
    print "All tests passed"

if __name__ == '__main__':

    _test(trajectory_builder)
