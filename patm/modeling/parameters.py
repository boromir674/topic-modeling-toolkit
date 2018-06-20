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
        self.chunks = IterationChunks(self._steady_iteration_ranges(self.group_iterations()))

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
        :return: chunks of fit calls that have to be made to satisfy the dynamic change of the parameter value
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
        This [5, 2, 1, 1, 1, 1] leads to this [[1,5], [6,7]]
        This [5, 2, 1, 1, 4, 1] leads to this [[1,5], [6,7], [10,13]]
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
    def __init__(self, iteration_chunks):
        self.chunks = iteration_chunks
        # if not all(map(lambda x: type(x) != list, self.chunks)):
        #     print self.chunks
        #     import sys
        #     sys,exit(1)

    def __len__(self):
        return len(self.chunks)

    def __str__(self):
        return str(self.chunks)

    def __getitem__(self, item):
        return self.chunks[item]

    def __iter__(self):
        for v in self.chunks:
            yield v

    def _compare(self, iter_tuple):
        
        l = max(iter_tuple[0], self._left)
        r = min(iter_tuple[1], self._right)
        if r - l >= 1:
            return [l, r]
        return None

    def common_chunks(self, iteration_chunks_obj):
        common_chunks = []
        ext_tuples = list(iteration_chunks_obj)
        iter_accum = 0
        ind = 0
        for self._left, self._right in ext_tuples:
            while iter_accum < self._right - 1 and ext_tuples[ind][0] < self._right and ext_tuples[ind][1] > self._left:
                res = self._compare(ext_tuples[ind])
                if res:
                    common_chunks.append(res)
                    iter_accum = res[1]
                    if ext_tuples[ind][1] - res[1] <= 1:
                        ind += 1
                elif ind >= len(ext_tuples) - 1:
                    break
        return IterationChunks(common_chunks)


def get_fit_iteration_chunks(param_traj_list):
    """

    :param param_traj_list:
    :return:
    """
    # TODO DOCUMENT!!!!
    # res = None
    # pre = param_traj_list[0]
    # for ci in param_traj_list[1:]:
    #     res = pre.chunks.common_chunks(ci.chunks)
    #     pre = ci
    # return res
    return reduce(lambda x, y: x.common_chunks(y), map(lambda x: x.chunks, param_traj_list))
    # return param_traj_list[0].chunks.common_chunks(param_traj_list[1].chunks)

class TrajectoryBuilder(object):
    """
    This class acts as a ParameterTrajectory factory.
    """
    def __init__(self):
        self._values = []
        self._name = ''

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
        if start is None:
            start = self._values[-1]
        prev_iter = len(self._values)
        iter_inds = range(prev_iter+1, prev_iter+1+iters)
        xs = [prev_iter, iter_inds[-1]]
        ys = [start, value]
        self._values.extend(list(np.interp(iter_inds, xs, ys)))
        return self

    def begin_trajectory(self, name):
        self._name = name
        self._values = []
        return self

    def create(self):
        return ParameterTrajectory(self._name, self._values)


def _test():
    builder = TrajectoryBuilder()
    tr1 = builder.begin_trajectory('tau').deactivate(2).steady_prev(3).interpolate_to(3, 1.2).steady_prev(
        2).interpolate_to(2, 1).create()
    tr2 = builder.begin_trajectory('tau').deactivate(2).steady_new(3, 0.8).interpolate_to(4, 1.2).steady_prev(
        2).interpolate_to(2, 1).create()
    tr3 = builder.begin_trajectory('tau').steady_new(3, 4).interpolate_to(3, 0.5).steady_prev(2).create()
    tr4 = builder.begin_trajectory('tau').interpolate_to(4, 0.5, start=1).steady_new(2, 0.2).create()
    assert tr1.group_iterations() == [5, 1, 1, 3, 1, 1]
    assert list(tr1.chunks) == [[1, 5], [8, 10]]
    assert tr2.group_iterations() == [2, 3, 1, 1, 1, 3, 1, 1]
    assert list(tr2.chunks) == [[1, 2], [3, 5], [9, 11]]
    assert tr3.group_iterations() == [3, 1, 1, 3]
    assert list(tr3.chunks) == [[1, 3], [6, 8]]
    assert tr4.group_iterations() == [1, 1, 1, 1, 2]
    assert list(tr4.chunks) == [[5, 6]]

    a = tr1.chunks.common_chunks(tr2.chunks)
    # print a
    # print [[1, 2], [3, 5], [9, 10]]
    # assert a == [[1, 2], [3, 5], [9, 10]]
    # print list(tr2.chunks.common_chunks(tr1.chunks))
    assert list(tr2.chunks.common_chunks(tr1.chunks)) == [[1, 2], [3, 5], [9, 10]]
    assert list(tr1.chunks.common_chunks(tr3.chunks)) == [[1, 3]]
    assert list(tr3.chunks.common_chunks(tr1.chunks)) == [[1, 3]]
    assert list(tr1.chunks.common_chunks(tr4.chunks)) == []
    assert list(tr4.chunks.common_chunks(tr1.chunks)) == []

    res = get_fit_iteration_chunks([tr1, tr2])
    assert list(res) == [[1, 2], [3, 5], [9, 10]]

    res = get_fit_iteration_chunks([tr1, tr2, tr3])
    assert list(res) == [[1, 2]]



if __name__ == '__main__':

    _test()

    # builder = TrajectoryBuilder()
    # tr1 = builder.begin_trajectory('tau').deactivate(2).steady_new(3, 0.8).interpolate_to(10, 1.2).steady_prev(2).interpolate_to(2, 1).create()
    # tr2 = builder.begin_trajectory('tau').interpolate_to(2, 0.5, start=1).deactivate(3).steady_new(3, 0.8).create()
    # # print tr1, type(tr1)
    # # print tr1.tau, type(tr1.tau)
    # # print tr1.last_tau, type(tr1.last_tau)
    # print tr2
    #
    # print tr1, len(tr1)
    # print tr1.group_iterations()
    # print tr1.chunks._steady_iteration_ranges()
    # print tr2, len(tr2)
    # print tr2.group_iterations()
    # print tr2.chunks._steady_iteration_ranges()