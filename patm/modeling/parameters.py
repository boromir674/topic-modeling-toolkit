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


if __name__ == '__main__':
    builder = TrajectoryBuilder()
    tr1 = builder.begin_trajectory('tau').deactivate(3).steady_new(3, 0.8).interpolate_to(3, 0.2).create()
    print tr1, type(tr1)
    print tr1.tau, type(tr1.tau)
    print tr1.last_tau, type(tr1.last_tau)
    print len(tr1)
    print tr1[3], tr1[6]
    print [_ for _ in tr1]
    print [_ for _ in tr1]