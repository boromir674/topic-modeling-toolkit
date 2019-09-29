
import pytest
from topic_modeling_toolkit.patm.tuning.parameters import ParameterSpan, ParameterGrid
from topic_modeling_toolkit.patm.modeling.regularization.trajectory import TrajectoryBuilder, IterationChunks, IterChunk, get_fit_iteration_chunks


@pytest.fixture(scope='module')
def parameter_grid():
    return ParameterGrid([
        ParameterSpan([1, 2, 3]),
        ParameterSpan(['a', 'b']),
        ParameterSpan([10, 0])])


def test_parameter_grid(parameter_grid):
    assert [_ for _ in parameter_grid] == [[1, 'a', 10],
                                           [1, 'a', 0],
                                           [1, 'b', 10],
                                           [1, 'b', 0],
                                           [2, 'a', 10],
                                           [2, 'a', 0],
                                           [2, 'b', 10],
                                           [2, 'b', 0],
                                           [3, 'a', 10],
                                           [3, 'a', 0],
                                           [3, 'b', 10],
                                           [3, 'b', 0]]


def test_trajectory_building():
    builder = TrajectoryBuilder()
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

    assert tr1.steady_chunks == g
    assert tr1.group_iterations == [5, 1, 1, 3, 1, 1]
    assert tr2.group_iterations == [2, 3, 1, 1, 1, 3, 1, 1]
    assert tr2.steady_chunks == IterationChunks([[1, 2], [3, 5], [9, 11]])
    assert tr3.group_iterations == [3, 1, 1, 3]
    assert tr3.steady_chunks == IterationChunks([[1, 3], [6, 8]])
    assert tr4.group_iterations == [1, 1, 1, 1, 2]
    assert tr4.steady_chunks == IterationChunks([[5, 6]])


    e1 = tr2.steady_chunks.common_chunks(tr1.steady_chunks)
    assert tr2.steady_chunks.common_chunks(tr1.steady_chunks) == IterationChunks([[1, 2], [3, 5], [9, 10]])
    assert tr1.steady_chunks.common_chunks(tr3.steady_chunks) == IterationChunks([[1, 3]])
    assert tr3.steady_chunks.common_chunks(tr1.steady_chunks) == IterationChunks([[1, 3]])
    assert tr1.steady_chunks.common_chunks(tr4.steady_chunks) == IterationChunks([])
    assert tr4.steady_chunks.common_chunks(tr1.steady_chunks) == IterationChunks([])

    res = get_fit_iteration_chunks([tr1, tr2])
    assert res == IterationChunks([[1, 2], [3, 5], [9, 10]])
    assert res.to_training_chunks(14) == IterationChunks([[1, 2], [3, 5], 1, 1, 1, [9, 10], 1, 1, 1, 1])

    res = get_fit_iteration_chunks([tr1, tr2, tr3])
    assert res == IterationChunks([[1, 2]])
    assert res.to_training_chunks(6) == IterationChunks([[1, 2], 1, 1, 1, 1])
