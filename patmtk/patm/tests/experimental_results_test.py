import os
import unittest
from random import randint
# import patm
from ..modeling.experimental_results import ExperimentalResults, TrackedKernel

# Random order for tests runs. (Original is: -1 if x<y, 0 if x==y, 1 if x>y).
unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)

# CONSTANTS
my_dir = os.path.dirname(os.path.realpath(__file__))
# test_file_path = os.path.join(my_dir, 'test-file')

_dir = 'dir'
label = 'gav'
topics = 5
doc_passes = 2
bg_t = ['t0', 't1']
dm_t = ['t2', 't3', 't4']
mods = {'dcn': 1, 'icn': 5}

# tr_t = TrackedTopTokens([1,2,3], {'t00': [4,5,6], 't01': [4,5,5]})
#
# assert tr_t.average_coherence.all == [1,2,3]
# assert tr_t.average_coherence.last == 3
# assert tr_t.t00.all == [4,5,6]
# assert tr_t.t01.last == 5

kernel_data = [[1,2], [3,4], [5,6], {'t01': {'coherence': [1,2,3],
                                                             'contrast': [6, 3],
                                                             'purity': [1, 8]},
                                                     't00': {'coherence': [10,2,3],
                                                             'contrast': [67, 36],
                                                             'purity': [12, 89]},
                                                     't02': {'coherence': [10,11],
                                                             'contrast': [656, 32],
                                                             'purity': [17, 856]}
                                                     }]

kernel_data1 = [[10,20], [30,40], [50,6], {'t01': {'coherence': [3, 9],
                                                             'contrast': [96, 3],
                                                             'purity': [1, 98]},
                                                     't00': {'coherence': [19,2,93],
                                                             'contrast': [7, 3],
                                                             'purity': [2, 89]},
                                                     't02': {'coherence': [0,11],
                                                             'contrast': [66, 32],
                                                             'purity': [17, 85]}
                                                     }]

top10_data = [[1, 2], {'t01': [12,22,3], 't00': [10,2,3], 't02': [10,11]}]
top100_data = [[10, 20], {'t01': [5,7,9], 't00': [12,32,3], 't02': [11,1]}]
tau_trajectories_data = {'phi': [1,2,3,4,5], 'theta': [5,6,7,8,9]}

tracked = {'perplexity': [1, 2, 3], 'sparsity_phi': [-2, -4, -6], 'sparsity_theta': [2, 4, 6],
           'topic-kernel-0.6': kernel_data, 'topic-kernel-0.8': kernel_data1, 'top-tokens-10': top10_data, 'top-tokens-100': top100_data, 'tau-trajectories': tau_trajectories_data}


def setUpModule():
    pass


def tearDownModule():
    pass


class TestExperimentalResults(unittest.TestCase):
    """Unittest."""

    maxDiff, __slots__ = None, ()

    def setUp(self):
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        self.tracked_kernel = TrackedKernel(*kernel_data)
        self.exp = ExperimentalResults(_dir, label, topics, doc_passes, bg_t, dm_t, mods, tracked)

    def tearDown(self):
        """Method to tear down the test fixture. Run AFTER the test methods."""
        pass

    def addCleanup(self, function, *args, **kwargs):
        """Function called AFTER tearDown() to clean resources used on test."""
        pass

    @classmethod
    def setUpClass(cls):
        """Class method called BEFORE tests in an individual class run. """
        pass  # Probably you may not use this one. See setUp().

    @classmethod
    def tearDownClass(cls):
        """Class method called AFTER tests in an individual class run. """
        pass  # Probably you may not use this one. See tearDown().

    def test_experimental_results(self):
        assert self.exp.scalars.nb_topics == 5
        assert self.exp.scalars.document_passes == 2
        assert self.exp.tracked.perplexity.last == 3
        assert self.exp.tracked.perplexity.all == [1, 2, 3]
        assert self.exp.tracked.sparsity_phi.all == [-2, -4, -6]
        assert self.exp.tracked.sparsity_theta.last == 6
        assert self.exp.tracked.kernel6.average.purity.all == [5, 6]
        assert self.exp.tracked.kernel6.t02.coherence.all == [10, 11]
        assert self.exp.tracked.kernel6.t02.purity.all == [17, 856]
        assert self.exp.tracked.kernel6.t02.contrast.last == 32
        assert self.exp.tracked.kernel6.average.coherence.all == [1, 2]

        assert self.exp.tracked.kernel8.average.purity.all == [50, 60]
        assert self.exp.tracked.kernel8.t02.coherence.last == 11
        assert self.exp.tracked.kernel8.t02.purity.all == [17, 85]
        assert self.exp.tracked.kernel8.t00.contrast.last == 3
        assert self.exp.tracked.kernel8.average.purity.last == 6

        assert self.exp.tracked.top10.t01.all == [12, 22, 3]
        assert self.exp.tracked.top10.t00.last == 3
        assert self.exp.tracked.top10.average_coherence.all == [1, 2]
        assert self.exp.tracked.tau_trajectories.phi.all == [1, 2, 3, 4, 5]
        assert self.exp.tracked.tau_trajectories.theta.last == 9

    def test_tracked_kernel(self):
        assert self.tracked_kernel.average.contrast.all == [3, 4]
        assert self.tracked_kernel.average.purity.last == 6
        assert self.tracked_kernel.average.coherence.all == [1, 2]
        assert self.tracked_kernel.t00.contrast.all == [67, 36]
        assert self.tracked_kernel.t02.purity.all == [17, 856]
        assert self.tracked_kernel.t01.coherence.last == 3

    # def test_json_creation(self):
    #


if __name__.__contains__("__main__"):
    # print(__doc__)
    unittest.main(warnings='ignore')
    # Run just 1 test.
    # unittest.main(defaultTest='TestWeedMaster.test_dataset_creation', warnings='ignore')
