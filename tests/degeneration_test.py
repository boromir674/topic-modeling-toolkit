import unittest
from random import randint
from topic_modeling_toolkit.patm.modeling.experiment import DegenerationChecker


# Random order for tests runs. (Original is: -1 if x<y, 0 if x==y, 1 if x>y).
unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)

# unittest.TestLoader.sortTestMethodsUsing = None


def setUpModule():
    pass


def tearDownModule():
    pass


class TestDegenerationCheck(unittest.TestCase):
    """Unittest."""

    maxDiff, __slots__ = None, ()

    def setUp(self):
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        self.all_info = [{'t00': ['a', 'b', 'c'],
                         't01': ['d', 'e', 'f'],
                         't02': ['g', 'h', 'i']}]
        self.t00_missing = [{'t00': [],
                           't01': ['d', 'e', 'f'],
                           't02': ['g', 'h', 'i']}]
        self.t00_n_t02_missing = [{'t00': [],
                                  't01': ['d', 'e', 'f']}]
        self.all_missing = [{}]
        self.degen_checker = DegenerationChecker(['t00', 't01', 't02'])
        self.gdi = self.degen_checker.get_degeneration_info

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

    def test_degenerated_tokens(self):
        assert {'t00': [(2, 6)], 't01': [], 't02': [(4, 6)]} == self.gdi(self.all_info * 2 + self.t00_missing * 2 + self.t00_n_t02_missing * 2)
        assert {'t00': [(2, 4), (5, 7)], 't01': [], 't02': [(5, 7)]} == self.gdi(self.all_info * 2 + self.t00_missing * 2 + self.all_info + self.t00_n_t02_missing * 2)
        assert {'t00': [(0, 2), (4, 10)], 't01': [], 't02': [(4, 10)]} == self.gdi(self.t00_missing * 2 + self.all_info * 2 + self.t00_n_t02_missing * 6)
        assert {'t00': [(2, 6)], 't01': [(2, 4)], 't02': [(2, 6)]} == self.gdi(self.all_info * 2 + self.all_missing * 2 + self.t00_n_t02_missing * 2)
        assert {'t00': [(0, 5)], 't01': [(0, 1)], 't02': [(0, 1), (3, 5)]} == self.gdi(self.all_missing + self.t00_missing * 2 + self.t00_n_t02_missing * 2)
        assert {'t00': [(0, 2), (4, 6)], 't01': [(4, 6)], 't02': [(0, 2), (4, 6)]} == self.gdi(self.t00_n_t02_missing * 2 + self.all_info * 2 + self.all_missing * 2)
        assert {'t00': [(2, 7)], 't01': [(6, 7)], 't02': [(4, 7)]} == self.gdi(self.all_info * 2 + self.t00_missing * 2 + self.t00_n_t02_missing * 2 + self.all_missing)

        # assert {'t00': [(4, 5)], 't01': [], 't02': []} == self.gdi(self.all_info * 2 + self.t00_missing * 2 + self.t00_n_t02_missing * 2)
        # assert {'t00': [(4, 5)], 't01': [], 't02': []} == self.gdi(self.all_info * 2 + self.t00_missing * 2 + self.t00_n_t02_missing * 2)
        # assert {'t00': [(4, 5)], 't01': [], 't02': []} == self.gdi(self.all_info * 2 + self.t00_missing * 2 + self.t00_n_t02_missing * 2)


if __name__.__contains__("__main__"):

    unittest.main(warnings='ignore')
    # Run just 1 test.
    # unittest.main(defaultTest='TestExperimentalResults.test_tracked_kernel', warnings='ignore')
