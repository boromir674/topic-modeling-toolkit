import pytest
from patm.discreetization import PoliticalSpectrumManager, SCALE_PLACEMENT, DISCRETIZATION, BinDesign, Bins

@pytest.fixture(scope='module')
def valid_design():
    return BinDesign([8, 16])
@pytest.fixture(scope='module')
def in_valid_design():
    return [8, 16, 10]


class TestDscreetization(object):

    def test_binning(self):
        pm = PoliticalSpectrumManager(SCALE_PLACEMENT, DISCRETIZATION)
        assert pm.class_names == ['extreme_left', 'left', 'left_of_middle', 'right_of_middle', 'right', 'extreme_right']
        assert pm.poster_id2ideology_label['10513336322'] == 'extreme_left'
        assert pm.poster_id2ideology_label['19013582168'] == 'left'
        assert pm.poster_id2ideology_label['131459315949'] == 'left_of_middle'
        assert pm.poster_id2ideology_label['8304333127'] == 'right_of_middle'
        assert pm.poster_id2ideology_label['15704546335'] == 'right'
        assert pm.poster_id2ideology_label['140738092630206'] == 'extreme_right'

    def test_design(self, valid_design):
         assert [_ for _ in valid_design.ranges(20)] == [range(8), range(8, 16), range(16, 20)]

         sc1 = [
             ('extreme left', ['The New Yorker',
                               'Slate',
                               'The Guardian',
                               'Al Jazeera',
                               'NPR',
                               'New York Times',
                               ]),
             ('left', ['PBS',
                       'BBC News',
                       'HuffPost',
                       'Washington Post',
                       'The Economist',
                       'Politico']),
             ('left of middle', ['MSNBC',
                                 'CNN',
                                 'NBC News',
                                 'CBS News',
                                 'ABC News',
                                 'USA Today',
                                 ]),
             ('right of middle', ['Yahoo News',
                                  'Wall Street Journal',
                                  ]),
             ('right', ['Fox News']),
             ('extreme right', ['Breitbart',
                                'The Blaze'])
         ]
         assert [x[1] for x in sc1] == list(Bins.from_design([6, 12, 18, 20, 21], SCALE_PLACEMENT))
