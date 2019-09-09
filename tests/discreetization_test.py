import pytest
from topic_modeling_toolkit.patm.discreetization import PoliticalSpectrum, BinDesign, Bins, Population, DiscreetizationScheme
from topic_modeling_toolkit.patm.definitions import SCALE_PLACEMENT, DISCREETIZATION


@pytest.fixture(scope='module')
def valid_design():
    return BinDesign([8, 16])


@pytest.fixture(scope='module')
def population(political_spectrum):
    return Population(political_spectrum)


class TestDscreetization(object):

    def test_binning(self):
        pm = PoliticalSpectrum(SCALE_PLACEMENT, DISCREETIZATION)
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

    def test_population(self, preprocess_phase, population):
        """Can considere to skip  this test as its success relies on heuristic optimization"""
        class_names = ['{}_Class'.format(x) for x in ['liberal', 'centre_liberal', 'centre_conservative', 'conservative']]
        pool_size = 50
        population.init_random(preprocess_phase.outlet_ids, pool_size, len(class_names)-1)
        population.evolve(nb_generations=50, prob=0.2)
        assert population.pool[0].fitness < 50
        y = DiscreetizationScheme.from_design(list(population.pool[0]), SCALE_PLACEMENT, class_names=class_names)

        # assert '' == str(y)
