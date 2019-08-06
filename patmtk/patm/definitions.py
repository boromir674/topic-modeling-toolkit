import os
from collections import OrderedDict


my_dir = os.path.dirname(os.path.realpath(__file__))

RESULTS_DIR_NAME = 'results'
MODELS_DIR_NAME = 'models'
GRAPHS_DIR_NAME = 'graphs'

REGULARIZERS_CFG = os.path.join(my_dir, '../regularizers.cfg')
TRAIN_CFG = os.path.join(my_dir, '../test-train.cfg')

patm_root_dir = '/data/thesis'
data_root_dir = '/data/thesis/data'
bows_dir = '/data/thesis/data/collections'
words_dir = '/data/thesis/data/collections'

COLLECTIONS_DIR_PATH = '/data/thesis/data/collections'

pre = 'pickles'

categories = ('posts', 'comments', 'reactions', 'users')

cat2dir = dict(zip(categories, map(lambda x: os.path.join(data_root_dir, x), categories)))

cat2file_ids = dict(zip(categories, ((2, 5, 38),
                                     (9, 16, 28),
                                     (1, 12, 19, 43),
                                     (24, 31, 36)
                                     )))

CATEGORY_2_FILES_HASH = {cat: [os.path.join(cat2dir[cat], '{}_{}_{}.pkl'.format(pre, iid, cat)) for iid in cat2file_ids[cat]] for cat in categories}


###### TOKEN COOCURENCE INFORMATION #####
COOCURENCE_DICT_FILE_NAMES = ['cooc_tf_', 'cooc_df_', 'ppmi_tf_', 'ppmi_df_']

###### CONSTANTS #####
BINARY_DICTIONARY_NAME = 'mydic.dict'


############## IDEOLOGY INFORMATION ##############

labels = ('extreme left', 'left', 'left of middle', 'right of middle', 'right', 'extreme right')

outlets_by_ideology = {
    'extreme left': {
        'The Guardian': '10513336322',
        'Al Jazeera': '7382473689',
        'NPR': '10643211755',
        'New York Times': '5281959998',
        'The New Yorker': '9258148868',
        'Slate': '21516776437'},
    'left': {
        'PBS': '19013582168',
        'BBC News': '228735667216',
        'HuffPost': '18468761129',
        'Washington Post': '6250307292',
        'The Economist': '6013004059',
        'Politico': '62317591679'},
    'left of middle': {
        'CBS News': '131459315949',
        'ABC News': '86680728811',
        'USA Today': '13652355666',
        'NBC News': '155869377766434',
        'CNN': '5550296508',
        'MSNBC': '273864989376427'},
    'right of middle': {
        'Wall Street Journal': '8304333127',
        'Yahoo News': '338028696036'},
    'right': {
        'Fox News': '15704546335'},
    'extreme right': {
        'Breitbart': '95475020353',
        'The Blaze': '140738092630206'}}


def id2ideology(_id):
    for ideology_label, innerdict in outlets_by_ideology.items():
        for outlet_name, outlet_id in innerdict.items():
            if _id == outlet_id:
                return ideology_label
    return None

def label2alphanumeric(label):
    return '_'.join(label.lower().split(' '))

poster_id2outlet_label = {poster_id: poster_name for ide_bin_dict in outlets_by_ideology.values() for poster_name, poster_id in ide_bin_dict.items()}

# should be exposed
poster_id2ideology_label = {poster_id: label2alphanumeric(id2ideology(poster_id)) for poster_id in poster_id2outlet_label}

# the outlets_by_ideology dict only sorted as in labels from extreme left to extreme right
id_label2outlet_dict = OrderedDict([('_'.join(ide.split(' ')), OrderedDict(sorted(map(lambda x: (x[0], int(x[1])), outlets_by_ideology[ide].items()), key=lambda x: x[0]))) for ide in labels])

CLASS_LABELS = list(id_label2outlet_dict.keys())

# id_label2outlet_dict['extreme_left']['The Guardian'] == 10513336322
IDEOLOGY_CLASS_NAME = '@labels_class'
DEFAULT_CLASS_NAME = '@default_class'


######################################

SCALE_PLACEMENT = [
    ('The New Yorker', '9258148868'),
    ('Slate', '21516776437'),
    ('The Guardian', '10513336322'),
    ('Al Jazeera', '7382473689'),
    ('NPR', '10643211755'),
    ('New York Times', '5281959998'),
    ('PBS', '19013582168'),
    ('BBC News', '228735667216'),
    ('HuffPost', '18468761129'),
    ('Washington Post', '6250307292'),
    ('The Economist', '6013004059'),
    ('Politico', '62317591679'),
    ('MSNBC', '273864989376427'),
    ('CNN', '5550296508'),
    ('NBC News', '155869377766434'),
    ('CBS News', '131459315949'),
    ('ABC News', '86680728811'),
    ('USA Today', '13652355666'),
    ('Yahoo News', '338028696036'),
    ('Wall Street Journal', '8304333127'),
    ('Fox News', '15704546335'),
    ('Breitbart', '95475020353'),
    ('The Blaze', '140738092630206')
]

DISCRETIZATION = {

    'legacy-scheme': [
        ('extreme left', ['The Guardian',
                          'Al Jazeera',
                          'NPR',
                          'New York Times',
                          'The New Yorker',
                          'Slate']),
        ('left', ['PBS',
                  'BBC News',
                  'HuffPost',
                  'Washington Post',
                  'The Economist',
                  'Politico']),
        ('left of middle', ['CBS News',
                            'ABC News',
                            'USA Today',
                            'NBC News',
                            'CNN',
                            'MSNBC']),
        ('right of middle', ['Wall Street Journal',
                             'Yahoo News']),
        ('right', ['Fox News']),
        ('extreme right', ['Breitbart',
                           'The Blaze'])
    ],

    '3-poles-scheme': [
        ('liberal', []),
        ('centre', []),
        ('conservative', [])
    ]
}
#
# AUDIENCE_IDEOLOGICAL_PLACEMENT_DISCRETIZATION = {
#
#     'scheme_1': [
#         ('extreme left', {'The Guardian': '10513336322',
#                           'Al Jazeera': '7382473689',
#                           'NPR': '10643211755',
#                           'New York Times': '5281959998',
#                           'The New Yorker': '9258148868',
#                           'Slate': '21516776437'}),
#         ('left', {'PBS': '19013582168',
#                   'BBC News': '228735667216',
#                   'HuffPost': '18468761129',
#                   'Washington Post': '6250307292',
#                   'The Economist': '6013004059',
#                   'Politico': '62317591679'}),
#         ('left of middle', {'CBS News': '131459315949',
#                             'ABC News': '86680728811',
#                             'USA Today': '13652355666',
#                             'NBC News': '155869377766434',
#                             'CNN': '5550296508',
#                             'MSNBC': '273864989376427'}),
#         ('right of middle', {'Wall Street Journal': '8304333127',
#                              'Yahoo News': '338028696036'}),
#         ('right', {'Fox News': '15704546335'}),
#         ('extreme right', {'Breitbart': '95475020353',
#                            'The Blaze': '140738092630206'})
#     ],


class PoliticalSpectrumManager(object):

    def __init__(self, discretization_specs):
        self._schemes = {k: OrderedDict(discretization) for k, discretization in discretization_specs}
        self.discretization_scheme = 'legacy-scheme'

    @property
    def discretization_scheme(self):
        return self._schemes[self._cur]

    @property
    def poster_id2ideology_label(self):
        return {poster_id: self.label2alphanumeric(self._id2ideology(poster_id)) for poster_id in self._poster_id2outlet_label}

    @discretization_scheme.setter
    def discretization_scheme(self, scheme):
        if scheme not in self._schemes:
            raise KeyError("The schemes impemented are [{}]. Requested '{}' instead.".format(' ,'.join(self._schemes.keys()), scheme))
        self._cur = scheme
        self._poster_id2outlet_label = {poster_id: poster_name for ide_bin_dict in self._schemes[self._cur].values() for poster_name, poster_id in ide_bin_dict.items()}
        self._id_label2outlet_dict = OrderedDict([('_'.join(ide.split(' ')),
                                                   OrderedDict(sorted([(outlet, outlet_id) for outlet, outlet_id in self._schemes[self._cur][ide].items()], key=lambda x: x[0]))) for ide in self._schemes[self._cur].keys()])

    @property
    def class_names(self):
        """The names of the discrete bins applied on the 10-point scale of ideological consistency"""
        return map(self.label2alphanumeric, self._schemes[self._cur].keys())

    @property
    def labels(self):
        return list(self._bins.keys())

    def _id2ideology(self, _id):
        for ideology_label, innerdict in self._schemes[self._cur].items():
            for outlet_name, outlet_id in innerdict.items():
                if _id == outlet_id:
                    return ideology_label
        return None

    @staticmethod
    def label2alphanumeric(label):
        return label.lower().replace(' ', '_')
