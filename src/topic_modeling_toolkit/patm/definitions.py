import os
from collections import OrderedDict
import attr
from functools import reduce


# COLLECTIONS_DIR_PATH = os.get'/data/thesis/data/collections'


my_dir = os.path.dirname(os.path.realpath(__file__))

RESULTS_DIR_NAME = 'results'
MODELS_DIR_NAME = 'models'
GRAPHS_DIR_NAME = 'graphs'
DATA_DIR = os.path.join(my_dir, '../../../data/')

pre = 'pickles'

cat2file_ids = dict(zip(('posts', 'comments', 'reactions', 'users'), ((2, 5, 38),
                                     (9, 16, 28),
                                     (1, 12, 19, 43),
                                     (24, 31, 36)
                                     )))
CATEGORY_2_FILES_HASH = {k: [os.path.join(DATA_DIR, '{}_{}_{}.pkl'.format(pre, iid, k)) for iid in v] for k,v in cat2file_ids.items()}
# categories = ('posts', 'comments', 'reactions', 'users')
# cat2dir = dict(zip(categories, map(lambda x: os.path.join(data_root_dir, x), categories)))

# CATEGORY_2_FILES_HASH = {cat: [os.path.join(data_root_dir, cat, '{}_{}_{}.pkl'.format(pre, iid, cat)) for iid in cat2file_ids[cat]] for cat in categories}


###### TOKEN COOCURENCE INFORMATION #####
COOCURENCE_DICT_FILE_NAMES = ['cooc_tf_', 'cooc_df_', 'ppmi_tf_', 'ppmi_df_']

###### CONSTANTS #####
BINARY_DICTIONARY_NAME = 'mydic.dict'

############## IDEOLOGY INFORMATION ##############
#
# labels = ('extreme left', 'left', 'left of middle', 'right of middle', 'right', 'extreme right')
#
# outlets_by_ideology = {
#     'extreme left': {
#         'The Guardian': '10513336322',
#         'Al Jazeera': '7382473689',
#         'NPR': '10643211755',
#         'New York Times': '5281959998',
#         'The New Yorker': '9258148868',
#         'Slate': '21516776437'},
#     'left': {
#         'PBS': '19013582168',
#         'BBC News': '228735667216',
#         'HuffPost': '18468761129',
#         'Washington Post': '6250307292',
#         'The Economist': '6013004059',
#         'Politico': '62317591679'},
#     'left of middle': {
#         'CBS News': '131459315949',
#         'ABC News': '86680728811',
#         'USA Today': '13652355666',
#         'NBC News': '155869377766434',
#         'CNN': '5550296508',
#         'MSNBC': '273864989376427'},
#     'right of middle': {
#         'Wall Street Journal': '8304333127',
#         'Yahoo News': '338028696036'},
#     'right': {
#         'Fox News': '15704546335'},
#     'extreme right': {
#         'Breitbart': '95475020353',
#         'The Blaze': '140738092630206'}}
#
#
# def id2ideology(_id):
#     for ideology_label, innerdict in outlets_by_ideology.items():
#         for outlet_name, outlet_id in innerdict.items():
#             if _id == outlet_id:
#                 return ideology_label
#     return None
#
# def label2alphanumeric(label):
#     return '_'.join(label.lower().split(' '))
#
# poster_id2outlet_label = {poster_id: poster_name for ide_bin_dict in outlets_by_ideology.values() for poster_name, poster_id in ide_bin_dict.items()}
#
# # should be exposed
# poster_id2ideology_label = {poster_id: label2alphanumeric(id2ideology(poster_id)) for poster_id in poster_id2outlet_label}
#
# # the outlets_by_ideology dict only sorted as in labels from extreme left to extreme right
# id_label2outlet_dict = OrderedDict([('_'.join(ide.split(' ')), OrderedDict(sorted(map(lambda x: (x[0], int(x[1])), outlets_by_ideology[ide].items()), key=lambda x: x[0]))) for ide in labels])
#
# CLASS_LABELS = list(id_label2outlet_dict.keys())

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

DISCREETIZATION = {

    'legacy-scheme': [
        ('extreme_left', ['The Guardian',
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
        ('left_of_middle', ['CBS News',
                            'ABC News',
                            'USA Today',
                            'NBC News',
                            'CNN',
                            'MSNBC']),
        ('right_of_middle', ['Wall Street Journal',
                             'Yahoo News']),
        ('right', ['Fox News']),
        ('extreme_right', ['Breitbart',
                           'The Blaze'])
    ],

    # 'legacy-replicate': {'design': [6, 12, 18, 20, 21], 'class_names': ['liberal_Class', 'centre_CLass', 'conservative_Class']}

    'megadata3': [('liberal_Class', ['The New Yorker', 'Slate', 'The Guardian', 'Al Jazeera', 'NPR', 'New York Times', 'PBS', 'BBC News']),
                  ('centre_Class', ['HuffPost', 'Washington Post', 'The Economist', 'Politico', 'MSNBC', 'CNN', 'NBC News']),
                  ('conservative_Class', ['CBS News', 'ABC News', 'USA Today', 'Yahoo News', 'Wall Street Journal', 'Fox News', 'Breitbart', 'The Blaze'])],

    'c-gigadata-4': [('liberal_Class', ['The New Yorker', 'Slate', 'The Guardian', 'Al Jazeera', 'NPR', 'New York Times']),
                     ('centre_liberal_Class', ['PBS', 'BBC News', 'HuffPost', 'Washington Post', 'The Economist']),
                     ('centre_conservative_Class', ['Politico', 'MSNBC', 'CNN', 'NBC News', 'CBS News', 'ABC News']),
                     ('conservative_Class', ['USA Today', 'Yahoo News', 'Wall Street Journal', 'Fox News', 'Breitbart', 'The Blaze'])],

    'c-gigadata5': [('liberal_Class', ['The New Yorker', 'Slate', 'The Guardian', 'Al Jazeera', 'NPR']),
                    ('centre_liberal_Class', ['New York Times', 'PBS', 'BBC News', 'HuffPost']),
                    ('centre_Class', ['Washington Post', 'The Economist', 'Politico', 'MSNBC']),
                    ('centre_conservative_Class', ['CNN', 'NBC News', 'CBS News', 'ABC News', 'USA Today']),
                    ('conservative_Class', ['Yahoo News', 'Wall Street Journal', 'Fox News', 'Breitbart', 'The Blaze'])],

    'c-gigadata6': [('more_liberal_Class', ['The New Yorker', 'Slate', 'The Guardian', 'Al Jazeera']),
                    ('liberal_Class', ['NPR', 'New York Times', 'PBS', 'BBC News']),
                    ('centre_liberal_Class', ['HuffPost', 'Washington Post', 'The Economist']),
                    ('centre_Class', ['Politico', 'MSNBC', 'CNN', 'NBC News']),
                    ('centre_conservative_Class', ['CBS News', 'ABC News', 'USA Today', 'Yahoo News']),
                    ('conservative_Class', ['Wall Street Journal', 'Fox News', 'Breitbart', 'The Blaze'])],

    'gigadata4': [('liberal_Class', ['The New Yorker', 'Slate', 'The Guardian', 'Al Jazeera', 'NPR', 'New York Times']),
                 ('centre_liberal_Class', ['PBS', 'BBC News', 'HuffPost', 'Washington Post', 'The Economist']),
                 ('centre_conservative_Class', ['Politico', 'MSNBC', 'CNN', 'NBC News', 'CBS News', 'ABC News']),
                 ('conservative_Class', ['USA Today', 'Yahoo News', 'Wall Street Journal', 'Fox News', 'Breitbart', 'The Blaze'])],

    'gigadata5': [('liberal_Class', ['The New Yorker', 'Slate', 'The Guardian', 'Al Jazeera', 'NPR']),
                  ('centre_liberal_Class', ['New York Times', 'PBS', 'BBC News', 'HuffPost']),
                  ('centre_Class', ['Washington Post', 'The Economist', 'Politico', 'MSNBC']),
                  ('centre_conservative_Class', ['CNN', 'NBC News', 'CBS News', 'ABC News', 'USA Today']),
                  ('conservative_Class', ['Yahoo News', 'Wall Street Journal', 'Fox News', 'Breitbart', 'The Blaze'])]

}
