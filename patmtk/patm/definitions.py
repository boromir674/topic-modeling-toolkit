import os
from collections import OrderedDict
import attr
from functools import reduce


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

DISCRETIZATION = {

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

    # '3-poles-scheme': [
    #     ('liberal', []),
    #     ('centre', []),
    #     ('conservative', [])
    # ]
}
from .discreetization import PoliticalSpectrumManager
pm = PoliticalSpectrumManager(SCALE_PLACEMENT, DISCRETIZATION)
poster_id2ideology_label = pm.poster_id2ideology_label
CLASS_LABELS = pm.class_names


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

#
# def _correct_order(instance, attribute, converted_value):
#     instance._schemes = {k: OrderedDict([(class_label, [_ for _ in instance.scale if _ in outlet_names]) for class_label, outlet_names in scheme.items()])
#                          for k, scheme in converted_value.items()}
#
# @attr.s
# class PoliticalSpectrumManager(object):
#     scale = attr.ib(init=True, converter=lambda x: OrderedDict(x), repr=True)
#     _schemes = attr.ib(init=True, converter=lambda x: {scheme_name:
#                                                            OrderedDict([(class_name.lower().replace(' ', '_'), outlet_names_list) for class_name, outlet_names_list in discretization_design])
#                                                        for scheme_name, discretization_design in x.items()},
#                        validator=_correct_order, repr=True)
#     _cur = attr.ib(init=True, default='legacy-scheme', converter=str, repr=True)
#     _outlet_id2outlet_name = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict((v, k) for k, v in self.scale.items()), takes_self=True), repr=False)
#     # _label_id2outlet_names = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict([(bin_name.lower().replace(' ', '_'), outlet_names_list) for bin_name, outlet_names_list in
#     #      self._schemes[self._cur].values()]), takes_self=True), repr=False)
#     # _outlet_id2outlet_name = attr.ib(init=False, default=attr.Factory(lambda self: {poster_id: poster_name for ide_bin_dict in self._schemes[self._cur].values() for
#     #                                 poster_name, poster_id in ide_bin_dict.items()}, takes_self=True), repr=False)
#     # _label_id2outlet_dict = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict([('_'.join(ide.split(' ')),
#     #                                                                                             OrderedDict(sorted([(outlet, outlet_id) for outlet, outlet_id in
#     #                                                                self._schemes[self._cur][ide].items()],
#     #                                                               key=lambda x: x[0]))) for ide in
#     #                                                                                            self._schemes[self._cur].keys()]), takes_self=True), repr=False)
#     @property
#     def discretization_scheme(self):
#         return self._schemes[self._cur]
#
#     @discretization_scheme.setter
#     def discretization_scheme(self, scheme):
#         if scheme not in self._schemes:
#             raise KeyError("The schemes implemented are [{}]. Requested '{}' instead.".format(' ,'.join(self._schemes.keys()), scheme))
#         self._cur = scheme
#         # self._outlet_id2outlet_name = {poster_id: poster_name for ide_bin_dict in self._schemes[self._cur].values() for poster_name, poster_id in ide_bin_dict.items()}
#         # self._label_id2outlet_names = OrderedDict([(bin_name.lower().replace(' ', '_'),  outlet_names_list) for bin_name, outlet_names_list in self._schemes[self._cur].values()])
#     @property
#     def poster_id2ideology_label(self):
#         return OrderedDict([(self.scale[name], class_label) for class_label, outlet_names in self._schemes[self._cur].items() for name in outlet_names])
#
#     @property
#     def class_names(self):
#         """The normalized class names matching the discrete bins applied on the 10-point scale of ideological consistency"""
#         return list(self._schemes[self._cur].keys())
#     # def _label(self, outlet_name):
#     #     for ideology_label, outlet_names_list in self._schemes[self._cur].items():
#     #         for current_outlet_name in outlet_names_list:
#     #             if outlet_name == current_outlet_name:
#     #                 return ideology_label
#     #
#     # def _iter(self, outlet_names):
#     #     return (_ for _ in SCALE_PLACEMENT if _[0] in outlet_names)


#
# @attr.s(cmp=True, repr=True, str=True, slots=True)
# class DiscreetizationScheme(object):
#     _bins = attr.ib(init=True, converter=lambda x: OrderedDict([(class_name, outlet_names_list) for class_name, outlet_names_list in x]), repr=True, cmp=True)
#     # bins = attr.ib(init=True, converter=list, cmp=True, repr=True)
#     # _class_names = attr.ib(init=True, default=attr.Factory(lambda self: ['my_class_{}'.format(x) for x in range(len(self.bins))], takes_self=True),
#     #                       converter=lambda x: [_.lower().replace(' ', '_') for _ in x], cmp=True, repr=True)
#     def __iter__(self):
#         for class_name, items in self._bins.items():
#             yield class_name, items
#
#     @property
#     def class_names(self):
#         return list(self._bins.keys())
#
#     @class_names.setter
#     def class_names(self, class_names):
#         if len(class_names) != len(self._bins):
#             raise RuntimeError("Please give equal number of class names ({} given) as the number of defined bins ({})".format(len(class_names), len(self._bins)))
#         self._bins = OrderedDict([(class_name, outlet_list) for class_name, outlet_list in zip(class_names, self._bins.items())])
#
#     @classmethod
#     def from_design(cls, design, scale, class_names=None):
#         if class_names:
#             return DiscreetizationScheme(zip(list(Bins.from_design(design, scale)), class_names))
#         return DiscreetizationScheme(Bins.from_design(design, scale))
#
#     # @classmethod
#     # def from_tuples(cls, data):
#     #     return DiscreetizationScheme([_[1] for _ in data], [_[0] for _ in data])
#         # a, b = (list(x) for x in zip(*[[doc, label] for doc, label in zip(self.corpus, self.outlet_ids) if doc]))
#         # return DiscreetizationScheme(*reversed([list(x) for x in zip(*data)]))
#
# def _check_nb_bins(instance, attribute, value):
#     if not 0 < len(value) <= 100:
#         raise ValueError("Resulted in {} bins but they should be in [1,100]".format(len(value)))
#     if sum(len(x) for x in value) != len(reduce(lambda x,y: x.union(y), [set(_) for _ in value])):
#         raise ValueError("Found same item in multiple bins")
#
# @attr.s(cmp=True, repr=True, str=True, slots=True, frozen=True)
# class Bins(object):
#     bins = attr.ib(init=True, cmp=True, repr=True, validator=_check_nb_bins)
#
#     # def __iter__(self):
#     #     for i, bin in enumerate(self.bins):
#     #         yield 'bin_{}'.format(i), bin
#
#     def __getitem__(self, item):
#         return self.bins[item]
#
#     @classmethod
#     def from_design(cls, design, scale_placement):
#         return Bins([[scale_placement[i][0] for i in a_range] for a_range in BinDesign(design).ranges(len(scale_placement))])
#
#
# def _check_design(instance, attribute, value):
#     if not 0 < len(value) < 100:
#         raise ValueError("Resulted in {} bins but they should be in [1,100]".format(len(value)+1))
#     for i in range(1, len(value)):
#         if value[i] <= value[i - 1]:
#             raise ValueError("Invalid design list. Each element should be greater than the previous one; prev: {}, current: {}".format(value[i - 1], value[i]))
#
#
# @attr.s(cmp=True, repr=True, str=True, slots=True, frozen=True)
# class BinDesign(object):
#     seps = attr.ib(init=True, validator=_check_design, cmp=True, repr=True)
#
#     def ranges(self, nb_elements):
#         if self.seps[-1] >= nb_elements:
#             raise ValueError("Last bin starts from index {}, but requested to build range indices for {} elements".format(self.sep[-1], nb_elements))
#         yield range(self.seps[0])
#         for i, item in enumerate(self.seps[1:]):
#             yield range(self.seps[i], item)
#         yield range(self.seps[-1], nb_elements)

#
# from processors import PipeHandler, Pipeline
# class UniformDiscreetizer(object):
#
#     def __init__(self):
#         self.pipe_hndler = PipeHandler('/tmp', 'posts', sample='all')
#         self.pipe_hndler.pipeline = Pipeline(OrderedDict([]))
#         self.population = {}
#         pipe_handler = PipeHandler('imaginary_collections_root', 'posts', sample='all')
#         pipe_handler.pipeline = Pipeline.from_tuples([
#             ('monospace', 1),
#             ('unicode', 1),
#             ('deaccent', 1),
#             ('normalize', 'lemmatize'),
#             ('minlength', 2),
#             ('maxlength', 25),
#             ('ngrams', 1),
#             ('nobelow', 1),
#             ('noabove', 0.5),
#             ('weight', 'counts')])
#         self.pipe_hndler.pass_through('imaginary_collection')
#
#
#     def fitness(self, design):
#         ds = DiscreetizationScheme.fr