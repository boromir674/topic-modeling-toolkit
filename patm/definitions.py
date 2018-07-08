import os
from collections import OrderedDict

from processors.string_processors import LowerCaser, MonoSpacer, UtfEncoder, DeAccenter, StringLemmatizer
from processors.generator_processors import MinLengthFilter, MaxLengthFilter, WordToNgramGenerator
from processors.disk_writer_processors import UciFormatWriter, VowpalFormatWriter

results_root = 'results'
models_root = 'models'

regularizers_param_cfg = '/data/thesis/code/regularizers.cfg'

root_dir = '/data/thesis'
data_root_dir = '/data/thesis/data'
bows_dir = '/data/thesis/data/collections'
words_dir = '/data/thesis/data/collections'

collections_dir = '/data/thesis/data/collections'

pre = 'pickles'

categories = ('posts', 'comments', 'reactions', 'users')

cat2dir = dict(zip(categories, map(lambda x: os.path.join(data_root_dir, x), categories)))

cat2file_ids = dict(zip(categories, ((2, 5, 38),
                                     (9, 16, 28),
                                     (1, 12, 19, 43),
                                     (24, 31, 36)
                                     )))

cat2files = {cat: [os.path.join(cat2dir[cat], '{}_{}_{}.pkl'.format(pre, iid, cat)) for iid in cat2file_ids[cat]] for cat in categories}


###### TOKEN COOCURENCE INFORMATION #####
COOCURENCE_DICT_FILE_NAMES = ['cooc_tf_', 'cooc_df_', 'ppmi_tf_', 'ppmi_df_']

###### CONSTANTS #####
BINARY_DICTIONARY_NAME = 'mydic.dict'

def get_generic_topic_names(nb_topics):
    return ['top_' + index for index in map(lambda x: str(x) if len(str(x)) > 1 else '0'+str(x), range(nb_topics))]
###### ###### ###### ###### ###### ######


encode_pipeline_cfg = {
    'lowercase': lambda x: bool(eval(x)),
    'monospace': lambda x: bool(eval(x)),
    'unicode': lambda x: bool(eval(x)),
    'deaccent': lambda x: bool(eval(x)),
    'normalize': str,
    'minlength': int,
    'maxlength': int,
    'nobelow': int,
    'noabove': float,
    'ngrams': int,
    'weight': str,
    'format': str
}

settings_value2processors = {
    'lowercase': lambda x: LowerCaser() if x else None,
    'monospace': lambda x: MonoSpacer() if x else None,
    'unicode': lambda x: UtfEncoder() if x else None,
    'deaccent': lambda x: DeAccenter() if x else None,
    'normalize': lambda x: StringLemmatizer() if x == 'lemmatize' else None,
    'minlength': lambda x: MinLengthFilter(x) if x else None,
    'maxlength': lambda x: MaxLengthFilter(x) if x else None,
    'nobelow': lambda x: x if x else None,
    'noabove': lambda x: x if x else None,
    'ngrams': lambda x: WordToNgramGenerator(x) if x else None,
    'weight': lambda x: x if x else None,
    'format': lambda x: {'uci': UciFormatWriter(), 'vowpal': VowpalFormatWriter()}[x] if x else None
}


def tuple2string(pipeline_component, value):
    if type(value) == bool:
        if value:
            return pipeline_component
        else:
            return ''
    elif pipeline_component == 'format':
        return value
    else:
        return '{}-{}'.format(pipeline_component, value)


def get_id(pipe_settings):
    assert isinstance(pipe_settings, OrderedDict)

    return '_'.join(tuple2string(pipeline_component, value) for pipeline_component, value in pipe_settings.items() if tuple2string(pipeline_component, value))


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

obi2 = {
    'extreme left': {
        'New York Times':  '5.281.959.998',
        'Al Jazeera'    :  '7.382.473.689',
        'The New Yorker':  '9.258.148.868',
        'The Guardian'  : '10.513.336.322',
        'NPR'           : '10.643.211.755',
        'Slate'         : '21.516.776.437'
    },
    'left': {
        'The Economist'  :   '6.013.004.059',
        'Washington Post':   '6.250.307.292',
        'HuffPost'       :  '18.468.761.129',
        'PBS'            :  '19.013.582.168',
        'Politico'       :  '62.317.591.679',
        'BBC News'       : '228.735.667.216'
    },
    'left of middle': {
        'CNN'      :       '5.550.296.508',
        'USA Today':      '13.652.355.666',
        'ABC News' :      '86.680.728.811',
        'CBS News' :     '131.459.315.949',
        'NBC News' : '155.869.377.766.434',
        'MSNBC'    : '273.864.989.376.427'
    },
    'right of middle': {
        'Wall Street Journal':   '8.304.333.127',
        'Yahoo News'         : '338.028.696.036'
    },
    'right': {
        'Fox News': '15.704.546.335'
    },
    'extreme right': {
        'Breitbart':      '95.475.020.353',
        'The Blaze': '140.738.092.630.206'
    }
}

def id2ideology(_id):
    for ideology_label, innerdict in outlets_by_ideology.items():
        for outlet_name, outlet_id in innerdict.items():
            if _id == outlet_id:
                return ideology_label
    return None

def label2alphanumeric(label):
    return '_'.join(label.lower().split(' '))

poster_id2outlet_label = {poster_id: poster_name for ide_bin_dict in outlets_by_ideology.values() for poster_name, poster_id in ide_bin_dict.items()}
poster_id2ideology_label = {poster_id: label2alphanumeric(id2ideology(poster_id)) for poster_id in poster_id2outlet_label}

# the outlets_by_ideology dict only sorted as in labels from extreme left to extreme right
id_label2outlet_dict = OrderedDict([(ide, OrderedDict(sorted(map(lambda x: (x[0], int(x[1])), outlets_by_ideology[ide].items()), key=lambda x: x[0]))) for ide in labels])

IDEOLOGY_CLASS_NAME = 'ideology'