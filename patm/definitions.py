import os
from collections import OrderedDict

from processors.string_processors import LowerCaser, MonoSpacer, UtfEncoder, DeAccenter, StringLemmatizer
from processors.generator_processors import MinLengthFilter, MaxLengthFilter, WordToNgramGenerator
from processors.disk_writer_processors import UciFormatWriter


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
                                     (42, 31, 36)
                                     )))

cat2files = {cat: [os.path.join(cat2dir[cat], '{}_{}_posts.pkl'.format(pre, iid)) for iid in cat2file_ids[cat]] for cat in categories}

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
    'weight': lambda x: x if x else None,  # if x == 'tfidf' else x,
    'format': lambda x: UciFormatWriter() if x == 'uci' else None
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


def get_id1(pipeline):
    b = ''
    for pr_name, proc in pipeline:
        b += '_'
        if type(proc) in (int, float):
            b += '{}-{}'.format(pr_name, proc)
        else:
            b += proc.to_id()
    return b[1:]
