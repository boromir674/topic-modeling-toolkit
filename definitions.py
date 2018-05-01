import os
from collections import OrderedDict

from processors.string_processors import LowerCaser, MonoSpacer, UtfEncoder, DeAccenter, StringLemmatizer
from processors.generator_processors import MinLengthFilter, MaxLengthFilter, WordToNgramGenerator
from processors.disk_writer_processors import UciFormatWriter


root_dir = '/data/thesis'


data_root_dir = '/data/thesis/data'
pre = 'pickles'

categories = ('posts', 'comments', 'reactions', 'users')

cat2dir = dict(zip(categories, map(lambda x: os.path.join(data_root_dir, x), categories)))

cat2file_ids = dict(zip(categories, ((2, 5, 38),
                                     (9, 16, 28),
                                     (1, 12, 19, 43),
                                     (42, 31, 36)
                                     )))

cat2files = {cat: [os.path.join(cat2dir[cat], '{}_{}_posts.pkl'.format(pre, iid)) for iid in cat2file_ids[cat]] for cat in categories}

nb_docs = float('inf')

encode_pipeline_cfg = {
    'lowercase': lambda x: bool(eval(x)),
    'mono_space': lambda x: bool(eval(x)),
    'unicode': lambda x: bool(eval(x)),
    'deaccent': lambda x: bool(eval(x)),
    'normalize': str,
    'min_length': int,
    'max_length': int,
    'no_below': int,
    'no_above': float,
    'ngrams': int,
    'weight': str,
    'format': str
}


settings_value2processors = {
    'lowercase': lambda x: LowerCaser() if x else None,
    'mono_space': lambda x: MonoSpacer() if x else None,
    'unicode': lambda x: UtfEncoder() if x else None,
    'deaccent': lambda x: DeAccenter() if x else None,
    'normalize': lambda x: StringLemmatizer() if x == 'lemmatize' else None,
    'min_length': lambda x: MinLengthFilter(x) if x else None,
    'max_length': lambda x: MaxLengthFilter(x) if x else None,
    'no_below': lambda x: x if x else None,
    'no_above': lambda x: x if x else None,
    'ngrams': lambda x: WordToNgramGenerator(x) if x else None,
    'weight': lambda x: UciFormatWriter(),  # if x == 'tfidf' else x,
    'format': lambda x: None
}

# def tuple2strings(pipe_component_name, value):
#     if type(value) == bool:
#         print pipe_component_name
#         if value:
#             return [pipe_component_name]
#         else:
#             return []
#     else:
#         return [pipe_component_name, str(value)]


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


# def get_id(pipe_settings):
#     assert isinstance(pipe_settings, OrderedDict)
#
#     def strings2string(string_list):
#         if string_list[0] == 'format':
#             return string_list[1]
#         else:
#             return '-'.join(string_list)
#     return '_'.join(strings2string(tuple2string(pipeline_component, value)) for pipeline_component, value in pipe_settings.items() if value)
#     # return '_'.join(strings2string(tuple2string(pipeline_component, value) for pipeline_component, value in pipe_settings.items() if tuple2string(pipeline_component, value)))


def get_id1(pipeline):
    # assert isinstance(pipeline, Pipeline)
    # ids = ['{}-{}'.format(pr_name, proc) for pr_name, proc in pipeline if type(proc) in (int, float) else proc.to_id()]
    # return '_'.join(ids)
    b = ''
    for pr_name, proc in pipeline:
        b += '_'
        if type(proc) in (int, float):
            b += '{}-{}'.format(pr_name, proc)
        else:
            b += proc.to_id()
    return b[1:]
