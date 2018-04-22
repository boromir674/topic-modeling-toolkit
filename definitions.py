import os


from processors import LowerCaser, MonoSpacer, UtfEncoder, DeAccenter, StringLemmatizer, MinLengthFilter, MaxLengthFilter, WordToUnigramGenerator

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
    'normalize': lambda x: StringLemmatizer() if x else None,
    'min_length': lambda x: MinLengthFilter(x),
    'max_length': lambda x: MaxLengthFilter(x),
    'no_below': lambda x: x,
    'no_above': lambda x: x,
    # 'occurence_thres': lambda x: OccurenceFilter(x),
    # 'ngrams': lambda x: NgramsGenerator(x),
    # 'weight': lambda x: FeatureComputer(x),
    # 'format': lambda x: DataFormater(x)
    'ngrams': lambda x: WordToUnigramGenerator(x),
    'weight': lambda x: x,
    'format': lambda x: None
}
