import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

from patm.definitions import cat2files, id2outlet

st = {'poster_id': Counter()}


def print_info(df):
    """
    :type df: pandas.DataFrame
    """
    b = 'Dataframe with {} rows, and {} complete \'datapoints\'\n'.format(len(df), get_complete(df, 'any'))
    b += 'with {} columns and unique values\n'.format(len(df.columns))
    t = ''
    for c in df.columns:
        t += ' {} var {} with {} unique and {} missing values\n'.format(df[c].dtype, c, df[c].nunique(), len(df) - get_complete(df, c))
    b += t
    b += 'with index\n{}\n'.format(df.index)
    for c in ['poster_id']:
        if c in df.columns:
            b += '{} with distr:\n{}'.format(c, df.groupby(c)['text'].count())
            st[c] += Counter(dict([(id2outlet(idd), int(val)) for idd, val in df.groupby(c)['text'].count().items()]))
    print b


def get_complete(df, field):
    if field == 'any':
        return len(df) - df.isnull().any(axis=1).sum()
    else:
        return len(df) - df[field].isnull().sum()


def get_sample(df):
    b = ''
    for c in df.columns:
        if c in ('timestamp', 'text', 'url'):
            continue
        var = df[c].dtype
        uniq = df[c].nunique()
        if uniq < 10:
            b += '{} type {} ALL:\n{}\n'.format(var, c, df[c].unique())
        else:
            b += '{} type {} SAMPLE:\n{}\n'.format(var, c, df[c].iloc[np.random.choice(np.arange(len(df)), 5, False)])
    return b


def explore(category):
    norm = 0
    for pfile in cat2files[category]:
        df = pd.read_pickle(pfile)
        print '-------------', os.path.basename(pfile), '-------------'
        print_info(df)
        print '\n', get_sample(df), '\n\n'
        norm += sum([_ for _ in st['poster_id'].values()])

    # print [_ for _ in st['poster_id'].values()]

    print 'norm', norm
    for k, v in st['poster_id'].items():
        st['poster_id'][k] = v / float(norm)
    b = '\n'.join(sorted(st['poster_id'], key=lambda x: x[0]))



if __name__ == '__main__':
    explore(sys.argv[1])
