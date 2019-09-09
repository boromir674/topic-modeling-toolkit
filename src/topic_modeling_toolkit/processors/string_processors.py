import re
from gensim.utils import deaccent as gen_deaccent
from gensim.utils import lemmatize as gen_lemmatize
from nltk.corpus import stopwords

from .processor import StateLessProcessor

en_stopwords = set(stopwords.words('english'))


class StringProcessor(StateLessProcessor):
    pass


def lowercase(a_string):
    return a_string.lower()


reg = re.compile(r'\s{2,}')


def mono_space(a_string):
    return re.sub(reg, ' ', a_string)


def deaccent(a_string):
    return gen_deaccent(a_string)


def utf8encode(a_string):
    return a_string
    # assert type(a_string) == str
    # return a_string.encode('utf-8')


def lemmatize(a_string):
    try:
        return ' '.join(x[0] for x in [str(x.decode()).split('/') for x in gen_lemmatize(a_string, stopwords=en_stopwords, min_length=2, max_length=50)])
    except TypeError as e:
        raise TypeError("Error: {}. Input {} of type {}".format(e, a_string, type(a_string).__name__))


class LowerCaser(StringProcessor):
    def __init__(self):
        super(StringProcessor, self).__init__(lowercase)


class MonoSpacer(StringProcessor):
    def __init__(self):
        super(StringProcessor, self).__init__(mono_space)


class UtfEncoder(StringProcessor):
    def __init__(self):
        super(StringProcessor, self).__init__(utf8encode)


class DeAccenter(StringProcessor):
    def __init__(self):
        super(StringProcessor, self).__init__(deaccent)


class StringLemmatizer(StringProcessor):
    def __init__(self):
        super(StringProcessor, self).__init__(lemmatize)
