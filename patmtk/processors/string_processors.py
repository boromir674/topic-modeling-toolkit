import re
from gensim.utils import deaccent as gen_deaccent
from gensim.utils import lemmatize as gen_lemmatize
from nltk.corpus import stopwords

from processor import StateLessProcessor

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
    return a_string.encode('utf-8')


def lemmatize(a_string):
    return ' '.join(map(lambda x: x.split('/')[0], gen_lemmatize(a_string, stopwords=en_stopwords, min_length=2, max_length=50)))


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
