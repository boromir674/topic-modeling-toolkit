import sys
from gensim.utils import lemmatize as gen_lemmatize
from gensim.utils import tokenize as gen_tokenize
from gensim.corpora.textcorpus import TextCorpus
import pandas as pd

from definitions import en_stopwords, nb_docs, cat2files


class MyCorpus(TextCorpus):
    stopwords = en_stopwords  # implicitly gets assigned as a class attribute (self.stopwords) by constructor

    # def __init__(self):
    #     super(TextCorpus, self).__init__()

    def getstream(self):

        l = 0
        for category in decode_category(self.input):
            for pickle_file in cat2files[category]:
                print 'Working with \'{}\' file'.format(pickle_file)
                # for doc in gen_docs(pickle_file, nb_docs):
                #     yield doc
                for i, (_, df_entry) in enumerate(pd.read_pickle(pickle_file).iterrows()):
                    # print nb_docs
                    # sys.exit(0)
                    if i > 15:
                        break
                    # print type(df_entry['text'].encode('utf-8'))
                    yield df_entry['text'].encode('utf-8')
                l += i + 1
        self.length = i

    def get_texts(self):
        for doc in self.getstream():
            l = [word for word in generate_words(doc, normalize='lemmatize', filter_stopwords=True)]
            # print len(l)
            yield l
            # yield [word for word in utils.to_unicode(doc).lower().split() if word not in self.stopwords]

    def __len__(self):
        return self.length

# class PostsCorpus(object):
#      def __iter__(self):
#          for line in read_pickle('mycorpus.txt'):
#              # assume there's one document per line, tokens separated by whitespace
#              yield dictionary.doc2bow(line.lower().split())

# def lemmatize(content, allowed_tags=<_sre.SRE_Pattern object>, light=False, stopwords=frozenset([]), min_length=2, max_length=15)
 

def get_normalizer(norm_type):
    """
    Returns a lambda function that acts as a normalizer for input words/tokens.\n
    :param norm_type: the normalizer type to "construct". Recommended 'lemmatize'. If type is not in the allowed types then does not normalize (lambda just forwards the input to output as it is)
    :type norm_type: {'lemmatize'}
    :return: the lambda callable object to perform normalization
    :rtype: lambda
    """
    # if norm_type == 'stem':
    #     stemmer = PorterStemmer()
    #     return lambda x: stemmer.stem(x)
    if norm_type == 'lemmatize':
        # lemmatizer = WordNetLemmatizer()
        return lambda x: gen_lemmatize(x)
    else:
        return lambda x: x


def generate_words(text_data, normalize='lemmatize', filter_stopwords=True):
    """
    Given input text_data, a normalize 'command' and a stopwords filtering flag, generates a normalized, lowercased word/token provided that it passes the filter and that its length is bigger than 2 characters.\n
    :param text_data: the text from which to generate (i.e. doc['text'])
    :type text_data: str
    :param normalize: the type of normalization to perform. Recommended 'lemmatize'
    :type normalize: {'lemmatize'}, else does not normalize
    :param filter_stopwords: switch/flag to control stopwords filtering
    :type filter_stopwords: boolean
    :return: the generated word/token
    :rtype: str
    """
    banned = set()
    if filter_stopwords:
        banned = en_stopwords
    normalizer = get_normalizer(normalize)
    for word in gen_tokenize(text_data, lowercase=True, deacc=False, encoding='utf8'):
        if len(word) < 10 or word in banned:
            continue
        norm = normalizer(word)
        if type(norm) == list:
            if len(norm) == 0:
                continue
            else:
                yield norm[0].split('/')[0]
        else:
            yield norm


def decode_category(category):
    return category.split('+')


def gen_docs(pickle_df, limit):
    for i, (_, df_entry) in enumerate(pd.read_pickle(pickle_df).iterrows()):
        if i > limit:
            break
        yield df_entry['text']


def lowercase_p(text):
    return text.lower()


def unicode_p(text):
    return unicode(text)

# def lemmatize_p(text):
# client
# thres = 3
# filt = get_length_filter(thres)


def min_len_f(text, thres):
    """A word is returned if its length is greater or equal to the threshold"""
    return text if len(text) >= thres else None
