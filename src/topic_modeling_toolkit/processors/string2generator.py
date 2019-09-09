from gensim.utils import lemmatize as gen_lemmatize
from nltk.corpus import stopwords

from .processor import StateLessProcessor

en_stopwords = set(stopwords.words('english'))


class StringToGenerator(StateLessProcessor):
    pass


def string2tokengenerator(a_string, splitter):
    return (_ for _ in a_string.split(splitter))


class StringToTokenGenerator(StringToGenerator):
    def __init__(self, splitter):
        self.splitter = splitter
        super(StringToGenerator, self).__init__(lambda x: string2tokengenerator(x, splitter))

    def __str__(self):
        return type(self).__name__ + "('" + str(self.splitter) + "')"

#
# class StringToLemmatizedTokenGenerator(StringToTokenGenerator):
#     def __init__(self):
#         super(StringToTokenGenerator, self).__init__(lemmatize_and_tokenize)
#
#
# def lemmatize_and_tokenize(a_string):
#     return [val.split('/')[0] for val in gen_lemmatize(a_string, stopwords=en_stopwords, min_length=2, max_length=50)]
