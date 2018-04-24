import re
from abc import ABCMeta, abstractmethod

from gensim.utils import deaccent as gen_deaccent
from gensim.utils import lemmatize as gen_lemmatize

from nltk.corpus import stopwords
en_stopwords = set(stopwords.words('english'))


class Processor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__mro__ = [Processor]

    @abstractmethod
    def process(self, data):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(a_class, C):
        if a_class is Processor:
            if any('process' in B.__dict__ for B in C.__mro__):
                # print ['process' in B.__dict__ for B in C.__mro__]
                return True
        return NotImplemented


class StateLessProcessor(object):

    def __init__(self, a_function):
        self.__mro__ = [Processor]
        self.func = a_function

    def __str__(self):
        return type(self).__name__

    def process(self, data):
        return self.func(data)


class StateFullProcessor(object):

    def __init__(self, a_function):
        self.__mro__ = [Processor]
        self.func = a_function

    def __str__(self):
        return type(self).__name__

    def process(self, data):
        return self.func(data)

    @property
    def get_state(self):
        return self.state


Processor.register(StateLessProcessor)
Processor.register(StateFullProcessor)


class StringProcessor(StateLessProcessor): pass


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
    return ' '.join(val.split('/')[0] for val in gen_lemmatize(a_string, stopwords=en_stopwords, min_length=2, max_length=50))

def lemmatize_and_tokenize(a_string):
    return [val.split('/')[0] for val in gen_lemmatize(a_string, stopwords=en_stopwords, min_length=2, max_length=50)]

def lemmatize_token(a_string):
    return gen_lemmatize(a_string, stopwords=en_stopwords, min_length=2, max_length=50).split('/')[0]

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


### MUTATOR ###
class StringToGenerator(StateLessProcessor): pass


def string2tokengenerator(a_string, splitter):
    return (_ for _ in a_string.split(splitter))


class StringToTokenGenerator(StringToGenerator):
    def __init__(self, splitter):
        self.splitter = splitter
        super(StringToGenerator).__init__(lambda x: string2tokengenerator(x, splitter))

    def __str__(self):
        return super(StringToGenerator, self).__str__() + '(' + str(self.splitter) + ')'


### GENERATOR PROCESSORS ###


class GeneratorProcessor(StateLessProcessor): pass


def gen_ngrams(word_generator, degree):
    assert degree > 1
    gen_empty = False
    total = 0
    queue = []
    while total < degree:
        queue.append(word_generator.next())
        total += 1
    yield '_'.join(queue)
    total = 1  # total unigrams generated
    while not gen_empty:
        try:
            del queue[0]
            queue.append(word_generator.next())
            yield '_'.join(queue)
            total += 1
        except StopIteration:
            print 'Generated {} unigrams'.format(total)
            gen_empty = True


def ngrams_convertion(word_generator, degree):
    if degree == 1:
        return word_generator
    else:
        return (_ for _ in gen_ngrams(word_generator, degree))


def min_length_filter(word_generator, min_length):
    if min_length < 2:
        min_length = 2
    return (w for w in word_generator if len(w) >= min_length)

def max_length_filter(word_generator, max_length):
    if max_length > 50:
        max_length = 50
    return (w for w in word_generator if len(w) <= max_length)


class MinLengthFilter(GeneratorProcessor):
    def __init__(self, min_length):
        super(GeneratorProcessor, self).__init__(lambda x: min_length_filter(x, min_length))
        self.min_length = min_length

    def __str__(self):
        return super(GeneratorProcessor, self).__str__() + '(' + str(self.min_length) + ')'


class MaxLengthFilter(GeneratorProcessor):
    def __init__(self, max_length):
        super(GeneratorProcessor, self).__init__(lambda x: max_length_filter(x, max_length))
        self.max_length = max_length

    def __str__(self):
        return super(GeneratorProcessor, self).__str__() + '(' + str(self.max_length) + ')'


class WordToUnigramGenerator(GeneratorProcessor):
    def __init__(self, degree):
        super(GeneratorProcessor, self).__init__(lambda x: ngrams_convertion(x, degree))
        self.degree = degree

    def __str__(self):
        return super(GeneratorProcessor, self).__str__() + '(' + str(self.degree) + ')'


# ### Disk Writer Processors

class StateLessDiskWriter(StateLessProcessor): pass


class UciFormatWriter(StateLessDiskWriter):
    """Ingests one doc vector at a time"""
    def __init__(self, fname='/data/thesis/data/myuci'):
        self.doc_num = 1
        self.fname = fname
        super(StateLessDiskWriter, self).__init__(lambda x: write_vector(fname, map(lambda word_id_weight_tuple: (word_id_weight_tuple[0]+1, word_id_weight_tuple[1]), x), self.doc_num))
        # + 1 to align with the uci format, which states that the minimum word_id = 1; implemented with map-lambda combo on every element of a document vector

    def __str__(self):
        return super(StateLessDiskWriter, self).__str__() + '(' + str(self.fname) + ')'

    def process(self, data):
        _ = super(StateLessDiskWriter, self).process(data)
        self.doc_num += 1
        return _


def write_vector(fname, doc_vector, doc_num):
    with open(fname, 'a') as f:
        # print doc_vector
        # f.writelines(map(lambda x: '{} {} {:.8f}\n'.format(doc_num, x[0], x[1]), doc_vector))
        f.writelines(map(lambda x: '{} {} {}\n'.format(doc_num, x[0], x[1]), doc_vector))


if __name__ == '__main__':

    sl = StateLessProcessor(lambda x: x + '_less')
    sf = StateFullProcessor(lambda x: x + '_full')
    sp = StringProcessor(lambda x: x + 'sp')

    print isinstance(sf, Processor)
    print issubclass(sf, Processor)

    msp = MonoSpacer()

    print isinstance(msp, Processor)
    print issubclass(msp, Processor)
    print isinstance(msp, StateLessProcessor)
    print issubclass(StateLessProcessor, Processor)

    docs = [['alpha', 're', 'gav', 'gav'],
            ['asto', 'paramithi', 're']
            ]

    generator = (w for w in docs[0])
    bfg = WordToUnigramGenerator(5)
    ng = bfg.process(generator)

    print [_ for _ in ng]

    # print [_ for _ in generator]
    # mlf = MinLengthFilter(3)
    #
    # g = mlf.process(generator)
    #
    # print [_ for _ in g]
