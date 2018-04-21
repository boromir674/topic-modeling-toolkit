import re
from abc import ABCMeta, abstractmethod

from gensim.utils import deaccent as gen_deaccen
from gensim.utils import lemmatize as gen_lemmatize


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

    def process(self, data):
        return self.func(data)


class StateFullProcessor(object):

    def __init__(self, a_function):
        self.__mro__ = [Processor]
        self.func = a_function

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
    return ' '.join(val.split('/')[0] for val in gen_lemmatize(a_string, min_length=2, max_length=20))


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


### GENERATOR PROCESSORS ###

class GeneratorProcessor(StateLessProcessor): pass


def min_length_filter(word_generator, min_length):
    return (w for w in word_generator if len(w) >= min_length)

def max_length_filter(word_generator, max_length):
    return (w for w in word_generator if len(w) < max_length)


class MinLengthFilter(GeneratorProcessor):
    def __init__(self, min_length):
        super(GeneratorProcessor, self).__init__(lambda x: min_length_filter(x, min_length))


class MaxLengthFilter(GeneratorProcessor):
    def __init__(self, max_length):
        super(GeneratorProcessor, self).__init__(lambda x: max_length_filter(x, max_length))




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

    print [_ for _ in generator]
    mlf = MinLengthFilter(3)

    g = mlf.process(generator)

    print [_ for _ in g]
