import re
from abc import ABCMeta, abstractmethod

from gensim.utils import deaccent as gen_deaccen
from gensim.utils import lemmatize as gen_lemmatize


class Processor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__mro__ = [Processor]
        self.a = 'gav'

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


Processor.register(StateLessProcessor)
Processor.register(StateFullProcessor)

class StringProcessor(StateLessProcessor): pass

class GeneratorProcessor(StateLessProcessor): pass


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
    return gen_lemmatize(a_string)


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










class MonoSpace(StringProcessor):
    def __init__(self):
        super(StringProcessor, self).__init__(lowercase)


if __name__ == '__main__':

    sl = StateLessProcessor(lambda x: x + '_less')
    sf = StateFullProcessor(lambda x: x + '_full')
    sp = StringProcessor(lambda x: x + 'sp')

    # print sl.process('ind')
    # print sl.a
    # print isinstance(sl, Processor)
    # print issubclass(sl, Processor)

    print sf.process('ind')
#    print sf.a
    print isinstance(sf, Processor)
    print issubclass(sf, Processor)

    msp = MonoSpaceer()


