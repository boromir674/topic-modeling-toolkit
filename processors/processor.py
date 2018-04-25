import re
from abc import ABCMeta, abstractmethod
import pandas as pd
from gensim.corpora import Dictionary


class MetaProcessor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__mro__ = [Processor]

    @abstractmethod
    def process(self, data):
        raise NotImplementedError

    def __str__(self):
        return type(self).__name__

    @classmethod
    def __subclasshook__(a_class, C):
        if a_class is Processor:
            if any('process' in B.__dict__ for B in C.__mro__):
                # print ['process' in B.__dict__ for B in C.__mro__]
                return True
        return NotImplemented


class StateFullMetaProcessor(MetaProcessor):

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(a_class, C):
        if a_class is Processor:
            if any('update' in B.__dict__ for B in C.__mro__):
                # print ['process' in B.__dict__ for B in C.__mro__]
                return True
        return NotImplemented


class Processor(object):
    def __init__(self, func):
        self.func = func

    def process(self, data):
        return self.func(data)

    def to_id(self):
        return self.func.__name__

    def __str__(self):
        return type(self).__name__


MetaProcessor.register(Processor)


class StateLessProcessor(Processor):
    pass


class StateFullProcessor(Processor):
    def __init__(self, func, state, callback):
        self.state = state
        self.callback = callback

        super(Processor, self).__init__(func)

    def __str__(self):
        return type(self).__name__ + '(' + str(self.state) + ')'

    def update(self, data):
        self.state.__getattribute__(self.callback)(data)


StateFullMetaProcessor.register(StateFullProcessor)


class PostUpdateSFProcessor(StateFullProcessor):
    def process(self, data):
        _ = super(StateFullProcessor, self).process(data)
        self.update(_)
        return _


class PreUpdateSFProcessor(StateFullProcessor):
    def process(self, data):
        self.update(data)
        return super(StateFullProcessor, self).process(data)

### MUTATORS ###




class PrimitiveTypeStateStringProcessor(StateFullProcessor):
    def __init__(self, param):
        super(StateFullProcessor, self).__init__()


if __name__ == '__main__':

    sl = StateLessProcessor(lambda x: x + '_less')
    # sf = StateFullProcessor(lambda x: x + '_full')
    sp = StringProcessor(lambda x: x + 'sp')


    msp = MonoSpacer()

    print isinstance(sl, Processor)
    print isinstance(sl, MetaProcessor)
    print isinstance(sp, Processor)
    print isinstance(sp, MetaProcessor)

    gdp = GensimDictGeneratorToListProcessor()
    # print issubclass(msp, Processor)
    # print issubclass(StateLessProcessor, Processor)

    docs = [['alpha', 're', 'gav', 'gav'],
            ['asto', 'paramithi', 're']
            ]

    generator = (w for w in docs[0])
    bfg = WordToNgramGenerator(2)
    ng = bfg.process(generator)

    print [_ for _ in ng]
