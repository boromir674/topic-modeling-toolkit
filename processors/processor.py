from abc import ABCMeta, abstractmethod


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


class ElementCountingProcessor(StateLessProcessor):
    def __init__(self, func):
        self.nb_elems = 0
        super(StateLessProcessor, self).__init__(func)

    def process(self, data):
        self.nb_elems += len(data)
        return super(StateLessProcessor, self).process(data)


class StateFullProcessor(Processor):
    def __init__(self, func, state, callback):
        self.state = state
        self.callback = callback
        super(StateFullProcessor, self).__init__(func)

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


if __name__ == '__main__':

    s = Processor(lambda x: x + '_proc')
    # a = s.process('gav')
    # print a
    print isinstance(s, MetaProcessor)
    sl = StateLessProcessor(lambda x: x + '_less')
    # sf = StateFullProcessor(lambda x: x + '_full', [1, 2], 'append')
    # sf = PreUpdateSFProcessor(lambda x: x + '_full', [1, 2], 'append')
    #
    # print isinstance(sl, Processor)

    # print isinstance(sf, Processor)
    # print isinstance(sf, MetaProcessor)
    #
    # # print sl.process('gav')
    #
    # print sf.state
    # print sf.process('gav')
    # print sf.state
    #
    # sf.update(100)
    # print
    # print sf.state
    # print sf.process('gav')
    # print sf.state
