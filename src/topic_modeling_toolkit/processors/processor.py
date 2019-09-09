import os
from abc import ABCMeta, abstractmethod, abstractproperty
from functools import reduce


import logging
logger = logging.getLogger(__name__)


class MetaProcessor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def process(self, data):
        raise NotImplemented

    def __str__(self):
        return type(self).__name__


class StateFullMetaProcessor(MetaProcessor):
    __metaclass__ = ABCMeta

    @abstractproperty
    def state(self):
        raise NotImplemented

    @abstractmethod
    def update(self, data):
        raise NotImplemented


class InitializationNeededComponent(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplemented

class FinalizationNeededComponent(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def finalize(self):
        raise NotImplemented

class DiskWriterMetaProcessor(MetaProcessor, InitializationNeededComponent, FinalizationNeededComponent):
    __metaclass__ = ABCMeta


class Processor(MetaProcessor):
    def __init__(self, func):
        self.func = func

    def process(self, data):
        return self.func(data)

    def to_id(self):
        return self.func.__name__


class StateFullProcessor(Processor, StateFullMetaProcessor):
    def __init__(self, func, state, callback):
        self._state = state
        self.callback = callback
        super(StateFullProcessor, self).__init__(func)

    @property
    def state(self):
        return self._state

    def update(self, data):
        self._state.__getattribute__(self.callback)(data)

    def __str__(self):
        return type(self).__name__ + '(' + str(self.state) + ')'


class PostUpdateSFProcessor(StateFullProcessor):
    def process(self, data):
        _ = super(PostUpdateSFProcessor, self).process(data)
        self.update(_)
        return _

class PreUpdateSFProcessor(StateFullProcessor):
    def process(self, data):
        self.update(data)
        return super(PreUpdateSFProcessor, self).process(data)


class BaseDiskWriter(Processor, DiskWriterMetaProcessor):
    """Ingests one doc vector at a time"""
    def __init__(self, fname, func):
        self.doc_num = 1
        self.fname = fname
        self.file_handler = None
        super(BaseDiskWriter, self).__init__(func)

    def __str__(self):
        return type(self).__name__ + '(' + str(self.fname) + ')'

    def process(self, data):
        try:
            _ = super(BaseDiskWriter, self).process(data)
            self.doc_num += 1
            return _
        except UnicodeEncodeError as e:
            print('Document vector below, produced a UnicodeEncodeError:')
            print('count:', self.doc_num, data[0])
            return None


    def initialize(self, *args, **kwargs):
        self.fname = kwargs['file_paths'][kwargs['disk_writer_index']]
        mkdir_p(os.path.dirname(self.fname))
        self.file_handler = open(self.fname, 'w+')

    def finalize(self):
        self.file_handler.close()


class BaseDiskWriterWithPrologue(BaseDiskWriter):
    def __init__(self, fname, func):
        super(BaseDiskWriterWithPrologue, self).__init__(fname, func)

    def finalize(self, *args):
        self.file_handler.seek(0)
        _ = self.file_handler.read()
        self.file_handler.seek(0, 0)
        self.file_handler.write(reduce(lambda x,y: x+y, map(lambda x: x + '\n', args[0])).rstrip('\r\n') + '\n' + _)
        super(BaseDiskWriterWithPrologue, self).finalize()


class StateLessProcessor(Processor):
    pass


class ElementCountingProcessor(StateLessProcessor):
    def __init__(self, func):
        self.nb_elems = 0
        super(StateLessProcessor, self).__init__(func)

    def process(self, data):
        self.nb_elems += len(data)
        return super(StateLessProcessor, self).process(data)



# Python > 2.5
import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
        logger.info("Created '{}' directory".format(path))
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

## Python > 3.2
# os.makedirs(path, exist_ok=True)

if __name__ == '__main__':

    s = Processor(lambda x: x + '_proc')
    # a = s.process('_builder')
    # print a
    print(isinstance(s, MetaProcessor))
    sl = StateLessProcessor(lambda x: x + '_less')
    # sf = StateUpdatableProcessor(lambda x: x + '_full', [1, 2], 'append')
    # sf = PreUpdateSFProcessor(lambda x: x + '_full', [1, 2], 'append')

