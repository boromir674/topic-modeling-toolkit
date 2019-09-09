from collections import OrderedDict
from configparser import ConfigParser

from topic_modeling_toolkit.processors.string_processors import MonoSpacer, StringProcessor, LowerCaser, UtfEncoder, DeAccenter, StringLemmatizer
from topic_modeling_toolkit.processors.generator_processors import GeneratorProcessor, MinLengthFilter, MaxLengthFilter, WordToNgramGenerator
from topic_modeling_toolkit.processors.string2generator import StringToTokenGenerator
from topic_modeling_toolkit.processors import Processor, InitializationNeededComponent, FinalizationNeededComponent, BaseDiskWriterWithPrologue
from topic_modeling_toolkit.processors.mutators import GensimDictTokenGeneratorToListProcessor, OneElemListOfListToGenerator

from .disk_writer_processors import UciFormatWriter, VowpalFormatWriter
from .processor import BaseDiskWriter


import logging
logger = logging.getLogger(__name__)

settings_value2processors = {
    'lowercase': lambda x: LowerCaser() if x else None,
    'monospace': lambda x: MonoSpacer() if x else None,
    'unicode': lambda x: UtfEncoder() if x else None,
    'deaccent': lambda x: DeAccenter() if x else None,
    'normalize': lambda x: StringLemmatizer() if x == 'lemmatize' else None,
    'minlength': lambda x: MinLengthFilter(x) if x else None,
    'maxlength': lambda x: MaxLengthFilter(x) if x else None,
    'ngrams': lambda x: WordToNgramGenerator(x) if x else None,
    'nobelow': lambda x: x if x else None,
    'noabove': lambda x: x if x else None,
    'weight': lambda x: x if x else None,
    'format': lambda x: {'uci': UciFormatWriter(), 'vowpal': VowpalFormatWriter()}[x] if x else None
}


class Pipeline(object):

    def __init__(self, settings):
        assert isinstance(settings, OrderedDict)
        self._parser = lambda x: ('format', x[1]) if x[0][:6] == 'format' else x
        self._settings = settings
        self.processors_names = [processor_name for processor_name, v in map(self._parser, settings.items()) if settings_value2processors[processor_name](v) is not None]
        self.processors = [settings_value2processors[processor_name](v) for processor_name, v in map(self._parser, settings.items()) if settings_value2processors[processor_name](v) is not None]
        assert len(self.processors_names) == len(self.processors)
        self.str2gen_processor_index = 0
        self.token_gen2list_index = 0
        if not self._check_processors_pipeline():
            print(self)
            raise ProcessorsOrderNotSoundException('The first n components of the pipeline have to be StringProcessors and the following m GeneratorProcessors with n,m>0')
        if any(isinstance(x, MonoSpacer) for x in self.processors):
            # self._insert(0, StringToTokenGenerator(' '), 'single-space-tokenizer')
            self.str2gen_processor = StringToTokenGenerator(' ')
        else:
            print(self)
            raise SupportedTokenizerNotFoundException('The implemented \'single-space\' tokenizer requires the presence of a MonoSpacer processor in the pipeline')
        self._inject_connectors()

    @property
    def settings(self):
        return self._settings

    def __len__(self):
        return len(self.processors)

    def __str__(self):
        b = '{}\nlen: {}\n'.format(type(self).__name__, len(self))
        b += '\n'.join('{}: {}'.format(pr_name, pr_instance) for pr_name, pr_instance in zip(self.processors_names, self.processors))
        return b

    def __getitem__(self, item):
        return self.processors_names[item], self.processors[item]

    def __iter__(self):
        for pr_name, pr in zip(self.processors_names, self.processors):
            yield pr_name, pr
    @property
    def transformers(self):
        return [(name, proc_obj) for name, proc_obj in self if isinstance(proc_obj, BaseDiskWriter)]

    @property
    def disk_writers(self):
        return [(name, proc_obj) for name, proc_obj in self if isinstance(proc_obj, BaseDiskWriter)]

    def initialize(self, *args, **kwargs):
        """Call this method to initialize each of the pipeline's processors"""
        if self.disk_writers and not 'file_paths' in kwargs:
            logger.error("You have to supply the 'file_paths' list as a key argument, with each element being the target file path one per BaseDiskWriter processor.")
            return
        disk_writer_index = 0
        for pr_name, pr_obj in self:
            if isinstance(pr_obj, InitializationNeededComponent):
                pr_obj.initialize(file_paths=kwargs.get('file_paths', []), disk_writer_index=disk_writer_index)
                if isinstance(pr_obj, BaseDiskWriter):
                    disk_writer_index += 1

    # def initilialize_processing_units(self):
    #     for pr_name, pr_obj in self:
    #         if isinstance(pr_obj, InitializationNeededComponent) and not isinstance(pr_obj, BaseDiskWriter):
    #             pr_obj.initialize()
    #
    # def initilialize_disk_writting_units(self, file_paths):
    #     i = -1
    #     for pr_name, pr_obj in self:
    #         if isinstance(pr_obj, BaseDiskWriter):
    #             i += 1
    #             pr_obj.initialize(file_name=file_paths[i])
    #
    # def initialize(self, *args, **kwargs):
    #     self.initilialize_processing_units()
    #     if 'format' in self.processors_names:
    #         self.initilialize_disk_writting_units(kwargs.get('file_paths', []))

    # def initialize(self, *args, **kwargs):
    #
    #     depth = len(self)
    #     if args:
    #         depth = args[0]
    #     file_names = kwargs.get('file_names', [])
    #     writer_index = 0
    #     for pr_name, pr_obj in self[:depth]:
    #         if isinstance(pr_obj, InitializationNeededComponent):
    #             if isinstance(pr_obj, BaseDiskWriter):
    #                 pr_obj.initialize(file_name=file_names[writer_index])
    #                 writer_index += 1
    #                 file_names = file_names[1:]
    #             else:
    #                 pr_obj.initialize()
    #     self._init_index = depth

    def pipe_through(self, data, depth):
        for proc in self.processors[:depth]:
            if isinstance(proc, Processor):
                data = proc.process(data)
        return data

    def pipe_through_processing_units(self, data):
        for proc in self.processors:
            if isinstance(proc, Processor) and not isinstance(proc, BaseDiskWriter):
                data = proc.process(data)
        return data

    def pipe_through_disk_writers(self, data, file_paths):
        for i, (name, proc) in enumerate(self.disk_writers):
            proc.initialize(file_paths, i)
        for name, proc in self.disk_writers:
            data = proc.process(data)

    def finalize(self, prologs=tuple([])):
        i = 0
        for pr_name, pr_obj in self:
            if isinstance(pr_obj, BaseDiskWriterWithPrologue):
                pr_obj.finalize(prologs[i])
                i += 1
            elif isinstance(pr_obj, FinalizationNeededComponent):
                pr_obj.finalize()

    def _insert(self, index, processor, type_name):
        self.processors.insert(index, processor)
        self.processors_names.insert(index, type_name)

    def _inject_connectors(self):
        assert (self.str2gen_processor_index != 0 and self.token_gen2list_index != 0)
        # STRING PROCESSORS (strings are passing through)
        self._insert(self.str2gen_processor_index, self.str2gen_processor, 'str2token_gen')
        # generators passing thorugh
        self._insert(self.token_gen2list_index + 1, GensimDictTokenGeneratorToListProcessor(), 'dict-builder')
        self._insert(self.token_gen2list_index + 2, OneElemListOfListToGenerator(), 'list2generator')

    def _check_processors_pipeline(self):
        i = 0
        proc = self[i][1]
        while isinstance(proc, StringProcessor):
            i += 1
            proc = self[i][1]
        if i == 0:
            return False
        l1 = i
        self.str2gen_processor_index = l1
        while isinstance(proc, GeneratorProcessor):
            i += 1
            proc = self[i][1]
        if i == l1:
            return False
        self.token_gen2list_index = i
        return True

    @staticmethod
    def _tuple2string(pipeline_component, value):
        if type(value) == bool:
            if value:
                return pipeline_component
            else:
                return ''
        elif pipeline_component == 'format':
            return value
        else:
            return '{}-{}'.format(pipeline_component, value)

    def get_id(self):
        return '_'.join(_ for _ in [self._tuple2string(pipeline_component, value) for pipeline_component, value in self._settings.items()] if _)

    @classmethod
    def from_cfg(cls, cfg_file_path):
        config = ConfigParser()
        config.read(cfg_file_path)
        pipe_settings = OrderedDict([item for sublist in map(lambda x: [(x[0], encode_pipeline_cfg[x[0]](x[1]))] if x[0] != 'format' else [(x[0] + str(i + 1), out_frmt) for i, out_frmt in enumerate(x[1].split(','))], config.items('preprocessing')) for item in sublist])
        # print 'Pipe-Config:\n' + ',\n'.join('{}: {}'.format(key, value) for key, value in pipe_settings.items())
        return Pipeline(pipe_settings)

    @classmethod
    def from_tuples(cls, data):
        return Pipeline(OrderedDict([item for sublist in
                                     map(lambda x: [(x[0], encode_pipeline_cfg[x[0]](x[1]))] if x[0] != 'format' else [(x[0] + str(i + 1)
                                                                                                                        , out_frmt)
                                                                                                                       for i, out_frmt in
                                                                                                                       enumerate(x[1].split(','))], data) for item in sublist]))

encode_pipeline_cfg = {
    'lowercase': lambda x: bool(eval(x)),
    'monospace': lambda x: bool(eval(x)),
    'unicode': lambda x: bool(eval(x)),
    'deaccent': lambda x: bool(eval(x)),
    'normalize': str,
    'minlength': int,
    'maxlength': int,
    'nobelow': int,
    'noabove': float,
    'ngrams': int,
    'weight': str,
    'format': str
}


class SupportedTokenizerNotFoundException(Exception): pass
class ProcessorsOrderNotSoundException(Exception): pass


