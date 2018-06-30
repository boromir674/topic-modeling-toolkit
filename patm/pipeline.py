from collections import OrderedDict

from definitions import settings_value2processors
from processors.string_processors import MonoSpacer, StringProcessor
from processors.generator_processors import GeneratorProcessor
from processors.string2generator import StringToTokenGenerator
from processors import Processor, InitializationNeededComponent, FinalizationNeededComponent, BaseDiskWriterWithPrologue
from processors.mutators import GensimDictTokenGeneratorToListProcessor, OneElemListOfListToGenerator


class Pipeline(object):

    def __init__(self, settings):
        assert isinstance(settings, OrderedDict)
        self._parser = lambda x: ('format', x[1]) if x[0][:6] == 'format' else x
        self.settings = settings
        self.processors_names = [processor_name for processor_name, v in map(self._parser, settings.items()) if settings_value2processors[processor_name](v) is not None]
        self.processors = [settings_value2processors[processor_name](v) for processor_name, v in map(self._parser, settings.items()) if settings_value2processors[processor_name](v) is not None]
        assert len(self.processors_names) == len(self.processors)
        self.str2gen_processor_index = 0
        self.token_gen2list_index = 0
        if not self._check_processors_pipeline():
            print self
            raise ProcessorsOrderNotSoundException('the first n components of the pipeline have to be StringProcessors and the following m GeneratorProcessors with n,m>0')
        if any(isinstance(x, MonoSpacer) for x in self.processors):
            # self._insert(0, StringToTokenGenerator(' '), 'single-space-tokenizer')
            self.str2gen_processor = StringToTokenGenerator(' ')
        else:
            print self
            raise SupportedTokenizerNotFoundException('The implemented \'single-space\' tokenizer requires the presence of a MonoSpacer processor in the pipeline')
        self._inject_connectors()

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

    def initialize(self):
        for pr_name, pr_obj in self:
            if isinstance(pr_obj, InitializationNeededComponent):
                pr_obj.initialize()

    def pipe_through(self, data, depth):
        for proc in self.processors[:depth]:
            if isinstance(proc, Processor):
                data = proc.process(data)
        return data

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
        self._insert(self.str2gen_processor_index, self.str2gen_processor, 'str2token_gen')
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


def get_pipeline(settings):
    # if pipe_type == 'uci': return UciOutputPipeline(settings)
    # if pipe_type == 'vowpal': return VowpalOutputPipeline(settings)
    return Pipeline(settings)


class SupportedTokenizerNotFoundException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class ProcessorsOrderNotSoundException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


        # class UciOutputPipeline(Pipeline):
        #     def _inject_connectors(self):
        #         super(UciOutputPipeline, self)._inject_connectors()
        #
        #
        #
        # class VowpalOutputPipeline(Pipeline):
        #     def _inject_connectors(self):
        #         super(VowpalOutputPipeline, self)._inject_connectors()
        #         self._insert(self.token_gen2list_index + 1, GensimDictTokenGeneratorToListProcessor(), 'dict-builder')
        #         self._insert(self.token_gen2list_index + 2, OneElemListOfListToGenerator(), 'list2generator')
        #