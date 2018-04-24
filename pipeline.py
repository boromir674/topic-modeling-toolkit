from collections import OrderedDict

from definitions import settings_value2processors
from processors import StringToTokenGenerator, MonoSpacer, StringProcessor, GeneratorProcessor


class Pipeline(object):

    def __init__(self, settings):
        assert isinstance(settings, OrderedDict)
        self.settings = settings
        self.processors_names = tuple([processor_name for processor_name, v in settings.items() if settings_value2processors[processor_name](v) is not None])
        self.processors = tuple([settings_value2processors[processor_name](v) for processor_name, v in settings.items() if settings_value2processors[processor_name](v) is not None])
        assert len(self.processors_names) == len(self.processors)
        if not check_processors_pipeline(self):
            print self
            raise ProcessorsOrderNotSoundException('the first n components of the pipeline have to be StringProcessors and the following m GeneratorProcessors wtih n,m>0')
        if any(isinstance(x, MonoSpacer) for x in self.processors):
            self.str2gen_processor = StringToTokenGenerator(' ')
        else:
            print self
            raise SupportedTokenizerNotFoundException('The single \'space\' \s implemented tokenizer requires the presence of a MonoSpacer processor in the pipeline')

    def __len__(self):
        return len(self.processors)

    def __str__(self):
        b = 'Pipeline\nlen: {}\n'.format(len(self))
        b += '\n'.join('{}: {}'.format(pr_name, pr_instance) for pr_name, pr_instance in zip(self.processors_names, self.processors))
        return b

    def __getitem__(self, item):
        return self.processors_names[item], self.processors[item]

    def __iter__(self):
        for pr_name, pr in zip(self.processors_names, self.processors):
            yield pr_name, pr

    def pipe_through(self, data):
        res = data
        i = 0
        el = self.processors[i]
        while isinstance(el, StringProcessor):
            res = el.process(res)
            i += 1
            el = self.processors[i]
        res = self.str2gen_processor.process(res)
        while isinstance(el, GeneratorProcessor):
            res = el.process(res)
            i += 1
            el = self.processors[i]
        # print 'Piped data from first {} processors'.format(i)
        return res

    def partial_pipe(self, data, start, end):
        res = data
        for pro in self.processors[start:end]:
            res = pro.process(res)
        return res


def check_processors_pipeline(pipe):
    assert isinstance(pipe, Pipeline)
    i = 0
    proc = pipe[i][1]
    while isinstance(proc, StringProcessor):
        i += 1
        proc = pipe[i][1]
    if i == 0:
        return False
    l1 = i
    while isinstance(proc, GeneratorProcessor):
        i += 1
        proc = pipe[i][1]
    if i == l1:
        return False
    return True


class DefaultPipeline(Pipeline):
    pass


class SupportedTokenizerNotFoundException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class ProcessorsOrderNotSoundException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)
