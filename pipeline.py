from collections import OrderedDict

from definitions import settings_value2processors


class Pipeline(object):

    def __init__(self, settings):
        assert isinstance(settings, OrderedDict)
        self.settings = settings
        self.processors_names = tuple([processor_name for processor_name, v in settings.items() if settings_value2processors[processor_name](v) is not None])
        self.processors = tuple([settings_value2processors[processor_name](v) for processor_name, v in settings.items() if settings_value2processors[processor_name](v) is not None])
        assert len(self.processors_names) == len(self.processors)

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
        for pro in self.processors:
            res = pro.process(res)
        return res

    def partial_pipe(self, data, start, end):
        res = data
        for pro in self.processors[start:end]:
            res = pro.process(res)
        return res


class DefaultPipeline(Pipeline):
    pass
    # def __init__(self, settings):
    #     self.text_processors = []
    #     self.filters = []
    #     self.tokenizer = None

    # def pipe(self, doc):
    #     pass

#
# def create_pipeline(settings):
#
#
#     processors = [settings_value2processors[k](v) for k, v in settings.items()]
#     processors = [pr for pr in processors if pr is not None]
#
#     return DefaultPipeline(processors)
