from collections import OrderedDict

from definitions import settings_value2processors


class Pipeline(object):

    def __init__(self, processors):
        self.processors = processors

    def pipe_through(self, data):
        res = data
        for pro in self.processors:
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


def create_pipeline(settings):
    assert isinstance(settings, OrderedDict)
    processors = [settings_value2processors[k](v) for k, v in settings.items()]

    return DefaultPipeline([pr for pr in processors if pr is not None])
