from gensim.corpora import Dictionary
from topic_modeling_toolkit.processors.processor import StateLessProcessor, PostUpdateSFProcessor, ElementCountingProcessor


class TokenGeneratorToList(StateLessProcessor):
    pass


class DefaultTokenGeneratorTolist(TokenGeneratorToList):
    def __init__(self):
        super(DefaultTokenGeneratorTolist, self).__init__(lambda x: [_ for _ in x])


class OneElemListOfListToGenerator(StateLessProcessor):
    def __init__(self):
        super(OneElemListOfListToGenerator, self).__init__(lambda x: (_ for _ in x[0]))


class ListToGenerator(StateLessProcessor):
    def __init__(self):
        super(ListToGenerator, self).__init__(lambda x: (_ for _ in x))


class ListWithCountingToGenerator(ElementCountingProcessor):
    def __init__(self):
        super(ListWithCountingToGenerator, self).__init__(lambda x: (_ for _ in x[0]))


class GensimDictTokenGeneratorToListProcessor(PostUpdateSFProcessor):
    def __init__(self):
        super(PostUpdateSFProcessor, self).__init__(lambda x: [[_ for _ in x]], Dictionary(), 'add_documents')


class StringToFieldsGenerator(StateLessProcessor):
    def __init__(self, category2files, fields, nb_docs='all'):
        self.failed = []
        self.category2files = category2files
        self.fields = tuple(fields)
        self.nb_processed = 0
        self.nb_docs = nb_docs
        super(StringToFieldsGenerator, self).__init__(lambda x: self.get_gen(x))

    def __str__(self):
        return '{}(docs:{}) -> [{}]'.format(type(self).__name__, self.nb_docs, ', '.join(self.fields))

    def get_gen(self, a_string):
        raise NotImplementedError
