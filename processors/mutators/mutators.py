import pandas as pd
from gensim.corpora import Dictionary
from processors.processor import StateLessProcessor, PostUpdateSFProcessor


class TokenGeneratorToList(StateLessProcessor):
    pass


class DefaultTokenGeneratorTolist(TokenGeneratorToList):
    def __init__(self):
        super(TokenGeneratorToList, self).__init__(lambda x: [_ for _ in x])


class ListToGenerator(StateLessProcessor):
    def __init__(self):
        super(StateLessProcessor, self).__init__(lambda x: (_ for _ in x[0]))


class GensimDictTokenGeneratorToListProcessor(PostUpdateSFProcessor):
    def __init__(self):
        super(PostUpdateSFProcessor, self).__init__(lambda x: [[_ for _ in x]], Dictionary(), 'add_documents')


class CategoryToTextGenerator(StateLessProcessor):
    def __init__(self, category2files, num_docs=None):
        self.failed = []
        self.category2files = category2files
        self.nb_processed = 0
        if num_docs:
            self.nb_docs = num_docs
        else:
            self.nb_docs = 'all'
        super(StateLessProcessor, self).__init__(lambda x: (_ for _ in self.gen_text(x)))

    def __str__(self):
        return super(StringToGenerator, self).__str__() + '(' + str(self.nb_docs) + ')'

    def gen_text(self, category):
        nb = self.nb_docs
        if self.nb_docs == 'all':
            nb = None
        gen = category2generator(category, self.category2files, num_docs=nb)
        for i, text in enumerate(gen):
            if type(text) not in (str, unicode):
                self.failed.append((i, type(text)))
            else:
                yield text
                self.nb_processed += 1


def category2generator(category, cat2files, num_docs=None):
    def gen_text():
        total_docs = 0
        for cat in category.split('+'):
            for fi, pickle_file in enumerate(cat2files[cat]):
                print '{}: Working with \'{}\' file'.format(fi, pickle_file)
                for (_, df_entry) in pd.read_pickle(pickle_file).iterrows():
                    if num_docs and total_docs >= num_docs:
                        break
                    yield df_entry['text']
                    total_docs += 1
        print 'Yielded {} documents'.format(total_docs)
    return (_ for _ in gen_text())
