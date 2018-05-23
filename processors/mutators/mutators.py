import pandas as pd
from gensim.corpora import Dictionary
from processors.processor import StateLessProcessor, PostUpdateSFProcessor, ElementCountingProcessor


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


class CategoryToFieldGenerator(StateLessProcessor):
    def __init__(self, category2files, extra_fields=(), sample_docs=None):
        self.failed = []
        self.category2files = category2files
        self.fields = tuple(['text'] + [_ for _ in extra_fields])
        self.nb_processed = 0
        if sample_docs:
            self.nb_docs = sample_docs
        else:
            self.nb_docs = 'all'
        super(CategoryToFieldGenerator, self).__init__(lambda x: self.get_gen(x))
        # super(StateLessProcessor, self).__init__(self.get_gen)

    def __str__(self):
        return type(self).__name__ + '(' + str(self.nb_docs) + ')'

    def get_gen(self, category):
        nb = self.nb_docs
        if self.nb_docs == 'all':
            nb = None
        for i, fields in enumerate(gen_fields(category, self.category2files, sample_docs=nb, fields=self.fields)):
            if type(fields['text']) not in (str, unicode):
                self.failed.append({'enum': i, 'type': type(fields), 'index': fields['index']})
            else:
                yield fields
                self.nb_processed += 1


def gen_fields(category, cat2files, sample_docs=None, fields=tuple('text')):
    total_docs = 0
    full_docs = 0
    for cat in category.split('+'):
        for fi, pickle_file in enumerate(cat2files[cat]):
            print '{}: Working with \'{}\' file'.format(fi, pickle_file)
            for index, df_entry in pd.read_pickle(pickle_file).iterrows():
                if sample_docs and total_docs >= sample_docs:
                    break
                try:
                    dd = dict({fld: df_entry[fld] for fld in fields}, **{'index': index})
                    # print dd
                    yield dd
                    # yield {fld: df_entry[fld] for fld in fields}.update({'index': index})
                    full_docs += 1
                except KeyError:
                    yield {'text': df_entry['text'], 'index': index}
                total_docs += 1
    print 'Yielded {} documents, with {} full'.format(total_docs, full_docs)


class CategoryToTextGenerator(CategoryToFieldGenerator):
    def __init__(self, category2files, sample_docs=None):
        super(CategoryToTextGenerator, self).__init__(category2files, sample_docs=sample_docs)

    def get_gen(self, category):
        return (out['text'] for out in super(CategoryToTextGenerator, self).get_gen(category))


class CategoryToTextNPosterIdGenerator(CategoryToFieldGenerator):
    def __init__(self, category2files, sample_docs=None):
        self.missing_poster_id = []
        super(CategoryToTextNPosterIdGenerator, self).__init__(category2files, extra_fields=tuple('poster_id'), sample_docs=sample_docs)

    def get_gen(self, category):
        for i, fields in enumerate(super(CategoryToTextNPosterIdGenerator, self).get_gen(category)):
            if 'poster_id' not in fields:
                self.missing_poster_id.append({'enum': i, 'type': type(fields), 'index': fields['index']})
            else:
                yield fields


class DictGenToStringGenProcessor(Sta)