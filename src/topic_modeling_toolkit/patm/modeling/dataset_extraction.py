import sys
import os
import pandas as pd

from topic_modeling_toolkit.patm.definitions import CATEGORY_2_FILES_HASH
from topic_modeling_toolkit.processors.mutators import StringToFieldsGenerator


class CategoryToFieldsGenerator(StringToFieldsGenerator):
    def __init__(self, fields, nb_docs='all'):
        super(CategoryToFieldsGenerator, self).__init__(CATEGORY_2_FILES_HASH, fields, nb_docs=nb_docs)

    def get_gen(self, category):
        text_filed_types = [str]
        if sys.version_info[0] < 3:  # python 3 does not have unicode type. So T.O.D.O: make sure logic does not break when removing this check
            text_filed_types.append(unicode)

        if self.nb_docs == 'all':
            sample = None
        else:
            sample = self.nb_docs
        for i, fields in enumerate(gen_fields(category, self.category2files, sample_docs=sample, fields=self.fields)):
            # if type(fields['text']) not in (str, unicode):  syntax correct for python 2 only
            if type(fields['text']) not in text_filed_types:  # if type(fields['text']) not in (str, unicode):  syntax correct for python 2 and 3
                self.failed.append({'enum': i, 'type': type(fields), 'index': fields['index']})
            else:
                yield fields
                self.nb_processed += 1


# def get_text_generator(nb_docs='all'):
#     return CategoryToFieldsGenerator('text', nb_docs=nb_docs)


# def get_posts_generator(nb_docs='all'):
#     return CategoryToFieldsGenerator(('text', 'poster_id'), nb_docs=nb_docs)


def gen_fields(category, cat2files, sample_docs=None, fields=tuple('text')):
    """
    Given a category (ie 'posts') generates dictionaries with fields defined by the user input. If a sample size is
    given then generates till reaching the ammount; generates till depletion otherwise.
    In case it fails to find all the requested fields it generates dictionaries solely having the 'text' field.
    Therefore it expects each document in the dataframes to have a 'text' field.\n
    :param str category: The category of pickled files to consider [posts, comments, posts+comments]. Currently supports only posts
    :param dict cat2files: mapping of categories [posts, comments] to their corresponding files: string -> list of strings mapping
    :param int sample_docs: Specifies the maximun number of items to generate. If not specified generates till depletion
    :param tuple fields: the fields of interest to include in the generated dictionaries
    """
    total_docs = 0
    full_docs = 0
    for cat in category.split('+'):
        for fi, pickle_file in enumerate(cat2files[cat]):
            print('{}: Working with \'{}\' file'.format(fi, pickle_file))
            for index, df_entry in pd.read_pickle(pickle_file).iterrows():
                if sample_docs and total_docs >= sample_docs:
                    break
                try:
                    dd = dict({fld: df_entry[fld] for fld in fields}, **{'index': index})
                    yield dd
                    full_docs += 1
                except KeyError:
                    yield {'text': df_entry['text'], 'index': index}
                total_docs += 1
    print('Yielded {} documents, with {} full'.format(total_docs, full_docs))
