import os
import re
import sys
import argparse
from operator import itemgetter
from collections import OrderedDict
import pandas as pd
from configparser import ConfigParser
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from .modeling.dataset_extraction import CategoryToFieldsGenerator
from .dataset import TextDataset
from topic_modeling_toolkit.processors import Pipeline

from .definitions import IDEOLOGY_CLASS_NAME, COOCURENCE_DICT_FILE_NAMES# = ['cooc_tf_', 'cooc_df_', 'ppmi_tf_', 'ppmi_df_']

# import logging
# logger = logging.getLogger(__name__)


class PipeHandler(object):
    def __init__(self):
        self.cat2textgen_proc = None
        self.text_generator = None
        self.doc_gen_stats = {}
        self.dct = None
        self.corpus = None
        self.nb_docs = 0
        self._pipeline = None
        self.dataset = None
        self._collection = ''
        self.vocab_file = ''
        self.uci_file = ''
        self.vowpal_file = ''
        self.outlet_ids = []
        self._pack_data = None
        self._data_models = {}
        self._data_model2constructor = {'counts': lambda x: x,
                                       'tfidf': TfidfModel}
        self._vec_gen = {'counts': lambda bow_model: (_ for _ in bow_model),
                         'tfidf': lambda bow_model: (_ for _ in map(lambda x: self._data_models['tfidf'][x], bow_model))}
        self._format_data_tr = {
            'uci': lambda x: x[1],
            'vowpal': lambda x: [map(lambda y: (self.dct[y[0]], y[1]), x[1]), {IDEOLOGY_CLASS_NAME: self.label(self.outlet_ids[x[0]])}]
        }
        self._labels_hash = {}

    @property
    def labels_hash(self):
        return self._labels_hash

    @labels_hash.setter
    def labels_hash(self, outlet_id2document_label_hash):
        self._labels_hash = outlet_id2document_label_hash

    def label(self, outlet_id):
        return self._labels_hash[outlet_id]

    @property
    def labels(self):
        return [self.label(x) for x in self.outlet_ids]

    @property
    def pipeline(self):
        return self._pipeline
    @pipeline.setter
    def pipeline(self, pipeline):
        """Set the processing pipeline for the handler to use.\n
        :param str or processors.pipeline.Pipeline pipeline:
        """
        if type(pipeline) == str:
            self._pipeline = Pipeline.from_cfg(pipeline)
        else:
            self._pipeline = pipeline

    def process(self, pipeline, category, sample='all', verbose=False):
        self.pipeline = pipeline
        if verbose:
            print(self._pipeline)
        self.pipe_through_processors(category, num_docs=sample)

    def persist(self, dataset_path, labels_hash, class_names, add_class_labels_to_vocab=True):
        self._prepare_storing(dataset_path)
        self._labels_hash = labels_hash
        self.pipe_through_disk_writers()
        self.class_names = class_names
        self.write_vocab(dataset_path, add_class_labels=add_class_labels_to_vocab)
        return self.create_dataset(dataset_path)

    def preprocess(self, category, pipeline, collection_path, labels_hash, class_names, sample='all', add_class_labels_to_vocab=True):
        self.process(pipeline, category, sample=sample)
        return self.persist(collection_path, labels_hash, class_names, add_class_labels_to_vocab=add_class_labels_to_vocab)

    def _prepare_storing(self, dataset_path):
        self._collection = os.path.basename(dataset_path)
        self.uci_file = os.path.join(dataset_path, 'docword.{}.txt'.format(self._collection))
        self.vowpal_file = os.path.join(dataset_path, 'vowpal.{}.txt'.format(self._collection))
        self.pipeline.initialize(file_paths=[self.uci_file, self.vowpal_file])

    #####
    def pipe_through_processors(self, category, num_docs='all'):
        doc_gens = []
        self.outlet_ids = []
        self.doc_gen_stats['corpus-tokens'] = 0
        self.cat2textgen_proc = CategoryToFieldsGenerator(('text', 'poster_id'), nb_docs=num_docs)
        self.text_generator = self.cat2textgen_proc.process(category)
        print(self.cat2textgen_proc, '\n')
        for i, doc in enumerate(self.text_generator):
            doc_gens.append(self._pipeline.pipe_through(doc['text'], len(self._pipeline) - 2))
            self.outlet_ids.append(str(doc['poster_id']))  # index outlets (document authors) ids

        self.dct = self._pipeline[self._pipeline.processors_names.index('dict-builder')][1].state
        # self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        # print '{} tokens in all generators\n'.format(sum_toks)
        # print 'total bow tuples in corpus: {}'.format(sum(len(_) for _ in self.corpus))
        # print "GENSIM-DICT:\nnum_pos (processes words): {}\nnum_nnz (nb of bow-tuples) {}\nvocab size: {}".format(self.dct.num_pos, self.dct.num_nnz, len(self.dct.items()))
        # print '\nnum_pos', self.dct.num_pos, '\nnum_nnz', self.dct.num_nnz, '\n{} items in dictionary'.format(len(self.dct.items()))
        print
        self._print_dict_stats()
        print("SAMPLE LEXICAL ITEMS:\n{}".format(
            '\n'.join(map(lambda x: '{}: {}'.format(x[0], x[1]), sorted(self.dct.iteritems(), key=itemgetter(0))[:5]))))

        tokens = [[token for token in tok_gen] for tok_gen in doc_gens]

        # print corpus stats before applying 'below' and 'above' filtering
        c = [self.dct.doc2bow(doc_tokens) for doc_tokens in tokens]
        self._print_bow_model_stats(c)

        print(' -- filter extremes -- ')
        self.dct.filter_extremes(no_below=self._pipeline.settings['nobelow'],
                                 no_above=self._pipeline.settings['noabove'])
        self._print_dict_stats()

        print(' -- compactify -- ')
        self.dct.compactify()
        self._print_dict_stats()

        # self.corpus = filter(None, [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens])
        self.corpus = [self.dct.doc2bow(doc_tokens) for doc_tokens in tokens]
        self._print_bow_model_stats(self.corpus)

        # REMOVE EMPTY DOCS
        c = [_ for _ in self.corpus if _]
        self.corpus, self.outlet_ids = (list(x) for x in zip(*[[doc, label] for doc, label in zip(self.corpus, self.outlet_ids) if doc]))
        assert len(c) == len(self.corpus) == len(self.outlet_ids)
        self._print_bow_model_stats(self.corpus)
        print

    def pipe_through_disk_writers(self):
        """Call to pass through the last BaseDiskWriter processors of the pieline. Assumes the last non BaseDsikWriter processor in the pipeline is a 'weight' so that a 'counts 'or 'tfidf' token weight model is computed"""
        if len(self.corpus) != len(self.outlet_ids):
            logger.warning("Please fix the logic because there is a missmatch between documents and labels: {} != {}".format(len(self.corpus), len(self.outlet_ids)))
        for _, processor in self.pipeline.disk_writers:
            for i, vector in enumerate(self._get_iterable_data_model(self.pipeline.settings['weight'])):  # 'counts' only supported (future work: 'tfidf')
                processor.process(self._format_data_tr[processor.to_id()]((i, vector)))
        self.doc_gen_stats.update({'docs-gen': self.cat2textgen_proc.nb_processed, 'docs-failed': len(self.cat2textgen_proc.failed)})

        # the first 3 lines of a uci formatted file: correspond to nb_docs, vocab_size, sum of nb of tuples (representing the bow model) found in all documents.
        # They should be written on the top
        prologue_lines = map(lambda x: str(x), [self.dct.num_docs, len(self.dct.items()), sum(len(_) for _ in self.corpus)])
        self.pipeline.finalize([prologue_lines])

    def _get_iterable_data_model(self, data_model):
        if data_model not in self._data_models:
            self._data_models[data_model] = self._data_model2constructor[data_model](self.corpus)
        return self._vec_gen[data_model](self.corpus)

    #######
    def write_vocab(self, dataset_path, add_class_labels=True):
        # Define file and dump the vocabulary (list of unique tokens, one per line)
        self.vocab_file = os.path.join(dataset_path, 'vocab.{}.txt'.format(os.path.basename(dataset_path)))
        if not os.path.isfile(self.vocab_file):
            with open(self.vocab_file, 'w') as f:
                for string_id, string in self._vocab_tokens_generator(include_class_labels=add_class_labels):
                    try:
                        # f.write('{}\n'.format(string.encode('utf-8')))
                        f.write('{}\n'.format(string))
                    except UnicodeEncodeError as e:
                        # f.write('\n'.join(map(lambda x: '{}'.format(str(x[1])), sorted([_ for _ in self.dct.iteritems()], key=itemgetter(0)))))
                        print('FAILED', type(string_id), string)
                        raise e
                print("Created '{}' file".format(self.vocab_file))
        else:
            print("File '{}' already exists. Skipping.".format(self.vocab_file))

    def _vocab_tokens_generator(self, include_class_labels=True):
        for gram_id, gram_string in self.dct.iteritems():
            yield gram_id, gram_string
        if include_class_labels:
            for class_label in [_ for _ in self.class_names if _ in set(self.labels)]:
                yield 'class_modality', '{} {}'.format(class_label, IDEOLOGY_CLASS_NAME)

    #######
    def create_dataset(self, dataset_path):
        dataset = TextDataset(os.path.basename(dataset_path), self._get_dataset_id(),
                                   len(self.corpus), len(self.dct.items()), sum(len(_) for _ in self.corpus),
                                   self.uci_file, self.vocab_file, self.vowpal_file)
        dataset.root_dir = dataset_path
        dataset.save()
        return dataset

    def _get_dataset_id(self):
        idd = self._pipeline.get_id()  # get_id(self._pipeline.settings)
        ri = idd.rfind('_')
        return str(len(self.corpus)) + '_' + idd[:ri] + '.' + idd[ri + 1:]

    ###### UTILS
    def _print_dict_stats(self):
        print("GENSIM-DICT:\nnum_pos (processes words): {}\nnum_nnz (nb of bow-tuples) {}\nvocab size: {}".format(
            self.dct.num_pos, self.dct.num_nnz, len(self.dct.items())))

    @classmethod
    def _print_bow_model_stats(cls, bow_corpus):
        print("BOW-MODEL:\nnumber of word position (num_pos): {}\ntotal number of tuples (num_nnz): {}\n number of docs: {}\nempty docs: {}".format(
            sum(sum(bow_tuple[1] for bow_tuple in doc) for doc in bow_corpus), sum(len(_) for _ in bow_corpus), len(bow_corpus), len([_ for _ in bow_corpus if not _])))
