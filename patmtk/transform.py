import os
import re
import sys
import argparse
from operator import itemgetter
from collections import OrderedDict
import pandas as pd
import ConfigParser
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from patm.modeling import get_posts_generator
from patm import Pipeline, TextDataset, get_pipeline
from patm.definitions import patm_root_dir, data_root_dir, encode_pipeline_cfg, cat2files, get_id, COLLECTIONS_DIR, poster_id2ideology_label, IDEOLOGY_CLASS_NAME, COOCURENCE_DICT_FILE_NAMES


class PipeHandler(object):
    def __init__(self, category, sample='all'):
        self.category = category
        self.sample = sample
        self.cat2textgen_proc = None
        self.text_generator = None
        self.doc_gen_stats = {}
        self.dct = None
        self.corpus = None
        self.nb_docs = 0
        self.pipeline = None
        self.dataset = None
        self._collection = ''
        self._col_dir = ''
        self.vocab_file = ''
        self.uci_file = ''
        self.vowpal_file = ''
        self.ideology_labels = []
        self._pack_data = None
        self._data_models = {}
        self._data_model2constructor = {'counts': lambda x: x,
                                       'tfidf': TfidfModel}
        self._vec_gen = {'counts': lambda bow_model: (_ for _ in bow_model),
                         'tfidf': lambda bow_model: (_ for _ in map(lambda x: self._data_models['tfidf'][x], bow_model))}
        self._format_data_tr = {
            'uci': lambda x: x[1],
            'vowpal': lambda x: [map(lambda y: (self.dct[y[0]], y[1]), x[1]), {IDEOLOGY_CLASS_NAME: self.ideology_labels[x[0]]}]
        }

    def create_pipeline(self, pipeline_cfg):
        pipe_settings = cfg2pipe_settings(pipeline_cfg, 'preprocessing')
        print 'Pipe-Config:\n' + ',\n'.join('{}: {}'.format(key, value) for key, value in pipe_settings.items())
        self.pipeline = get_pipeline(pipe_settings)
        return self.pipeline

    def set_doc_gen(self, category, num_docs='all'):
        self.sample = num_docs
        self.cat2textgen_proc = get_posts_generator(nb_docs=self.sample)
        self.text_generator = self.cat2textgen_proc.process(category)
        print self.cat2textgen_proc, '\n'

    # TODO refactor in OOP style preprocess
    def preprocess(self, a_pipe, collection):
        self._collection = collection
        self._col_dir = os.path.join(COLLECTIONS_DIR, collection)
        if not os.path.exists(self._col_dir):
            os.makedirs(self._col_dir)
            print 'Created \'{}\' as target directory for persisting'.format(self._col_dir)
        self.set_doc_gen(self.category, num_docs=self.sample)

        self.uci_file = os.path.join(COLLECTIONS_DIR, self._collection, 'docword.{}.txt'.format(self._collection))
        self.vowpal_file = os.path.join(COLLECTIONS_DIR, self._collection, 'vowpal.{}.txt'.format(self._collection))
        self.pipeline[-2][1].fname = self.uci_file
        self.pipeline[-1][1].fname = self.vowpal_file

        self.pipeline.initialize()

        # PASS DATA THROUGH PIPELINE
        doc_gens = []
        self.doc_gen_stats['corpus-tokens'] = 0
        for i, doc in enumerate(self.text_generator):
            doc_gens.append(a_pipe.pipe_through(doc['text'], len(a_pipe)-2))
            self.ideology_labels.append(poster_id2ideology_label[str(doc['poster_id'])])

        self.dct = a_pipe[a_pipe.processors_names.index('dict-builder')][1].state
        # self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        # print '{} tokens in all generators\n'.format(sum_toks)
        # print 'total bow tuples in corpus: {}'.format(sum(len(_) for _ in self.corpus))
        print '\nnum_pos', self.dct.num_pos, '\nnum_nnz', self.dct.num_nnz, '\n{} items in dictionary'.format(len(self.dct.items()))
        print '\n'.join(map(lambda x: '{}: {}'.format(x[0], x[1]), sorted(self.dct.iteritems(), key=itemgetter(0))[:5]))

        print 'filter extremes'
        self.dct.filter_extremes(no_below=a_pipe.settings['nobelow'], no_above=a_pipe.settings['noabove'])
        print '\nnum_pos', self.dct.num_pos, '\nnum_nnz', self.dct.num_nnz, '\n{} items in dictionary'.format(len(self.dct.items()))

        print 'compactify'
        self.dct.compactify()
        print '\nnum_pos', self.dct.num_pos, '\nnum_nnz', self.dct.num_nnz, '\n{} items in dictionary'.format(len(self.dct.items()))

        self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        print 'total bow tuples in corpus: {}\n'.format(sum(len(_) for _ in self.corpus))
        print 'corpus len (nb_docs):', len(self.corpus), 'empty docs', len([_ for _ in self.corpus if not _])

        # DO SOME MANUAL FILE WRITING
        self._write_vocab()
        self._write()
        pr_lines = map(lambda x: str(x), [self.dct.num_docs, len(self.dct.items()), sum(len(_) for _ in self.corpus)])
        # pr2 = ['a', 'b'] # self.pipeline.finalize([pr_lines, pr2])
        self.pipeline.finalize([pr_lines])
        self.dataset = TextDataset(self._collection, self.get_dataset_id(a_pipe),
                                   len(self.corpus), len(self.dct.items()), sum(len(_) for _ in self.corpus), self.uci_file, self.vocab_file, self.vowpal_file)
        self.dataset.save()
        return self.dataset

    def _write(self):
        for processor in self.pipeline.processors[-2:]: # iterate through the last two processors, which are the disk writers
            for i, vector in enumerate(self._get_iterable_data_model(self.pipeline.settings['weight'])):  # 'counts' only supported (future work: 'tfidf')
                processor.process(self._format_data_tr[processor.to_id()]((i, vector)))
                # self.doc_gen_stats['nb-bows'] += len(vec)
        self.doc_gen_stats.update({'docs-gen': self.cat2textgen_proc.nb_processed, 'docs-failed': len(self.cat2textgen_proc.failed)})

    def _get_iterable_data_model(self, data_model):
        if data_model not in self._data_models:
            self._data_models[data_model] = self._data_model2constructor[data_model](self.corpus)
        return self._vec_gen[data_model](self.corpus)

    def _write_vocab(self):
        # Define file and dump the vocabulary (list of unique tokens, one per line)
        self.vocab_file = os.path.join(COLLECTIONS_DIR, self._collection, 'vocab.{}.txt'.format(self._collection))
        if not os.path.isfile(self.vocab_file):
            with open(self.vocab_file, 'w') as f:
                for gram_id, gram_string in self.dct.iteritems():
                    try:
                        f.write('{}\n'.format(gram_string.encode('utf-8')))
                    except UnicodeEncodeError as e:
                        # f.write('\n'.join(map(lambda x: '{}'.format(str(x[1])), sorted([_ for _ in self.dct.iteritems()], key=itemgetter(0)))))
                        print 'FAILED', type(gram_string), gram_string
                        raise e
                print 'Created \'{}\' file'.format(self.vocab_file)
        else:
            print 'File \'{}\' already exists'.format(self.vocab_file)

    def write_cooc_information(self, window, min_tf, min_df): # 'cooc_tf_', 'cooc_df_', 'ppmi_tf_', 'ppmi_df_']
        """Assumes vocabulary and vowpal files have already been created"""
        d = {COOCURENCE_DICT_FILE_NAMES[0]: min_tf, COOCURENCE_DICT_FILE_NAMES[1]: min_df, COOCURENCE_DICT_FILE_NAMES[2]: min_tf, COOCURENCE_DICT_FILE_NAMES[3]: min_df}
        files = map(lambda x: os.path.join(self._col_dir, x+str(d[x])), COOCURENCE_DICT_FILE_NAMES)
        create_cooc_file(self.vowpal_file, self.vocab_file, window, files[0], files[1], files[2], files[3], min_tf=min_tf, min_df=min_df)

    def get_words_file_name(self, a_pipe):
        assert isinstance(a_pipe, Pipeline)
        idd = re.sub('_weight-[a-zA-Z]+', '', get_id(a_pipe.settings))
        return str(self.cat2textgen_proc.nb_processed) + '_' + idd.replace('_uci', '.words')

    def get_bow_file_name(self, a_pipe):
        assert isinstance(a_pipe, Pipeline)
        idd = get_id(a_pipe.settings)
        ri = idd.rfind('_')
        return str(self.cat2textgen_proc.nb_processed) + '_' + idd[:ri] + '.' + idd[ri + 1:]

    def get_dataset_id(self, a_pipe):
        assert isinstance(a_pipe, Pipeline)
        idd = get_id(a_pipe.settings)
        ri = idd.rfind('_')
        return str(self.cat2textgen_proc.nb_processed) + '_' + idd[:ri] + '.' + idd[ri + 1:]


def cfg2pipe_settings(config_path, section):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    return OrderedDict([item for sublist in map(lambda x: [(x[0], encode_pipeline_cfg[x[0]](x[1]))] if x[0] != 'format' else [(x[0]+str(i+1), out_frmt) for i, out_frmt in enumerate(x[1].split(','))], config.items(section)) for item in sublist]) # [(setting_name, encode_pipeline_cfg[setting_name](value)) for setting_name, value in config.items(section)])


def create_cooc_file(vowpal_file, vocab_file, cooc_window, tf_f, df_f, ppmi_tf, ppmi_df, min_tf=0, min_df=0):
    """

    :param str vowpal_file: path to vowpal-formated bag-of-words file
    :param str vocab_file: path to uci-formated (list of unique tokens) vocabulary file
    :param int cooc_window: number of tokens around specific token, which are used in calculation of cooccurrences
    :param int min_tf: minimal value of cooccurrences of a pair of tokens that are saved in dictionary of cooccurrences
    :param int min_df: minimal value of documents in which a specific pair of tokens occurred together closely
    :return:
    """
    # TF: save dictionary of co-occurrences with frequencies of co-occurrences of every specific pair of tokens in whole collection
    # DF: save dictionary of co-occurrences with number of documents in which every specific pair occured together
    # PPMI TF: save values of positive pmi of pairs of tokens from cooc_tf dictionary
    # PPMI DF: save values of positive pmi of pairs of tokens from cooc_df dictionary
    cmd = 'bigartm -c {} -v {} --cooc-window {} --cooc-min-tf {} --write-cooc-tf {} --cooc-min-df {} --write-cooc-df {}' \
          ' --write-ppmi-tf {} --write-ppmi-df {} --force'.format(vowpal_file, vocab_file, cooc_window, min_tf, tf_f, min_df, df_f, ppmi_tf, ppmi_df)
    os.system(cmd)

def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='transform.py', description='Transforms pickled panda.DataFrame\'s into Vowpal Wabbit format', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('category', help='the category of data to use')
    parser.add_argument('config', help='the .cfg file to use for constructing a pipeline')
    parser.add_argument('collection', help='a given name for the collection')
    parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()
    nb_docs = args.sample
    if nb_docs != 'all':
        nb_docs = int(nb_docs)
    ph = PipeHandler(args.category, sample=nb_docs)
    pipe = ph.create_pipeline(args.config)
    print '\n', pipe, '\n'
    uci_dt = ph.preprocess(pipe, args.collection)
    print uci_dt

    print 'Building coocurences informtion'
    ph.write_cooc_information(10, 5, 5)
