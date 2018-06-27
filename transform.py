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
from patm import Pipeline, get_dataset, get_pipeline
from patm.definitions import root_dir, data_root_dir, encode_pipeline_cfg, cat2files, get_id, collections_dir, poster_id2ideology_label, IDEOLOGY_LABEL_NAME


class PipeHandler(object):
    def __init__(self, category, sample='all'):
        self.category = category
        self.sample = sample
        self.cat2textgen_proc = None
        self.text_generator = None
        self.doc_gen_stats = {}
        self.dct = None
        self.model = None
        self.corpus = None
        self.nb_docs = 0
        self.pipeline = None
        self.dataset = None
        self._collection = ''
        self.vocab_file = ''
        self.ideology_labels = []
        self._pack_data = None

    def create_pipeline(self, pipeline_cfg):
        pipe_settings = cfg2pipe_settings(os.path.join(root_dir, 'code', pipeline_cfg), 'preprocessing')
        print 'Pipe-Config:\n' + ',\n'.join('{}: {}'.format(key, value) for key, value in pipe_settings.items())
        self.pipeline = get_pipeline(pipe_settings['format'], pipe_settings)
        if self.pipeline.settings['format'] == 'uci':
            self._pack_data = lambda x: x[1]
        elif self.pipeline.settings['format'] == 'vowpal':
            self._pack_data = lambda x: [map(lambda y: (self.dct[y[0]], y[1]), x[1]), {IDEOLOGY_LABEL_NAME: self.ideology_labels[x[0]]}]
        return self.pipeline

    def set_doc_gen(self, category, num_docs='all', labels=False):
        self.sample = num_docs
        self.cat2textgen_proc = get_posts_generator(nb_docs=self.sample)
        self.text_generator = self.cat2textgen_proc.process(category)
        print self.cat2textgen_proc, '\n'

    # TODO refactor in OOP style preprocess
    def preprocess(self, a_pipe, collection, labels=False):
        self._collection = collection
        self.set_doc_gen(self.category, num_docs=self.sample, labels=labels)
        target_folder = os.path.join(collections_dir, collection)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print 'Created \'{}\' as target directory for persisting'.format(target_folder)
        self.doc_gen_stats['corpus-tokens'] = 0
        doc_gens = []
        for i, doc in enumerate(self.text_generator):
            doc_gens.append(a_pipe.pipe_through(doc['text']))
            self.ideology_labels.append(poster_id2ideology_label[str(doc['poster_id'])])

        self.dct = a_pipe[a_pipe.processors_names.index('dict-builder')][1].state
        # self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        # print '{} tokens in all generators\n'.format(sum_toks)
        # print 'total bow tuples in corpus: {}'.format(sum(len(_) for _ in self.corpus))
        print '\nnum_pos', self.dct.num_pos
        print 'num_nnz', self.dct.num_nnz
        print '{} items in dictionary'.format(len(self.dct.items()))
        print '\n'.join(map(lambda x: '{}: {}'.format(x[0], x[1]), sorted(self.dct.iteritems(), key=itemgetter(0))[:5]))

        print 'filter extremes'
        self.dct.filter_extremes(no_below=a_pipe.settings['nobelow'], no_above=a_pipe.settings['noabove'])
        print 'num_pos', self.dct.num_pos
        print 'num_nnz', self.dct.num_nnz
        print '{} items in dictionary'.format(len(self.dct.items()))

        print 'compactify'
        self.dct.compactify()
        print 'num_pos', self.dct.num_pos
        print 'num_nnz', self.dct.num_nnz
        print '{} items in dictionary'.format(len(self.dct.items()))

        self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]

        print 'total bow tuples in corpus: {}\n'.format(sum(len(_) for _ in self.corpus))
        print 'corpus len (nb_docs):', len(self.corpus), 'empty docs', len([_ for _ in self.corpus if not _])

        self._write_vocab()

        if self.pipeline.settings['format'] == 'uci':
            files = [os.path.join(collections_dir, collection, 'docword.{}.txt'.format(collection)), self.vocab_file]
            self.pipeline[-1][1].fname = files[0]
        elif self.pipeline.settings['format'] == 'vowpal':
            files = [os.path.join(collections_dir, self._collection, 'vowpal.{}.txt'.format(self._collection))]
            self.pipeline[-1][1].fname = files[0]
        self._write()
        self.dataset = get_dataset(self.pipeline.settings['format'], self._collection, self.get_dataset_id(a_pipe), len(self.corpus), len(self.dct.items()), sum(len(_) for _ in self.corpus), files)
        self.dataset.save()
        return self.dataset

    def _write(self):
        if self.pipeline.settings['format'] == 'uci':
            with open(self.pipeline.processors[-1].fname, 'w') as f:
                # Write the first 3 lines of the uci-formated file that contain stats: nb_docs\n vocab_size\n nb_bow_tuples\n
                f.writelines('{}\n{}\n{}\n'.format(self.dct.num_docs, len(self.dct.items()), sum(len(_) for _ in self.corpus)))
        if self.pipeline.settings['weight'] == 'counts':
            gene = (_ for _ in self.corpus)
        elif self.pipeline.settings['weight'] == 'tfidf':
            self.model = TfidfModel(self.corpus)
            gene = (_ for _ in map(lambda x: self.model[x], self.corpus))
        for i, vec in enumerate(gene):
            self.doc_gen_stats['corpus-tokens'] += len(vec)
            self.pipeline[-1][1].process(self._pack_data((i, vec)))

        self.doc_gen_stats.update({'docs-gen': self.cat2textgen_proc.nb_processed, 'docs-failed': len(self.cat2textgen_proc.failed)})

    def _write_vocab(self):
        # Define file and dump the vocabulary (list of unique tokens, one per line)
        self.vocab_file = os.path.join(collections_dir, self._collection, 'vocab.{}.txt'.format(self._collection))
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
          ' --write-ppmi-tf {} --write-ppmi-df {}'.format(vowpal_file, vocab_file, cooc_window, min_tf, tf_f, min_df, df_f, ppmi_tf, ppmi_df)
    os.system(cmd)
    # --write-cooc-tf cooc_tf_ --cooc-min-df 200 --write-cooc-df cooc_df_ --write-ppmi-tf ppmi_tf_ --write-ppmi-df ppmi_df_

def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='transform.py', description='Transforms pickled panda.DataFrame\'s into Vowpal Wabbit format', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('category', help='the category of data to use')
    parser.add_argument('config', help='the .cfg file to use for constructing a pipeline')
    parser.add_argument('collection', help='a given name for the collection')
    parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    parser.add_argument('--labels', action='store_true', help='whether or not to load outlet labels per document if present. Defaults to not use labels')
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
    uci_dt = ph.preprocess(pipe, args.collection, labels=args.labels)
    print uci_dt

    vocab_created = 'vowpal.{}.txt'.format(args.collection)
    if os.path.isfile(vocab_created):
        print 'Building coocurences information'
        create_cooc_file(vocab_created, ph.vocab_file, 10, 'cooc_tf_', 'cooc_df_', 'ppmi_tf_', 'ppmi_df_', min_tf=200, min_df=200)
