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
from patm import Pipeline, UciDataset, get_pipeline
from patm.definitions import root_dir, data_root_dir, encode_pipeline_cfg, cat2files, get_id, collections_dir


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

    def create_pipeline(self, pipeline_cfg):
        pipe_settings = cfg2pipe_settings(os.path.join(root_dir, 'code', pipeline_cfg), 'preprocessing')
        print pipe_settings
        return get_pipeline(pipe_settings['format'], pipe_settings)

    def set_doc_gen(self, category, num_docs='all', labels=False):
        self.cat2textgen_proc = get_posts_generator(nb_docs=num_docs)
        self.text_generator = self.cat2textgen_proc.process(category)

    def preprocess(self, a_pipe, collection, labels=False):
        self.set_doc_gen(self.category, num_docs=self.sample, labels=labels)
        target_folder = os.path.join(collections_dir, collection)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print 'Created \'{}\' as target directory for persisting'.format(target_folder)
        self.doc_gen_stats['corpus-tokens'] = 0
        doc_gens = []
        for i, doc in enumerate(self.text_generator):
            doc_gens.append(a_pipe.pipe_through(doc['text']))

        self.dct = a_pipe[a_pipe.processors_names.index('dict-builder')][1].state
        # self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        # print '{} tokens in all generators\n'.format(sum_toks)
        # print 'total bow tuples in corpus: {}'.format(sum(len(_) for _ in self.corpus))
        print 'num_pos', self.dct.num_pos
        print 'nnz', self.dct.num_nnz
        print '{} items in dictionary'.format(len(self.dct.items()))
        print '\n'.join(map(lambda x: '{}: {}'.format(x[0], x[1]), sorted([_ for _ in self.dct.iteritems()], key=itemgetter(0))[:10]))

        print 'filter extremes'
        self.dct.filter_extremes(no_below=a_pipe.settings['nobelow'], no_above=a_pipe.settings['noabove'])
        print '{} items in dictionary'.format(len(self.dct.items()))
        # print '\n'.join(map(lambda x: '{}: {}'.format(x[0], x[1]), sorted([_ for _ in self.dct.iteritems()], key=itemgetter(0))[:10]))
        # print '\n'.join(['{}: {}'.format(k, v) for k, v in self.dct.iteritems()][:10])

        print 'compactify'
        self.dct.compactify()
        print '{} items in dictionary'.format(len(self.dct.items()))
        # print '\n'.join(map(lambda x: '{}: {}'.format(x[0], x[1]), sorted([_ for _ in self.dct.iteritems()], key=itemgetter(0))[:10]))
        # print '\n'.join(['{}: {}'.format(k, v) for k, v in self.dct.iteritems()][:10]

        self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        # for i, j in enumerate(self.corpus):
        #     print i, len(j)
        print 'total bow tuples in corpus: {}\n'.format(sum(len(_) for _ in self.corpus))
        print 'corpus len (nb_docs):', len(self.corpus), 'empty docs', len([_ for _ in self.corpus if not _])

        docword_file = os.path.join(collections_dir, collection, 'docword.{}.txt'.format(collection))
        a_pipe[-1][1].fname = docword_file

        with open(a_pipe.processors[-1].fname, 'w') as f:
            f.writelines('{}\n{}\n{}\n'.format(self.dct.num_docs, len(self.dct.items()), sum(len(_) for _ in self.corpus)))

        if a_pipe.settings['weight'] == 'counts':
            for vec in self.corpus:
                self.doc_gen_stats['corpus-tokens'] += len(vec)
                a_pipe[-1][1].process(vec)
        elif a_pipe.settings['weight'] == 'tfidf':
            self.model = TfidfModel(self.corpus)
            for vec in map(lambda x: self.model[x], self.corpus):
                self.doc_gen_stats['corpus-tokens'] += len(vec)
                a_pipe[-1][1].process(vec)

        self.doc_gen_stats.update({'docs-gen': self.cat2textgen_proc.nb_processed, 'docs-failed': len(self.cat2textgen_proc.failed)})

        vocab_file = os.path.join(collections_dir, collection, 'vocab.{}.txt'.format(collection))
        if not os.path.isfile(vocab_file):
            with open(vocab_file, 'w') as f:
                for gram_id, gram_string in self.dct.iteritems():
                    try:
                        f.write('{}\n'.format(gram_string.encode('utf-8')))
                    except UnicodeEncodeError as e:
                        # f.write('\n'.join(map(lambda x: '{}'.format(str(x[1])), sorted([_ for _ in self.dct.iteritems()], key=itemgetter(0)))))
                        print 'FAILED', type(gram_string), gram_string
                        sys.exit(1)
                print 'Created \'{}\' file'.format(vocab_file)
        else:
            print 'File \'{}\' already exists'.format(vocab_file)

        uci_dataset = UciDataset(collection, self.get_dataset_id(a_pipe), docword_file, vocab_file)
        uci_dataset.save()
        print uci_dataset
        return uci_dataset

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
    return OrderedDict([(setting_name, encode_pipeline_cfg[setting_name](value)) for setting_name, value in config.items(section)])


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
    # print 'Arguments given:', args

    nb_docs = args.sample
    if nb_docs != 'all':
        nb_docs = int(nb_docs)

    ph = PipeHandler(args.category, sample=nb_docs)

    # ph.set_doc_gen(args.category, nb_docs)
    pipe = ph.create_pipeline(args.config)
    print '\n', ph.cat2textgen_proc
    print '\n', pipe, '\n'
    uci_dt = ph.preprocess(pipe, args.collection, labels=args.labels)

    print 'nb docs gen:', ph.doc_gen_stats['docs-gen']
    print 'nb docs failed:', ph.doc_gen_stats['docs-failed']
    print ph.cat2textgen_proc.failed
    print 'unique words:', len(ph.dct.items())
    print 'total words:', ph.doc_gen_stats['corpus-tokens']
    print
