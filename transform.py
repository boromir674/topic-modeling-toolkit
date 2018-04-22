import os
import sys
import argparse
import ConfigParser
from collections import OrderedDict

from pipeline import DefaultPipeline
from definitions import nb_docs, root_dir, encode_pipeline_cfg, cat2files
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.ucicorpus import UciWriter


def configfile2dict(config_path, section):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    return OrderedDict([(pipeline_component, encode_pipeline_cfg[pipeline_component](value)) for pipeline_component, value in config.items(section)])


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='transform.py', description='Transforms pickled panda.DataFrame\'s into Vowpal Wabbit format', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('category', help='the category of data to use')
    parser.add_argument('config', help='the .cfg file to use for constructing a pipeline')
    parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def transform(category, config_path):
    pipe_settings = configfile2dict(os.path.join(root_dir, 'code', config_path), 'preprocessing')


class PipeHandler(object):
    def __init__(self):
        self.dct = Dictionary()
        self.model = None
        self.corpus = None
        self.nb_docs = 0
        self.writer = UciWriter('/data/thesis/data/uci1')

    def create_pipeline(self, pipeline_cfg):
        pipe_settings = configfile2dict(os.path.join(root_dir, 'code', pipeline_cfg), 'preprocessing')
        return DefaultPipeline(pipe_settings)

    def gen_docs(self, category, nb_files=None, num_docs=None):
        for category in category.split('+'):
            for fi, pickle_file in enumerate(cat2files[category]):
                if nb_files and fi >= nb_files:
                    break
                print '{}: Working with \'{}\' file'.format(fi, pickle_file)
                for i, (_, df_entry) in enumerate(pd.read_pickle(pickle_file).iterrows()):
                    if num_docs and i >= num_docs:
                        break
                    # print '{} doc'.format(i)
                    yield df_entry['text'].encode('utf-8')
                    self.nb_docs += 1
        print 'Generated {} documents'.format(self.docs)

    def pipe_files(self, a_pipe, nb_sample=3):
        doc_gens = []
        sum_toks = 0
        for i, doc in enumerate(self.gen_docs('posts')):
            if i == nb_sample - 1:
                break
            # print doc
            doc = a_pipe.partial_pipe(doc, 0, 3)
            # print doc.split(' ')
            gen1 = (w for w in doc.split(' '))
            gen1 = a_pipe.partial_pipe(gen1, 5, 7)
            gen2 = (w for w in doc.split(' '))
            gen2 = a_pipe.partial_pipe(gen2, 5, 7)
            doc_gens.append(gen2)
            toks = [_ for _ in gen1]
            sum_toks += len(toks)
            self.dct.add_documents([toks])
            # print [_ for _ in gen][:10]
        print 'SUM generators\' tokens', sum_toks, self.dct.num_pos, len(self.dct.items())
        self.dct.filter_extremes(no_below=a_pipe.settings['no_below'], no_above=a_pipe.settings['no_above'])
        print 'SUM tokens', self.dct.num_pos, len(self.dct.items())
        self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        print 'Corpus size:', sum(len(_) for _ in self.corpus)
        if a_pipe.settings['weight'] == 'counts':
            # self.writer.fake_headers(self.dct.num_docs, self.dct.num_pos, self.dct.num_nnz)
            print 'type1', type(self.corpus[0]), type(self.corpus[0])
            print 'len1:', len(self.corpus), len(self.corpus[0])
            self.writer.write_corpus(self.writer.fname, self.corpus)
        elif a_pipe.settings['weight'] == 'tfidf':
            self.model = TfidfModel(self.corpus)
            self.writer.fake_headers(self.dct.num_docs, self.dct.num_pos, self.dct.num_nnz)
            c2 = map(lambda x: self.model[x], self.corpus)
            print 'type2', type(c2), type(c2[0])
            print 'len2:', len(c2), len(c2[0])

            self.writer.write_corpus(self.writer.fname, c2)


if __name__ == '__main__':
    args = get_cl_arguments()
    print 'Arguments given:', args

    if args.sample == 'all':
        nb_docs = float('inf')
    else:
        nb_docs = eval(args.sample)

    ph = PipeHandler()
    pipe = ph.create_pipeline(args.config)
    print '\n', pipe, '\n'
    ph.pipe_files(pipe, nb_sample=nb_docs)

    print 'nb docs:', ph.dct.num_docs
    print 'num_words:', ph.dct.num_nnz, ph.dct.num_pos

    # pipeline_settings = configfile2dict(os.path.join(root_dir, 'code', args.config), 'preprocessing')
    # for k, v in pipeline_settings.items():
    #     print k, v
    # print