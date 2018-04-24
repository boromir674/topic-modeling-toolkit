import os
import sys
import argparse
import ConfigParser
from collections import OrderedDict
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.ucicorpus import UciWriter

from pipeline import DefaultPipeline
from definitions import nb_docs, root_dir, encode_pipeline_cfg, cat2files, get_id
from processors import StringToTextGenerator


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
    return parser.parse_args()


class PipeHandler(object):
    def __init__(self):
        self.str2text_proc = None
        self.text_generator = None
        self.doc_gen_stats = {'corpus-tokens': 0}
        self.dct = Dictionary()
        self.model = None
        self.corpus = None
        self.nb_docs = 0
        self.writer = UciWriter('/data/thesis/data/cnguci')

    def create_pipeline(self, pipeline_cfg):
        pipe_settings = configfile2dict(os.path.join(root_dir, 'code', pipeline_cfg), 'preprocessing')
        print pipe_settings
        return DefaultPipeline(pipe_settings)

    def set_doc_gen(self, category, num_docs=None):
        self.str2text_proc = StringToTextGenerator(cat2files, num_docs=num_docs)
        self.text_generator = self.str2text_proc.process(category)

    def pipe_files(self, a_pipe):
        doc_gens = []
        doc_gens2 = []
        sum_toks = 0
        for i, doc in enumerate(self.text_generator):
            gen1 = a_pipe.pipe_through(doc)
            gen2 = a_pipe.pipe_through(doc)
            gen3 = a_pipe.pipe_through(doc)

            doc_gens.append(gen2)
            doc_gens2.append(gen3)
            toks = [_ for _ in gen1]
            sum_toks += len(toks)
            self.dct.add_documents([toks])
            # print [_ for _ in gen][:10]

        self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens2]
        print '{} tokens in all generators\n'.format(sum_toks)
        print '{} items in dictionary'.format(len(self.dct.items()))
        print 'total words in corpus: {}'.format(sum(len(_) for _ in self.corpus))
        print 'num_pos', self.dct.num_pos
        print 'nnz', self.dct.num_nnz

        self.dct.filter_extremes(no_below=a_pipe.settings['no_below'], no_above=a_pipe.settings['no_above'])
        print '{} items in dictionary'.format(len(self.dct.items()))
        print 'num_pos', self.dct.num_pos
        print 'nnz', self.dct.num_nnz

        self.corpus = [self.dct.doc2bow([token for token in tok_gen]) for tok_gen in doc_gens]
        print 'total words in corpus: {}\n'.format(sum(len(_) for _ in self.corpus))
        # print '{} tokens in bow corpus'.format(sum(len(_) for _ in self.corpus))

        with open(a_pipe.processors[-1].fname, 'w') as f:
            f.writelines('{}\n{}\n{}\n'.format(self.dct.num_docs, len(self.dct.items()), sum(len(_) for _ in self.corpus)))
        self.writer.write_corpus(self.writer.fname, self.corpus)

        if a_pipe.settings['weight'] == 'counts':

            # self.writer.fake_headers(self.dct.num_docs, self.dct.num_pos, self.dct.num_nnz)
            # print 'type1', type(self.corpus[0]), type(self.corpus[0])
            # print 'len1:', len(self.corpus), len(self.corpus[0])

            for vec in self.corpus:
                self.doc_gen_stats['corpus-tokens'] += len(vec)
                a_pipe[-1][1].process(vec)
        elif a_pipe.settings['weight'] == 'tfidf':
            self.model = TfidfModel(self.corpus)
            # self.writer.fake_headers(self.dct.num_docs, self.dct.num_pos, self.dct.num_nnz)
            c2 = map(lambda x: self.model[x], self.corpus)
            # print 'type2', type(c2), type(c2[0])
            # print 'len2:', len(c2), len(c2[0])

            # self.writer.write_corpus(self.writer.fname, c2)
            for vec in c2:
                self.doc_gen_stats['corpus-tokens'] += len(vec)
                a_pipe[-1][1].process(vec)

        self.doc_gen_stats.update({'docs-gen': self.str2text_proc.nb_processed, 'docs-failed': len(self.str2text_proc.failed)})


if __name__ == '__main__':
    args = get_cl_arguments()
    print 'Arguments given:', args

    if args.sample == 'all':
        nb_docs = float('inf')
    else:
        nb_docs = eval(args.sample)

    ph = PipeHandler()

    ph.set_doc_gen(args.category, nb_docs)
    pipe = ph.create_pipeline(args.config)
    print '\n', pipe, '\n'

    print get_id(pipe.settings)
    ph.pipe_files(pipe)

    print 'nb docs gen:', ph.doc_gen_stats['docs-gen']
    print 'nb docs failed:', ph.doc_gen_stats['docs-failed']
    print ph.str2text_proc.failed
    print 'unique words:', len(ph.dct.items())
    print 'total words:', ph.doc_gen_stats['corpus-tokens']
