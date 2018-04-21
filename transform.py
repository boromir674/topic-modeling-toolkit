import os
import sys
import argparse
import ConfigParser
from collections import OrderedDict

from pipeline import DefaultPipeline
from definitions import nb_docs, root_dir, encode_pipeline_cfg, cat2files
import pandas as pd
from gensim.corpora import Dictionary


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
                    print '{} doc'.format(i)
                    yield df_entry['text'].encode('utf-8')

    def pipe_files(self, a_pipe, nb_sample=3):
        for i, doc in enumerate(self.gen_docs('posts')):
            if i == nb_sample - 1:
                break
            # print doc
            doc = a_pipe.partial_pipe(doc, 0, 4)
            # print doc.split(' ')
            gen = (w for w in doc.split(' '))
            gen = a_pipe.partial_pipe(gen, 5, 7)
            # print [_ for _ in gen]
            us  = [_ for _ in gen]
            print us
            # dct.add_documents([[_ for _ in gen]])
            self.dct.add_documents([us])
            # print [_ for _ in gen][:10]
        # dct.filter_extremes()


if __name__ == '__main__':
    args = get_cl_arguments()
    print 'Arguments given:', args

    ph = PipeHandler()
    pipe = ph.create_pipeline(args.config)
    print pipe
    ph.pipe_files(pipe)

    # pipeline_settings = configfile2dict(os.path.join(root_dir, 'code', args.config), 'preprocessing')
    # for k, v in pipeline_settings.items():
    #     print k, v
    # print