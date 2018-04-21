import os
import sys
import argparse
import ConfigParser
from collections import OrderedDict

from pipeline import create_pipeline
from preprocessing import decode_category
from definitions import nb_docs, root_dir, encode_pipeline_cfg


def configfile2dict(config_path, section):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    return OrderedDict([(k, encode_pipeline_cfg[k](value)) for k, value in config.items(section)])


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
        pass

    def create_pipeline(self, pipeline_cfg):

    def pipe_files(self, category, nb_files=None, nb_docs=None):

        l = 0
        for category in decode_category(self.input):
            for pickle_file in cat2files[category]:
                print 'Working with \'{}\' file'.format(pickle_file)
                # for doc in gen_docs(pickle_file, nb_docs):
                #     yield doc
                for i, (_, df_entry) in enumerate(pd.read_pickle(pickle_file).iterrows()):
                    # print nb_docs
                    # sys.exit(0)
                    if i > 15:
                        break
                    # print type(df_entry['text'].encode('utf-8'))
                    yield df_entry['text'].encode('utf-8')
                l += i + 1
        self.length = i





if __name__ == '__main__':
    args = get_cl_arguments()
    print 'Arguments given:', args
    pipeline_settings = configfile2dict(os.path.join(root_dir, 'code', args.config), 'preprocessing')
    for k, v in pipeline_settings.items():
        print k, v
    print

    dpl = create_pipeline(pipeline_settings)
    for pr in dpl.processors:
        print pr
