#!/home/kostas/software_and_libs/anaconda2/bin/python

import argparse
import os
import sys

from patm.definitions import COLLECTIONS_DIR_PATH
from patm.build_coherence import CoherenceFilesBuilder
from processors import PipeHandler


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='transform.py', description='Extracts from pickled data, preprocesses text and saves all necessary files to train an artm model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('category', help='the category of data to use')
    parser.add_argument('config', help='the .cfg file to use for constructing a pipeline')
    parser.add_argument('collection', help='a given name for the collection')
    parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    parser.add_argument('--window', '-w', default=10, type=int, help='number of tokens around specific token, which are used in calculation of cooccurrences')
    parser.add_argument('--min_tf', default=0, type=int,
                        help='Minimal value of cooccurrences of a pair of tokens that are saved in dictionary of cooccurrences.')
                             # 'For each int value a file is built to be used for coherence computation. By default builds one with min_tf=0')
    parser.add_argument('--min_df', default=0, type=int,
                        help='Minimal value of documents in which a specific pair of tokens occurred together closely.')
                             # 'For each int value a file is built to be used for coherence computation. By default builds one with min_df=0')
    parser.add_argument('--exclude_class_labels_from_vocab', '--ecliv', action='store_true', default=False, help='Whether to ommit adding document labels in the vocabulary file generated. If set to False document labels are treated as valid registered vocabulary tokens.')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()
    nb_docs = args.sample
    if nb_docs != 'all':
        nb_docs = int(nb_docs)
    ph = PipeHandler(COLLECTIONS_DIR_PATH, args.category, sample=nb_docs)
    ph.pipeline = args.config
    # ph.create_pipeline(args.config)
    print '\n', ph.pipeline, '\n'

    uci_dt = ph.preprocess(args.collection, not args.exclude_class_labels_from_vocab)
    print uci_dt

    print '\nBuilding coocurences information'
    coherence_builder = CoherenceFilesBuilder(os.path.join(COLLECTIONS_DIR_PATH, args.collection))
    coherence_builder.create_files(cooc_window=args.window,
                                   min_tf=args.min_tf,
                                   min_df=args.min_df,
                                   apply_zero_index=False)
    # ph.write_cooc_information(args.window, args.min_tf, args.min_df)
