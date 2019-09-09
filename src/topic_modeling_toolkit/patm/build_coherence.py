#!/usr/bin/env python

import os
import re
import in_place
import subprocess
from glob import glob

import logging

logger = logging.getLogger(__name__)


class CoherenceFilesBuilder:
    def __init__(self, collection_root):
        self._root = collection_root
        self._col_name = os.path.basename(self._root)
        try:
            self._vocab = self._glob("vocab*.txt")[0]
        except IndexError:
            raise VocabularyNotFoundError("Glob pattern '{}' did not match any file in '{}'".format("vocab*.txt", self._root))
        if self._col_name not in os.path.basename(self._vocab):
            logger.warning("{} Instead '{}' found.".format("Vocabulary file usually has the format 'vocab.{col_name}.txt.", os.path.basename(self._vocab)))

        self._splits = sorted([(re.search(r"vowpal\.{}-?([\w\-]*)\.txt".format(self._col_name), f).group(1), f) for f in self._glob("vowpal*.txt")], key=lambda x: x[0], reverse=True)
        if [_[0] for _ in self._splits] != [''] and [_[0] for _ in self._splits] != ['train', 'test']:
            raise InvalidSplitsError("Either 'train' and 'test' splits must be defined or a '' split no splitting; all dataset used for training)")

    def _glob(self, pattern):
        return glob("{}/{}".format(self._root, pattern))

    def _path(self, *args, **kwargs):
        return os.path.join(self._root, '_'.join(map(str, [_ for _ in args if _ != ''])) + (lambda x: '.'+x if x else '')(kwargs.get('extension', '')))

    def create_files(self, cooc_window=5, min_tf=0, min_df=0, apply_zero_index=True):
        _file = {}
        for s, vowpal in self._splits:
            _file.update({s: {'cooc_tf': self._path('cooc', min_tf, 'tf', s, extension='txt'),
                              'cooc_df': self._path('cooc', min_df, 'df', s, extension='txt'),
                              'ppmi_tf': self._path('ppmi', min_tf, 'tf', s, extension='txt'),
                              'ppmi_df': self._path('ppmi', min_df, 'df', s, extension='txt')}
                          })

            _ = self.create_cooc_files(vowpal, self._vocab,
                                       _file[s]['cooc_tf'], _file[s]['cooc_df'],
                                       _file[s]['ppmi_tf'], _file[s]['ppmi_df'],
                                       cooc_window=cooc_window, min_tf=min_tf, min_df=min_df)
            if _ == 0:
                print("Created ppmi files for '{}' split.".format((lambda x: 'all' if not x else x)(s)))
            else:
                print("Something went wrong when creating ppmi files for '{}' split.".format((lambda x: 'all' if not x else x)(s)))

            if apply_zero_index:
                print("Applying zero indexing of tokens (ids)")
                for key, path in _file[s].items():
                    with in_place.InPlace(path, backup_ext='.bak') as fp:
                        for line in fp:
                            match_obj = re.search(r'^(\d+) (\d+) (.+)$', line).groups()
                            fp.write('{} {} {}\n'.format(int(match_obj[0])-1, int(match_obj[1])-1 , match_obj[2]))

    @staticmethod
    def create_cooc_files(vowpal_file, vocab_file, cooc_tf, cooc_df, ppmi_tf, ppmi_df, cooc_window=5, min_tf=0, min_df=0):
        """
        :param str vowpal_file: path to vowpal-formated bag-of-words file
        :param str vocab_file: path to uci-formated (list of unique tokens) vocabulary file
        :param str cooc_tf: file path to save the terms frequency (tf) dictionary of co-occurrences of every specific pair of tokens: total number of times a pair of tokens appears in the dataset
        :param str cooc_df: file path to save the document-frequency (df) dictionary with the number of documents in which each pair of tokens appeared: in how many documents pair 'x' can be found
        :param str ppmi_tf: file path to save values of positive pmi (point-wise mutual information) of pairs of tokens from cooc_tf dictionary
        :param str ppmi_df: file path to save values of positive pmi (point-wise mutual information) of pairs of tokens from cooc_df dictionary
        :param int min_tf: minimal value of cooccurrences of a pair of tokens that are saved in dictionary of cooccurrences
        :param int min_df: minimal value of documents in which a specific pair of tokens occurred together closely
        :param int cooc_window: number of tokens around specific token, which are used in calculation of cooccurrences
        :return:
        """
        # let everything flow into the terminal
        args = ['bigartm', '-c', vowpal_file, '-v', vocab_file, '--cooc-window', str(cooc_window),
                '--cooc-min-tf', str(min_tf), '--write-cooc-tf', cooc_tf,
                '--cooc-min-df', str(min_df), '--write-cooc-df', cooc_df,
                '--write-ppmi-tf', ppmi_tf,
                '--write-ppmi-df', ppmi_df,
                '--force']
        print('Executing:', ' '.join(args))
        return subprocess.call(args)


class VocabularyNotFoundError(Exception): pass
class InvalidSplitsError(Exception): pass


import click

@click.command()
@click.argument('collection')
@click.option('--window', '-w', default=5, show_default=True, help="number of tokens around specific token, which are used in calculation of cooccurrences")
@click.option('--cooc_min_tf', '-min_tf', default=0, show_default=True, help="minimal value of cooccurrences of a pair of tokens that are saved in dictionary of cooccurrences")
@click.option('--cooc_min_df', '-min_df', default=0, show_default=True, help="minimal value of documents in which a specific pair of tokens occurred together closely")
def main(collection, window, cooc_min_tf, cooc_min_df):
    cfb = CoherenceFilesBuilder(os.path.join(collections_root, collection))
    cfb.create_files(cooc_window=window, min_tf=cooc_min_tf, min_df=cooc_min_df)


if __name__ == '__main__':
    collections_root = "/data/thesis/data/collections"
    main()