#!/home/kostas/software_and_libs/anaconda2/bin/python

import os
import re
import pickle
import glob
import argparse
from patm.dataset import TextDataset
from patm.definitions import BINARY_DICTIONARY_NAME, COOCURENCE_DICT_FILE_NAMES

class bcolors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DatasetReporter(object):
    files_info = ('vocab', 'vowpal', 'docword', COOCURENCE_DICT_FILE_NAMES[2], COOCURENCE_DICT_FILE_NAMES[3])
    strs = dict(zip(files_info, [lambda x: None, lambda x: 'Vowpal', lambda x: 'Uci', lambda x: "positive 'tf'>{} pmi".format(x), lambda x: "positive 'df'>{} pmi".format(x)]))
    _reg = re.compile('(' + '|'.join(files_info) +')[\w.]*')
    extrs = {COOCURENCE_DICT_FILE_NAMES[2]: lambda x: re.match(re.compile(COOCURENCE_DICT_FILE_NAMES[2] + '(\w*)'), x).group(1),
             COOCURENCE_DICT_FILE_NAMES[3]: lambda x: re.match(re.compile(COOCURENCE_DICT_FILE_NAMES[3] + '(\w*)'), x).group(1)}
    dels = {True: 0, False: 2}

    def __init__(self, collections_root_dir):
        self._r = collections_root_dir
        self._col_names = os.listdir(self._r)
        self._data = {}
        self._cur_col = None
        self._info = []

    def get_infos(self, details=True):
        for self._cur_col in self._col_names:
            self._info.append(self._build_str(details=details))
        return self._info

    def _build_str(self, details=True):
        if not details:
            return '\n'.join(map(lambda x: '{}/{}'.format(self._cur_col, os.path.basename(x.path)), self.load_dts()))
        else:
            return '{}\n{}'.format('\n'.join(map(lambda x: self._wrap_color(str(x)), self.load_dts())),  ', '.join(self._extract_files_info()))

    def _wrap_color(self, b):
        ind = b.index(':')
        return bcolors.UNDERLINE + b[:ind] + bcolors.ENDC + b[ind:]

    def load_dts(self):
        dataset_pattern = '{}/*.pkl'.format(os.path.join(self._r, self._cur_col))
        for pickled_dataset in glob.glob(dataset_pattern):
            dataset_object = None
            try:
                dataset_object = TextDataset.load(pickled_dataset)
            except RuntimeError:
                pass
            if dataset_object:
                yield dataset_object

    def _extract_files_info(self):
        return filter(None, map(lambda x: self.strs[x[0]](self.extrs.get(x[0], lambda y: y)(x[1])), map(lambda x: (x.group(1), x.group()), filter(None, map(lambda file_path: re.match(self._reg, file_path), os.listdir(os.path.join(self._r, self._cur_col)))))))


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on topic-modeling datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--details', '-d', dest='details', default=False, action='store_true', help='Switch to show details about the datasets')
    parser.add_argument('--select-dataset', '-s', dest='dataset_label', help='Whether to show information about a specific dataset only.')
    return parser.parse_args()


if __name__ == '__main__':
    cols_root_dir = '/data/thesis/data/collections'
    args = get_cli_arguments()
    dt_rprt = DatasetReporter(cols_root_dir)
    multiline_datasets_strings = dt_rprt.get_infos(details=args.details)
    if args.dataset_label:
        l = []
        for i, line in enumerate(multiline_datasets_strings):
            if args.dataset_label in line:
                print line
                break
                # l.extend([line] + multiline_datasets_strings[i + 1 : i + 5])
                # break
        # print('\n'.join(l))
    # b = '\n'.join(dt_rprt.get_infos(details=args.details))
    else:
        print '\n'.join(multiline_datasets_strings)
