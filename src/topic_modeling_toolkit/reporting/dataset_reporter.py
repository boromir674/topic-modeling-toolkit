import os
import re
import glob
import argparse
from collections import Counter
from topic_modeling_toolkit.patm.dataset import TextDataset
from topic_modeling_toolkit.patm.definitions import BINARY_DICTIONARY_NAME, COOCURENCE_DICT_FILE_NAMES


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

    def get_infos(self, details=True, selection='all'):
        if selection == 'all' or selection is None:
            selection = self._col_names
        if type(selection) != list:
            selection = [selection]
        return [self._build_str(details=details) for self._cur_col in selection]

    def _build_str(self, details=True):
        if not details:
            return '\n'.join(map(lambda x: '{}/{}'.format(self._cur_col, os.path.basename(x.path)), self.load_dts()))
        else:
            c, n = class_distribution(os.path.join(self._r, self._cur_col, 'vowpal.{}.txt'.format(os.path.basename(self._cur_col))))
            return '{}\n{}\n{}'.format('\n'.join(map(lambda x: self._wrap_color(str(x)), self.load_dts())),
                                       ', '.join(self._extract_files_info()),
                                       'Classes: [{}] with documents distribution [{}]'.format(
                                           ' '.join(sorted(c.keys())), ' '.join('{:.3f}'.format(c[x]/float(n)) for x in sorted(c.keys()))
                                       ))

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


def class_distribution(vowpal_file):

    regs = {'modality' : r'(?:{})'.format('|'.join(['@ideology_class', '@labels_class'])),
           'doc_class_label': r'\w+'}
    rr = re.compile(r'{modality} [\t\ ] ({doc_class_label})'.format(**regs), re.X)

    with open(vowpal_file, 'r') as f:
        # doc_labels = rr.findall(f.read())
        c = Counter(rr.findall(f.read()))
        n = sum(c.values())
    return c, n
