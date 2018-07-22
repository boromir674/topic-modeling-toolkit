import os
import re
import sys
import abc
import glob
import time
import json
import argparse
import threading
from functools import reduce
from collections import Counter
from collections import defaultdict


results_root = 'results'
models_root = 'models'
collections_dir = '/data/thesis/data/collections'

def load_results(results_path):
    with open(results_path, 'r') as results_file:
        results = results_file.read()
    return json.loads(results)

def load_reportables(results_path):
    d = load_results(results_path)
    return {'model_label': d['model_label'],
            'nb_topics': d['model_parameters']['nb_topics'][-1][1],
            'total': sum(d['collection_passes']) * d['model_parameters']['document_passes'][-1][1],
            'collection_passes': sum(d['collection_passes']),
            'document_passes': d['model_parameters']['document_passes'][-1][1],
            'reg_names': ', '.join(sorted(map(lambda x: x[1]['name'], d['reg_parameters'][-1][1].items()))),
            'perplexity': float(d['trackables']['perplexity']['value'][-1]),
            'sparsity-phi': float(d['trackables']['sparsity-phi']['value'][-1]),
            'sparsity-theta': float(d['trackables']['sparsity-theta']['value'][-1]),
            'background-tokens-ratio': float(d['trackables']['background-tokens-ratio']['value'][-1]),
            'kernel-coherence': float(d['trackables']['topic-kernel']['average_coherence'][-1]),
            'kernel-contrast': float(d['trackables']['topic-kernel']['average_contrast'][-1]),
            'kernel-purity': float(d['trackables']['topic-kernel']['average_purity'][-1]),
            'top-10-coherence': float(d['trackables']['top-tokens-10']['average_coherence'][-1]),
            'top-100-coherence': float(d['trackables']['top-tokens-100']['average_coherence'][-1])
            }


class ModelReporter(object):
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    lb = len(r'\033[1m')
    lu = len(r'\033[4m')
    le = len(r'\033[0m')

    def __init__(self, collection_name):
        self._r = None
        results_dir = os.path.join(collections_dir, collection_name, results_root)
        self._transf = {'fit-func':'fitness_value', 'tpcs':'nb_topics', 'col-iters':'collection_passes', 'total': 'total', 'prplx':'perplexity', 'coher':'kernel-coherence', 'contr':'kernel-contrast', 'purity':'kernel-purity',
                        '10coher':'top-10-coherence', '100coher':'top-100-coherence', 'sprst-p':'sparsity-phi', 'sprst-t':'sparsity-theta', 'regs':'reg_names'}
        self._to_string_defs = {'fitness_value':'', 'nb_topics':'{}', 'collection_passes':'{}', 'total':'{}', 'perplexity': '{:.1f}', 'kernel-coherence': '{:.4f}', 'kernel-contrast': '{:.4f}', 'kernel-purity': '{:.4f}',
                                'top-10-coherence': '{:.4f}', 'top-100-coherence': '{:.4f}', 'sparsity-phi': '{:.2f}', 'sparsity-theta': '{:.2f}', 'reg_names':'[{}]'}
        self.result_paths = glob.glob('{}/*-train.json'.format(results_dir))
        self._biggest_len = max(map(lambda x: len(os.path.basename(x).replace('-train.json', '')), self.result_paths))
        # self._detail_switch = {False: lambda x: x['model_label'], True: self.to_string}
        self.highlight = ''
        self.fitness_computer = None
        self._infos = []
        self._row_strings = []
        self._insertions = []
        self._columns_titles = ['fit-func', 'tpcs', 'col-iters', 'total', 'prplx', 'coher', 'contr', 'purity', '10coher', '100coher', 'sprst-p', 'sprst-t', 'regs'] # 13
        # self._header_space_offsets = map(lambda x: 0 if len(x) >= self._to_string_defs[self._transf[x]], self._columns_titles)
        self._header_space_offsets = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0] # fitted design of column headers detailed string output
        self._header = ' '.join(map(lambda x: x[1] + ' '*self._header_space_offsets[x[0]], enumerate(self._columns_titles)))
        self._max_col_lens = []
        # stripped_phi_names = [os.path.basename(phi_path).replace('.phi', '') for phi_path in glob.glob('{}/*-train.phi'.format(models_dir))]
        # if stripped_phi_names != stripped_result_names:
        #     print '{} phi matrices do not correspond to resuls jsons'.format([_ for _ in stripped_phi_names if _ not in stripped_result_names], )
        #     print '{} results jsons do not correspond to phi matrices'.format([_ for _ in stripped_result_names if _ not in stripped_phi_names])

    def get_info_list(self, sort_function=False, add_fitness_info=False):
        if sort_function:
            self.fitness_computer = FitnessCalculator(sort_function)
            self._to_string_defs['fitness_value'] = self._to_string_defs[sort_function.single_metric]
            self._infos = sorted(map(lambda x: load_reportables(x), self.result_paths), key=lambda x: self.fitness_computer.compute_fitness(x, add_info=add_fitness_info), reverse=True)
            self._max_col_lens = list(map(lambda col_abrv: max(map(lambda reportable: len(self._stringnify1(reportable[self._transf[col_abrv]], col_abrv)), self._infos)), self._columns_titles))
        else:
            self._infos = map(lambda x: load_reportables(x), self.result_paths)
        self._insertions = [[] for _ in range(len(self._infos))]
        return self._infos

    def get_model_labels_string(self, sorting_function):
        il = self.get_info_list(sort_function=sorting_function, add_fitness_info=False)
        return '\n'.join(map(lambda x: x['model_label'], il))

    def get_highlighted_detailed_string1(self, sorting_function, highlight_type):
        self.highlight = highlight_type
        _ = self.get_info_list(sort_function=sorting_function, add_fitness_info=True)
        return '{}\n{}'.format(' '*(self._biggest_len+2)+self._header, '\n'.join(map(lambda x: self._convert_to_row_string(x[0], list(x[1])), map(lambda x: self._make_columns(x), self._infos))))

    def _make_columns(self, reportable):
        return reportable['model_label']+' '*(self._biggest_len-len(reportable['model_label']))+': ', map(lambda x: reportable[self._transf[x]], self._columns_titles)

    def _convert_to_row_string(self, prefix_string, values_list):
        return '{}{}'.format(prefix_string, ' '.join(map(lambda x: self._transform(x[1], x[0]), enumerate(values_list))))

    def _transform(self, column_value, column_index):
        column_full_name = self._transf[self._columns_titles[column_index]] # +' '*(self._max_col_lens[x[0]]-len()
        string_value = self._to_string_defs[column_full_name].format(column_value)
        if self._columns_titles[column_index] == 'col-iters':
            assert len(self._columns_titles[column_index]) > self._max_col_lens[column_index]
        if len(self._columns_titles[column_index]) > self._max_col_lens[column_index]:
            gaps = len(self._columns_titles[column_index]) - len(string_value)
        else:
            gaps = self._max_col_lens[column_index] - len(string_value)
        space_post_fix = ' ' * gaps
        if column_value == self.fitness_computer.best.get(column_full_name, -1):
            return self.highlight + string_value + self.ENDC + space_post_fix
        return string_value + space_post_fix

    def _stringnify(self, column_value, column_index):
        return self._to_string_defs[self._transf[self._columns_titles[column_index]]].format(column_value)

    def _stringnify1(self, column_value, column_abrv):
        return self._to_string_defs[self._transf[column_abrv]].format(column_value)

class FitnessFunction:
    def __init__(self, extractors, coefficients, names, ordering='natural'):
        self._extr = extractors
        self._coeff = coefficients
        self._names = names
        assert ordering in ('natural', 'reversed')
        self.order = ordering
        if len(self._names) == 1:
            self.single_metric = self._names[0]
        assert sum(map(lambda x: abs(x), self._coeff)) - 1 < abs(1e-6)

    def compute(self, individual):
        c = map(lambda x: x, self._extr)
        cc = zip(map(lambda x: x, self._coeff), map(lambda x: x(individual), c))
        c1 = map(lambda x: x[0]*x[1], cc)
        return reduce(lambda x,y: x+y, c1)

    def __call__(self, *args, **kwargs):
        return self.compute(args[0])
    def __str__(self):
        return ' + '.join(map(lambda x: '{}*{}'.format(x[0], x[1]), zip(self._names, self._coeff)))

def create_key_extractor(key): # demek
    return lambda x: x[key]

class FitnessFunctionBuilder:
    def __init__(self):
        self._extractors = []
        self._coeff_values = []
        self._order = ''
        self._names = []
    def new(self, ordering='natural'):
        self._extractors = []
        self._coeff_values = []
        self._order = ordering
        self._names = []
        return self
    def coefficient(self, name, value):
        self._extractors.append(create_key_extractor(name))
        self._coeff_values.append(value)
        self._names.append(name)
        return self
    def build(self):
        return FitnessFunction(self._extractors, self._coeff_values, self._names, ordering=self._order)

fitness_function_builder = FitnessFunctionBuilder()

_orders = defaultdict(lambda: 'natural', perplexity='reversed')

class FitnessFunctionFactory:
    def __init__(self):
        self._functions = {}

    def get_single_metric_function(self, name):
        return self._functions.get(name, fitness_function_builder.new(ordering=_orders[name]).coefficient(name, 1).build())

    def linear_combination_function(self, function_expression):
        pass
        # FUTURE WORK to support custom linnear combination of evaluation metrics

fitness_function_factory = FitnessFunctionFactory()

class FitnessCalculator(object):
    def __init__(self, calculating_function):
        self.func = calculating_function
        self._best = {'perplexity': float('inf'), 'kernel-coherence': 0, 'kernel-purity': 0, 'kernel-contrast': 0,
                      'top-10-coherence': 0, 'top-100-coherence': 0, 'sparsity-phi': 0, 'sparsity-theta': 0}

    def set_function(self, calc_function):
        self.func = calc_function

    def compute_fitness(self, model_reportable, add_info=False):
        value = self.func(model_reportable)
        for k, v in self._best.items():
            el1 = fitness_value_ordering2constructor[_orders[k]](model_reportable[k])
            el2 = fitness_value_ordering2constructor[_orders[k]](v)
            # print k, _orders[k], el1.value, el2.value
            if el1 > el2:
                self._best[k] = model_reportable[k]
        if add_info:
            model_reportable['fitness_function'] = str(self.func)
            model_reportable['fitness_value'] = value
        return fitness_value_ordering2constructor[self.func.order](value)

    @property
    def best(self):
        return self._best


class FitnessValue(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, value):
        self.value = value
    def __abs__(self):
        return abs(self.value)
    def __eq__(self, other):
        return self.value == other.value
    def __ne__(self, other):
        return not self.__ne__(other)
class NaturalFitnessValue(FitnessValue):
    def __init__(self, value):
        super(NaturalFitnessValue, self).__init__(value)
    def __lt__(self, other):
        return self.value < other.value
    def __le__(self, other):
        return self.value <= other.value
    def __gt__(self, other):
        return self.value > other.value
    def __ge__(self, other):
        return self.value >= other.value
class ReversedFitnessValue(FitnessValue):
    def __init__(self, value):
        super(ReversedFitnessValue, self).__init__(value)
    def __lt__(self, other):
        return self.value > other.value
    def __le__(self, other):
        return self.value >= other.value
    def __gt__(self, other):
        return self.value < other.value
    def __ge__(self, other):
        return self.value <= other.value

fitness_value_ordering2constructor = {'natural': NaturalFitnessValue, 'reversed': ReversedFitnessValue}

class Spinner:
    busy = False
    delay = 0.1
    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor
    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay
    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()
    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()
    def stop(self):
        self.busy = False
        time.sleep(self.delay)

def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on existing saved model topic model instances developed on the specified dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('--details', '-d', default=False, action='store_true', help='Switch to show details about the models')
    parser.add_argument('--sort', '-s', default='', help='Whether to sort the found experiments by checking the desired metric against the corresponding models')
    return parser.parse_args()

if __name__ == '__main__':
    collections_root_dir = '/data/thesis/data/collections'
    cli_args = get_cli_arguments()

    if cli_args.sort:
        fitness_function = fitness_function_factory.get_single_metric_function(cli_args.sort)
    else:
        fitness_function = False
    model_reporter = ModelReporter(cli_args.dataset)
    spinner = Spinner(delay=0.2)
    spinner.start()
    try:
        if cli_args.details:
            b1 = model_reporter.get_highlighted_detailed_string1(fitness_function, model_reporter.UNDERLINE)
        else:
            b1 = model_reporter.get_model_labels_string(fitness_function)
    except RuntimeError as e:
        spinner.stop()
        raise e
    spinner.stop()
    print(b1)
