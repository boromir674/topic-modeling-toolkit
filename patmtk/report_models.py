import os
import re
import sys
import abc
import glob
import time
import argparse
import threading
from functools import reduce
from collections import Counter, defaultdict
from patm.definitions import COLLECTIONS_DIR, RESULTS_DIR_NAME

import numpy as np
from patm.modeling.experimental_results import experimental_results_factory


columns = ['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity',
           'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi', 'sparsity-theta',
           'background-tokens-ratio', 'regularizers']


def get_kernel_sub_hash(entity):
    assert entity in ('coherence', 'contrast', 'purity')
    return {'kernel-'+entity: {'extractor': lambda x: map(lambda y: getattr(y.average, entity).last, x.tracked_kernels),
                               'column-title': lambda x: map(lambda y: 'k'+str(x)[2:]+'.'+entity[:3], x.tracked.kernel_thresholds),
                               'to-string': '{:.4f}'}}


columns_hash = {
    # 'fitness-function': {},
    'nb-topics': {'extractor': lambda x: [x.scalars.nb_topics],
                  'column-title': lambda x: ['tpcs']},
    'collection-passes': {'extractor': lambda x: [x.scalars.dataset_iterations],
                          'column-title': lambda x: ['col-i']},
    'document-passes': {'extractor': lambda x: [x.scalars.document_passes],
                        'column-title': lambda x: ['doc-i']},
    'total-phi-updates': {'extractor': lambda x: [x.scalars.dataset_iterations * x.scalars.document_passes],
                          'column-title': lambda x: ['phi-u']},
    'perplexity': {'extractor': lambda x: [x.tracked.perplexity.last],
                  'column-title': lambda x: ['prpl'],
                  'to-string': lambda x: ['{:.1f}']},
    'top-tokens-coherence': {'extractor': lambda x: map(lambda y: y.average_coherence.last, x.tracked_top_tokens),
                             'column-title': lambda x: map(lambda y: 'top'+str(y)+'c', x.tracked.top_tokens_cardinalities),
                             'to-string': '{:.4f}'},
    'sparsity-phi': {'extractor': lambda x: map(lambda y: y.last, x.phi_sparsities),
                     'column-title': lambda x: map(lambda y: 'spp@'+y, x.tracked.modalities_initials),
                     'to-string': '{:.2f}'},
    'sparsity-theta': {'extractor': lambda x: [x.tracked.sparsity_theta.last],
                       'column-title': lambda x: ['spt'],
                       'to-string': '{:.2f}'},
    'background-tokens-ratio': {'extractor': lambda x: [getattr(x.tracked, 'background_tokens_ratio_'+str(x.scalars.background_tokens_threshold)[2:]).last],
                                'column-title': lambda x: ['btr'+str(x.scalars.background_tokens_threshold)[2:]],
                                'to-string': '{:.2f}'},
    'regularizers': {'extractor': lambda x: ['[{}]'.format(', '.join(x.regularizers))],
                     'column-title': lambda x: ['regs']}
}

columns_hash = reduce(lambda x,y: dict(y, **x), [columns_hash] + map(lambda z: get_kernel_sub_hash(z), ['coherence', 'contrast', 'purity']))


# LEGACY
# def get_extractor(column):
#     if column in extractors_hash:
#         return extractors_hash[column]
#     return lambda x: getattr(x, column)
# def load_results(results_path):
#     with open(results_path, 'r') as results_file:
#         results = results_file.read()
#     return json.loads(results)
# def load_reportables(results_path):
#     d = load_results(results_path)
#     _ = {'model_label': d['model_label'],
#             'nb_topics': d['model_parameters']['_nb_topics'][-1][1],
#             'total': sum(d['collection_passes']) * d['model_parameters']['document_passes'][-1][1],
#             'collection_passes': sum(d['collection_passes']),
#             'document_passes': d['model_parameters']['document_passes'][-1][1],
#             'reg_names': ', '.join(sorted(map(lambda x: x[1]['name'], d['reg_parameters'][-1][1].items()))),
#             'perplexity': float(d['trackables']['perplexity']['value'][-1]),
#             'sparsity-phi': float(d['trackables']['sparsity-phi']['value'][-1]),
#             'sparsity-theta': float(d['trackables']['sparsity-theta']['value'][-1]),
#             'background-tokens-ratio': float(d['trackables']['background-tokens-ratio']['value'][-1]),
#             'kernel-coherence': float(d['trackables']['topic-kernel']['average_coherence'][-1]),
#             'kernel-contrast': float(d['trackables']['topic-kernel']['average_contrast'][-1]),
#             'kernel-purity': float(d['trackables']['topic-kernel']['average_purity'][-1]),
#             'top-10-coherence': float(d['trackables']['top-tokens-10']['average_coherence'][-1]),
#             'top-100-coherence': float(d['trackables']['top-tokens-100']['average_coherence'][-1])
#             }
#     for i, v in _.items():
#         if not v:
#             print 'nV', i, v
#         else:
#             print 'yV', i, v
#     assert all(map(lambda x: _[x] not in (None, ''), _.keys()))
#     return _


class ModelReporter(object):
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    lb = len(r'\033[1m')
    lu = len(r'\033[4m')
    le = len(r'\033[0m')

    def __init__(self, collection_name):
        results_dir = os.path.join(COLLECTIONS_DIR, collection_name, RESULTS_DIR_NAME)
        self._label_separator = ':'
        self._columns_to_render = {column: [] for column in columns}
        self._columns_to_render_sets = {column: set() for column in columns}
        self.result_paths = glob.glob('{}/*.json'.format(results_dir))
        print 'results-list', self.result_paths
        self._transf = {'fit-func':'fitness_value', 'tpcs':'nb_topics', 'col-i':'collection_passes', 'doc-i':'document_passes',
                        'total':'total', 'prplx':'perplexity', 'coher':'kernel-coherence', 'contr':'kernel-contrast',
                        'purity':'kernel-purity', '10coher':'top-10-coherence', '100coher':'top-100-coherence',
                        'sprst-p':'sparsity-phi', 'sprst-t':'sparsity-theta', 'regs':'reg_names'}
        self._to_string_defs = {'fitness_value':'', 'nb_topics':'{}', 'collection_passes':'{}', 'document_passes':'{}',
                                'total':'{}', 'perplexity': '{:.1f}', 'kernel-coherence': '{:.4f}', 'kernel-contrast': '{:.4f}',
                                'kernel-purity': '{:.4f}', 'top-10-coherence': '{:.4f}', 'top-100-coherence': '{:.4f}',
                                'sparsity-phi': '{:.2f}', 'sparsity-theta': '{:.2f}', 'reg_names':'[{}]'}

        # self._biggest_len = max(map(lambda x: len(os.path.basename(x).replace('-train.json', '')), self.result_paths))
        # self._detail_switch = {False: lambda x: x['model_label'], True: self.to_string}
        self.highlight = ''
        self.fitness_computer = None
        self._infos = []
        self._insertions = []
        self._columns_titles = ['fit-func', 'tpcs', 'col-i', 'doc-i', 'total', 'prplx', 'coher', 'contr', 'purity', '10coher', '100coher', 'sprst-p', 'sprst-t', 'regs'] # 14
        # self._header_space_offsets = map(lambda x: 0 if len(x) >= self._to_string_defs[self._transf[x]], self._columns_titles)
        self._header_space_offsets = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0] # fitted design of column headers detailed string output
        # assert len(self._columns_titles) == len(self._header_space_offsets)
        self._header = ' '.join(map(lambda x: x[1] + ' '*self._header_space_offsets[x[0]], enumerate(self._columns_titles)))
        self._max_col_lens = []
        # stripped_phi_names = [os.path.basename(phi_path).replace('.phi', '') for phi_path in glob.glob('{}/*-train.phi'.format(models_dir))]
        # if stripped_phi_names != stripped_result_names:
        #     print '{} phi matrices do not correspond to resuls jsons'.format([_ for _ in stripped_phi_names if _ not in stripped_result_names], )
        #     print '{} results jsons do not correspond to phi matrices'.format([_ for _ in stripped_result_names if _ not in stripped_phi_names])

    # def get_value(self, column, experimental_results):
    #
    #     if column == 'fitness-function':
    #         return [None]
    #     if column in extractors_hash:
    #         return [extractors_hash[column](experimental_results)]
    #     if hasattr(experimental_results.scalars, column.replace('-', '_')):
    #         return [getattr(experimental_results.scalars, column.replace('-', '_'))]
    #     if column.startswith('kernel'):
    #         return map(lambda x: getattr(x.average, column.split('-')[-1]).last, experimental_results.tracked_kernels)
    #     if column.startswith('top-tokens'):
    #         return map(lambda x: x.average_coherence.last, experimental_results.tracked_top_tokens)
    #     if column.startswith('sparsity-phi'):
    #         return map(lambda x: x.last, experimental_results.phi_sparsities)
    #     if column.startswith('background-tokens-ratio-'):
    #         return [getattr(experimental_results.tracked, 'background_tokens_ratio_' + column.split('-')[-1][2:]).last]
    #     return [getattr(experimental_results.tracked, column.replace('-', '_')).last]
    # def get_column_title(self, column, experimental_results):
    #     if column == 'fitness-function':
    #         return [None]
    #     if column.
    #     if hasattr(experimental_results.scalars, column.replace('-', '_')):
    #         return [getattr(experimental_results.scalars, column.replace('-', '_'))]
    #     if column.startswith('kernel'):
    #         eval_defs = map(lambda x: getattr(x.average, column.split('-')[-1]), experimental_results.tracked_kernels)
    #         return map(lambda x: getattr(getattr(experimental_results.tracked, 'kernel'+str(x[2:])).average, column.split('-')[1]).last, experimental_results.kernel_thresholds)
    #     if column.startswith('top-tokens'):
    #         return map(lambda x: getattr(getattr(experimental_results.tracked, 'top' + str(x)),
    #                                      'average_' + column.split('-')[1]).last, experimental_results.top_tokens_cardinalities)
    #     if column.startswith('sparsity-phi-@'):
    #         return [getattr(experimental_results.tracked, 'sparsity_phi_' + column.split('@')[-1][0]).last]
    #     if column.startswith('background-tokens-ratio-'):
    #         return [getattr(experimental_results.tracked, 'background_tokens_ratio_' + column.split('-')[-1][2:]).last]
    #     return [getattr(experimental_results.tracked, column.replace('-', '_')).last]

    def get_values(self, column, experimental_results):
        return columns_hash[column]['extractor'](experimental_results)

    def get_columns_abbrvs(self, column, experimental_results):
        return columns_hash[column]['column-title'](experimental_results)

    def get_values_n_columns(self, column, experimental_results):
        return [columns_hash[column]['extractor'](experimental_results),
                columns_hash[column]['column-title'](experimental_results)]

    def result_path2info_element(self, result_path):
        return experimental_results_factory.create_from_json_file(result_path)

    def info_element2string(self, info_element):
        st = '{}{}{}'.format(info_element.scalars.model_label, self._max_label_len, self._label_separator)
        self.get_values_n_columns()
        st += ' '.join(map(lambda x: self.get_values_n_columns(), columns))

    def _exp_res(self, result_path):
        print 'MODEL_REPORTER._exp_res.results_path', result_path
        return experimental_results_factory.create_from_json_file(result_path)

    def _get_columns_to_render(self, result_path_list):
        # return dict(map(lambda z: (z, reduce(lambda x,y: x+y, map(lambda t: self._build_column_to_render(t, z), columns))), result_path_list))
        return dict(map(lambda z: (z, reduce(lambda x,y: x+y, map(lambda t: self._build_column_to_render(z, self._exp_res(t)), result_path_list))), columns))
        # {column: reduce(lambda x,y: x+y, )}

    def _build_column_to_render(self, column, experimental_results):
        print 'exp type:', type(experimental_results), experimental_results
        print 'c', column
        print 'keys', columns_hash[column].keys()
        _ = columns_hash[column]['column-title'](experimental_results)
        print 'sub-cols:', _
        # self._columns_to_render[column].extend(filter(None, map(lambda x: self._filter_to_add(column, x), _)))
        _ = filter(None, map(lambda x: self._filter_to_add(column, x), _))
        # self._columns_to_render_sets[col
        return _

    def _filter_to_add(self, column, column_title):
        if column_title in self._columns_to_render_sets[column]: return None
        return column_title

    def adw(self):
        c = self._get_columns_to_render(self.result_paths)
        import pprint
        print 'ADW'
        pprint.pprint(c, indent=2)

    def _build_column_value_array(self, column, expimental_results):
        arr = np.full((len(self._columns_to_render[column])), None)
        result_column_titles = self._adapter(columns_hash[column]['column-title'](expimental_results))
        result_column_values = self._adapter(columns_hash[column]['extractor'](expimental_results))
        np.put(arr, map(lambda x: self._columns_to_render[column].index(x), result_column_titles), result_column_values)
        return arr

    def _set_columns_to_render(self):
        pass

    def compute_info_list(self, sort_function=False, add_fitness_info=False):
        if sort_function:
            self.fitness_computer = FitnessCalculator(sort_function)
            self._to_string_defs['fitness_value'] = self._to_string_defs[sort_function.single_metric]
            self._infos = sorted(map(lambda x: load_reportables(x), self.result_paths),
                                 key=lambda x: self.fitness_computer.compute_fitness(x, add_info=add_fitness_info), reverse=True)
            self._max_col_lens = list(map(lambda col_abrv:
                                          max(map(lambda reportable:
                                                  len(self._stringnify(reportable[self._transf[col_abrv]], col_abrv)), self._infos)), self._columns_titles))
        else:
            self._infos = map(lambda x: load_reportables(x), self.result_paths)
        self._insertions = [[] for _ in range(len(self._infos))]

    def get_model_labels_string(self, sorting_function):
        self.compute_info_list(sort_function=sorting_function, add_fitness_info=False)
        return '\n'.join(map(lambda x: x['model_label'], self._infos))

    def get_highlighted_detailed_string(self, sorting_function, highlight_type):
        self.highlight = highlight_type
        self.compute_info_list(sort_function=sorting_function, add_fitness_info=True)
        return '{}\n{}'.format(' '*(self._biggest_len+2)+self._header, '\n'.join(map(lambda x: self._convert_to_row_string(x[0], list(x[1])), map(lambda x: self._make_columns(x), self._infos))))

    def _make_columns(self, reportable):
        return reportable['model_label']+' '*(self._biggest_len-len(reportable['model_label']))+': ', map(lambda x: reportable[self._transf[x]], self._columns_titles)

    def _convert_to_row_string(self, prefix_string, values_list):
        return '{}{}'.format(prefix_string, ' '.join(map(lambda x: self._transform(x[1], x[0]), enumerate(values_list))))

    def _transform(self, column_value, column_index):
        column_full_name = self._transf[self._columns_titles[column_index]] # +' '*(self._max_col_lens[x[0]]-len()
        string_value = self._to_string_defs[column_full_name].format(column_value)
        if self._columns_titles[column_index] == 'col-i':
            assert len(self._columns_titles[column_index]) > self._max_col_lens[column_index]
        if len(self._columns_titles[column_index]) > self._max_col_lens[column_index]:
            gaps = len(self._columns_titles[column_index]) - len(string_value)
        else:
            gaps = self._max_col_lens[column_index] - len(string_value)
        space_post_fix = ' ' * gaps
        if column_value == self.fitness_computer.best.get(column_full_name, -1):
            return self.highlight + string_value + self.ENDC + space_post_fix
        return string_value + space_post_fix

    def _stringnify(self, column_value, column_abrv):
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

def _column_title2abrv(eval_definition):
    return _eval_def2abbrv_hash.get(eval_definition, _get_label(eval_definition))

_eval_def2abbrv_hash = {'perplexity': 'prpl',
                        'collection-passes': 'col-i'}


def _get_label(eval_definition):
    tokens = []
    for el in eval_definition.split('-'):
        try:
            _ = float(el)
            if int(_) == _:
                tokens.append(str(int(_)))
            else:
                tokens.append(str(_))
        except ValueError:
            if el[0] == '@':
                tokens.append(el[:2])
            else:
                tokens.append(el[0])
    return ''.join(tokens)


def _get_eval_type(eval_definition):
    tokens = []
    for el in eval_definition.split('-'):
        try:
            _ = float(el)
        except ValueError:
            if el[0] != '@':
                tokens.append(el)
    return '-'.join(tokens)


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on existing saved model topic model instances developed on the specified dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('--details', '-d', default=False, action='store_true', help='Switch to show details about the models')
    parser.add_argument('--sort', '-s', default='', help='Whether to sort the found experiments by checking the desired metric against the corresponding models')
    return parser.parse_args()

if __name__ == '__main__':

    cli_args = get_cli_arguments()
    print _get_label('sparsity-phi')
    print _get_label('sparsity-theta-@ic')
    print _get_label('sparsity-theta-@dc')
    print _get_label('topic-kernel-0.6')
    print _get_label('top-tokens-10')

    # exp_results = experimental_results_factory.create_from_json_file(os.path.join(COLLECTIONS_DIR, cli_args.dataset, RESULTS_DIR_NAME))
    reporter = ModelReporter(cli_args.dataset)

    reporter.adw()

    # if cli_args.sort:
    #     fitness_function = fitness_function_factory.get_single_metric_function(cli_args.sort)
    # else:
    #     fitness_function = False
    # model_reporter = ModelReporter(cli_args.dataset)
    # spinner = Spinner(delay=0.2)
    # spinner.start()
    # try:
    #     if cli_args.details:
    #         b1 = model_reporter.get_highlighted_detailed_string(fitness_function, model_reporter.UNDERLINE)
    #     else:
    #         b1 = model_reporter.get_model_labels_string(fitness_function)
    # except RuntimeError as e:
    #     spinner.stop()
    #     raise e
    # spinner.stop()
    # print(b1)
