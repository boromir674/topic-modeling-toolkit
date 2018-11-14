#!/home/kostas/software_and_libs/anaconda2/bin/python2.7

import os
import re
import sys
import abc
import glob
import time
import argparse
import threading
from collections import Counter, defaultdict, Iterable
from patm.definitions import COLLECTIONS_DIR, RESULTS_DIR_NAME

import numpy as np
from patm.modeling.experimental_results import experimental_results_factory


COLUMNS = ['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity',
           'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi', 'sparsity-theta',
           'background-tokens-ratio', 'regularizers']

KERNEL_SUB_ENTITIES = ('coherence', 'contrast', 'purity')

def get_kernel_sub_hash(entity):
    assert entity in KERNEL_SUB_ENTITIES
    return {'kernel-'+entity: {'extractor': lambda x,y: getattr(x.tracked, 'kernel'+y[2:]).average.last if hasattr(x.tracked, 'kernel'+y[2:]) else None,
                               'column-title': lambda x: 'k'+entity[:3:2]+'.'+str(x)[2:],
                               'to-string': '{:.4f}',
                               'definitions': lambda x: map(lambda z: 'kernel-{}-{:.2f}'.format(entity, z), x.tracked.kernel_thresholds)}}

COLUMNS_HASH = {
    # 'fitness-function': {},
    'nb-topics': {'extractor': lambda x: x.scalars.nb_topics,
                  'column-title': lambda: 'tpcs'},
    'collection-passes': {'extractor': lambda x: x.scalars.dataset_iterations,
                          'column-title': lambda: 'col-i'},
    'document-passes': {'extractor': lambda x: x.scalars.document_passes,
                        'column-title': lambda: 'doc-i'},
    'total-phi-updates': {'extractor': lambda x: x.scalars.dataset_iterations * x.scalars.document_passes,
                          'column-title': lambda: 'phi-u'},
    'perplexity': {'extractor': lambda x: x.tracked.perplexity.last,
                  'column-title': lambda: 'prpl',
                  'to-string': '{:.1f}'},
    'top-tokens-coherence': {'extractor': lambda x,y: getattr(x.tracked, 'top'+str(y)).average_coherence.last if hasattr(x.tracked, 'top'+str(y)) else None,
                             'column-title': lambda x: 'top'+str(x)+'ch',
                             'to-string': '{:.4f}',
                             'definitions': lambda x: map(lambda y: 'top-tokens-coherence-'+str(y), x.tracked.top_tokens_cardinalities)},
    'sparsity-phi': {'extractor': lambda x,y: getattr(x.tracked, 'sparsity_phi_'+y).last if hasattr(x.tracked, 'sparsity_phi_'+y) else None,
                     'column-title': lambda y: 'spp@'+y,
                     'to-string': '{:.2f}',
                     'definitions': lambda x: map(lambda z: 'sparsity-phi-{}'.format(z), x.tracked.modalities_initials)},
    'sparsity-theta': {'extractor': lambda x: x.tracked.sparsity_theta.last,
                       'column-title': lambda: 'spt',
                       'to-string': '{:.2f}'},
    'background-tokens-ratio': {'extractor': lambda x,y: getattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]).last if hasattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]) else None,
                                'column-title': lambda x: 'btr.'+str(x)[2:],
                                'to-string': '{:.2f}',
                                'definitions': lambda x: map(lambda y: 'background-tokens-ratio-{:.2f}'.format(y), x.tracked.background_tokens_thresholds)},
    'regularizers': {'extractor': lambda x: '[{}]'.format(', '.join(x.regularizers)),
                     'column-title': lambda: 'regs'}
}

COLUMNS_HASH = reduce(lambda x, y: dict(y, **x), [COLUMNS_HASH] + map(lambda z: get_kernel_sub_hash(z), KERNEL_SUB_ENTITIES))


def _parse_column_definition(definition):
    return map(lambda y: list(filter(None, y)), zip(*map(lambda x: (x, None) if _is_token(x) else (None, x), definition.split('-'))))


def _is_token(definition_element):
    try:
        _ = float(definition_element)
        return False
    except ValueError:
        if definition_element[0] == '@' or len(definition_element) == 1:
            return False
        return True


class ResultsExtractor(object):
    def __init__(self, extractable_entities):
        self._columns = extractable_entities

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        self._columns = columns

    def __call__(self, *args):
        return self.extract_all(args[0])

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
    DYNAMIC_COLUMNS = ['kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi', 'background-tokens-ratio']

    def __init__(self):
        self._label_separator = ':'
        self._columns_to_render = []
        self.result_paths = []
        self._strings_lists = []
        self.extractor = ResultsExtractor(self._columns_to_render)
        #
        # self._transf = {'fit-func':'fitness_value', 'tpcs':'nb_topics', 'col-i':'collection_passes', 'doc-i':'document_passes',
        #                 'total':'total', 'prplx':'perplexity', 'coher':'kernel-coherence', 'contr':'kernel-contrast',
        #                 'purity':'kernel-purity', '10coher':'top-10-coherence', '100coher':'top-100-coherence',
        #                 'sprst-p':'sparsity-phi', 'sprst-t':'sparsity-theta', 'regs':'reg_names'}
        # self._to_string_defs = {'fitness_value':'', 'nb_topics':'{}', 'collection_passes':'{}', 'document_passes':'{}',
        #                         'total':'{}', 'perplexity': '{:.1f}', 'kernel-coherence': '{:.4f}', 'kernel-contrast': '{:.4f}',
        #                         'kernel-purity': '{:.4f}', 'top-10-coherence': '{:.4f}', 'top-100-coherence': '{:.4f}',
        #                         'sparsity-phi': '{:.2f}', 'sparsity-theta': '{:.2f}', 'reg_names':'[{}]'}

        # self._biggest_len = max(map(lambda x: len(os.path.basename(x).replace('-train.json', '')), self.result_paths))
        # self._detail_switch = {False: lambda x: x['model_label'], True: self.to_string}
        self.highlight = ''
        self.fitness_computer = None
        self._infos = []
        self._insertions = []
        self._header = ' '.join(map(lambda x: x[1] + ' '*self._header_space_offsets[x[0]], enumerate(self._columns_titles)))
        self._max_label_len = 0
        self._max_col_lens = []

    def _initialize(self, collection_name, custom_columns=None):
        results_dir = os.path.join(COLLECTIONS_DIR, collection_name, RESULTS_DIR_NAME)
        self.result_paths = glob.glob('{}/*.json'.format(results_dir))
        self._max_label_len = max(map(lambda x: ModelReporter._get_label(x), self.result_paths), key=lambda x: len(x))
        if not custom_columns:
            self.columns_to_render = self._sort(self._determine_maximal_column_for_rendering(self.result_paths))
        else:
            try:
                self.columns_to_render = custom_columns
            except InvalidColumnsException as e:
                print e
                print 'Automatically using all possible columns inferred from the experimental results to report'
                self.columns_to_render = self._sort(self._determine_maximal_column_for_rendering(self.result_paths))
        self._columns_titles = map(lambda z: COLUMNS_HASH['-'.join(z[0])]['column-title'](*z[1]), map(lambda x: _parse_column_definition(x), self.columns_to_render))
        self._max_col_lens = map(lambda x: len(x), self._columns_titles)

    @property
    def columns_to_render(self):
        return self._columns_to_render

    @columns_to_render.setter
    def columns_to_render(self, custom_columns):
        if not isinstance(custom_columns, Iterable):
            raise InvalidColumnsException(
                "Input columns are of type '{}' instead of iterable".format(type(custom_columns)))
        invalid_columns = ModelReporter._get_invalid_columns(custom_columns)
        if invalid_columns:
            raise InvalidColumnsException("Input columns [{}] are not valid".format(', '.join(invalid_columns)))
        self._columns_to_render = custom_columns
        self.extractor.columns = custom_columns

    def report(self, collection_name, details=None, sort=None, columns=None):
        self._initialize(collection_name, custom_columns=columns)
        self._strings_lists = self._get_strings_lists()

        print 'To render:'
        print '[{}]'.format(', '.join(self.columns_to_render))
        print '[{}]'.format(', '.join(self._columns_titles))
        print '[{}]'.format(', '.join(map(lambda x: str(x), self._max_col_lens)))
        st = '\n'.join(map(lambda x: ' '.join(self._exp_result2_elements_list(x)), self.result_paths))
        print st

    def _exp_result2_elements_list(self, json_path):
        return [str(ModelReporter._get_label(json_path))] + map(lambda x: self._to_string(self._extract(x, experimental_results_factory.create_from_json_file(json_path))), self.columns_to_render)

    def _extract(self, column_definition, exp_results):
        tokens, parameters = _parse_column_definition(column_definition)
        self._column_key = '-'.join(tokens)
        _ = COLUMNS_HASH[self._column_key]['extractor'](*list([exp_results] + parameters))
        print 'key', self._column_key, type(_), _,
        return _

    def _to_string(self, column_value):
        print 'str format', COLUMNS_HASH[self._column_key].get('to-string', '{}')
        _ = '-'
        if column_value is not None:
            _ = COLUMNS_HASH[self._column_key].get('to-string', '{}').format(column_value)
        self._max_col_lens[COLUMNS.index(self._column_key)] = max(self._max_col_lens[COLUMNS.index(self._column_key)], len(_))
        return str(_)
    # TODO after getting strings lists, sort according to fitness function and stringify appending suitable white spaces'
    # TODO according to elf._max_col_len which is computed when data pass through self._exp_result2_elements_list
    def _compute_strings_lists(self, sort_by_metric=False):
        if sort_by_metric:
            self._fitness_function = fitness_function_factory.get_single_metric_function(sort_by_metric)
            self.fitness_computer = FitnessCalculator(self._fitness_function)
            self._to_string_defs['fitness_value'] = self._to_string_defs[sort_function.single_metric]
            self._strings_lists = sorted(map(lambda x: self._exp_result2_elements_list(x), self.result_paths), key=lambda x: self.fitness_computer.compute_fitness(x), reverse=True)
            self._infos = sorted(map(lambda x: load_reportables(x), self.result_paths),
                                 key=lambda x: self.fitness_computer.compute_fitness(x, add_info=add_fitness_info), reverse=True)
            self._max_col_lens = list(map(lambda col_abrv:
                                          max(map(lambda reportable:
                                                  len(self._stringnify(reportable[self._transf[col_abrv]], col_abrv)), self._infos)), self._columns_titles))
        else:
            self._infos = map(lambda x: load_reportables(x), self.result_paths)
        self._insertions = [[] for _ in range(len(self._infos))]
        return map(lambda x: self._exp_result2_elements_list(x), self.result_paths)

    @staticmethod
    def _get_label(json_path):
        return re.search('/([\w\-.]+)\.json$', json_path).group(1)


    def _sort(self, columns):
        return reduce(lambda i,j: i+j, map(lambda x: sorted([_ for _ in columns if _.startswith(x)]), COLUMNS))

    def _determine_maximal_column_for_rendering(self, results_paths):
        return list(reduce(lambda i,j: i.union(j), map(lambda x: set(self._get_all_columns(experimental_results_factory.create_from_json_file(x))), results_paths)))

    def _get_all_columns(self, exp_results):
        return reduce(lambda i,j: i+j, map(lambda x: COLUMNS_HASH[x]['definitions'](exp_results) if x in ModelReporter.DYNAMIC_COLUMNS else [x], COLUMNS))

    @staticmethod
    def _get_invalid_columns(columns):
        return [_ for _ in columns if _ not in reduce(lambda i,j: i+j, map(lambda x: sorted([_ for _ in columns if _.startswith(x)]), COLUMNS))]

    def compute_info_list(self, sort_function=False, add_fitness_info=False):


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

class FitnessFunctionBuilder:
    def __init__(self):
        self._column_definitions = []
        self._extractors = []
        self._coeff_values = []
        self._order = ''
        self._names = []

    def _create_key_extractor(self, key):  # demek
        return lambda x: x[self._column_definitions.index(key)]

    def new(self, column_definitions, ordering='natural'):
        self._column_definitions = column_definitions
        self._extractors = []
        self._coeff_values = []
        self._order = ordering
        self._names = []
        return self
    def coefficient(self, name, value):
        self._extractors.append(self._create_key_extractor(name))
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


class InvalidColumnsException(Exception):
    def __init__(self, msg):
        super(InvalidColumnsException, self).__init__(msg)

def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on existing saved model topic model instances developed on the specified dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('--details', '-d', default=False, action='store_true', help='Switch to show details about the models')
    parser.add_argument('--sort', '-s', default='', help='Whether to sort the found experiments by checking the desired metric against the corresponding models')
    return parser.parse_args()

if __name__ == '__main__':

    cli_args = get_cli_arguments()

    res_ext = ResultsExtractor(['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity'])

    res_dir = os.path.join(COLLECTIONS_DIR, cli_args.dataset, RESULTS_DIR_NAME)
    # _columns_to_render = {column: [] for column in COLUMNS}
    # result_paths = glob.glob('{}/*.json'.format(res_dir))
    # exp_res = experimental_results_factory.create_from_json_file(result_paths[0])
    # out = res_ext.extract_all(exp_res)
    # print 'OUT', out

    reporter = ModelReporter()
    reporter.report(cli_args.dataset, details=None, sort=None, columns=None)


    # reporter = ModelReporter(cli_args.dataset)
    #
    # reporter.adw()

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
