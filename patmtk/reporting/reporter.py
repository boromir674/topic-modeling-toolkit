import re
import os
from glob import glob
from collections import Iterable

from . import fitness_factory
from fitness import FitnessCalculator
from patm.modeling.experimental_results import experimental_results_factory


KERNEL_SUB_ENTITIES = ('coherence', 'contrast', 'purity')

def get_kernel_sub_hash(entity):
    assert entity in KERNEL_SUB_ENTITIES
    return {'kernel-'+entity: {'extractor': lambda x,y: getattr(x.tracked, 'kernel'+y[2:]).average.last if hasattr(x.tracked, 'kernel'+y[2:]) else None,
                               'column-title': lambda x: 'k'+entity[:3:2]+'.'+str(x)[2:],
                               'to-string': '{:.4f}',
                               'definitions': lambda x: map(lambda z: 'kernel-{}-{:.2f}'.format(entity, z), x.tracked.kernel_thresholds)}}

COLUMNS_HASH = {
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
                  'to-string': '{:.1f}',
                   'min-value': float('inf')},
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


class ModelReporter(object):
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    lb = len(r'\033[1m')
    lu = len(r'\033[4m')
    le = len(r'\033[0m')
    DYNAMIC_COLUMNS = ['kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi', 'background-tokens-ratio']
    DEFAULT_COLUMNS = ['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity'] + DYNAMIC_COLUMNS[:-1] + ['sparsity-theta'] + [DYNAMIC_COLUMNS[-1]] + ['regularizers']

    def __init__(self, collections_dir_path, results_dir_name='results'):
        self._collections_dir = collections_dir_path
        self._results_dir_name = results_dir_name
        self._label_separator = ':'
        self._columns_to_render = []
        self._column_keys_to_highlight = ['perplexity', 'kernel-coherence', 'kernel-purity', 'kernel-contrast', 'top-tokens-coherence',
                                          'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio']
        self.highlight_pre_fix = ModelReporter.UNDERLINE
        self.highlight_post_fix = ModelReporter.ENDC
        self.fitness_computer = FitnessCalculator()
        self._max_label_len = 0
        self._max_col_lens = []

    def _initialize(self, collection_name, columns=None, column_definitions=None, metric=''):
        self._columns_to_render = []
        self._result_paths = glob('{}/*.json'.format(os.path.join(self._collections_dir, collection_name, self._results_dir_name)))
        self._model_labels = map(lambda x: ModelReporter._get_label(x), self._result_paths)
        self._max_label_len = max(map(lambda x: len(x), self._model_labels))
        if column_definitions is not None:
            try:
                self.columns_to_render = column_definitions
            except InvalidColumnsException as e:
                print e
        if not self.columns_to_render:
            if columns is not None:
                self.columns_to_render = ModelReporter._get_column_definitions(columns, self._determine_maximal_column_for_rendering(self._result_paths))
            else:
                print 'Automatically using all possible columns inferred from the experimental results to report'
                self.columns_to_render = ModelReporter._get_column_definitions(ModelReporter.DEFAULT_COLUMNS, self._determine_maximal_column_for_rendering(self._result_paths))
        if metric and metric not in self.columns_to_render:
            raise InvalidMetricException("Metric '{}' is not recognized within [{}]".format(metric, ', '.join(self.columns_to_render)))
        self._metric = metric
        self._columns_titles = map(lambda z: COLUMNS_HASH['-'.join(z[0])]['column-title'](*z[1]), map(lambda x: ModelReporter._parse_column_definition(x), self.columns_to_render))
        self._max_col_lens = map(lambda x: len(x), self._columns_titles)
        print 'max col lens', self._max_col_lens
        self.fitness_computer.highlightable_columns = [_ for _ in self.columns_to_render if ModelReporter._get_hash_key(_) in self._column_keys_to_highlight]

    @property
    def columns_to_render(self):
        return self._columns_to_render

    @columns_to_render.setter
    def columns_to_render(self, column_definitions):
        if not isinstance(column_definitions, Iterable):
            raise InvalidColumnsException("Input column definitions are of type '{}' instead of iterable".format(type(column_definitions)))
        invalid_columns = ModelReporter._get_invalid_column_definitions(column_definitions)
        if invalid_columns:
            raise InvalidColumnsException('Input column definitions [{}] are not valid'.format(', '.join(invalid_columns)))
        self._columns_to_render = column_definitions

    def get_formatted_string(self, collection_name, columns=None, columns_definitions=None, metric=''):
        """
        :param str collection_name:
        :param list columns:
        :param list columns_definitions:
        :param str metric:
        :return:
        :rtype: str
        """
        self._initialize(collection_name, columns=columns, column_definitions=columns_definitions, metric=metric)
        print 'To render:'
        print '[{}]'.format(', '.join(self.columns_to_render))
        # print '[{}]'.format(', '.join(self._columns_titles))
        print 'max col lens', ' '.join(map(lambda x: '{}:{}'.format(x[0], x[1]), zip(self._columns_titles, self._max_col_lens)))
        body = '\n'.join(self._compute_rows(metric=metric))
        print 'max col lens', ' '.join(map(lambda x: '{}:{}'.format(x[0], x[1]), zip(self._columns_titles, self._max_col_lens)))
        head = '{}{} {} {}'.format(' '*self._max_label_len,
                                   ' '*len(self._label_separator),
                                   ' '.join(map(lambda x: '{}{}'.format(x[1], ' '*(self._max_col_lens[x[0]] - len(x[1]))), enumerate(self._columns_titles[:-1]))),
                                   self._columns_titles[-1])
        print 'max col lens', ' '.join(map(lambda x: '{}:{}'.format(x[0], x[1]), zip(self._columns_titles, self._max_col_lens)))
        return head + '\n' + body

    def _compute_rows(self, metric=''):
        self._model_labels, values_lists = self._get_labels_n_values(sort_by=metric)
        return map(lambda y: self._to_row(y[0], y[1]), zip(self._model_labels, map(lambda x: self._to_list_of_strings(x), values_lists)))

    def _get_labels_n_values(self, sort_by=''):
        """Call this method to get a list of model labels and a list of lists of reportable values that correspond to each label
        Fitness_computer finds the maximum values per eligible column definition that need to be highlighted."""
        if sort_by:
            self._fitness_function = fitness_factory.get_single_metric_function(self.columns_to_render, sort_by)
            self.fitness_computer.initialize(self._fitness_function, self.columns_to_render)
            return map(lambda t: list(t), zip(*sorted(zip(self._model_labels, self._get_values_lists()), key=lambda y: self.fitness_computer.compute_fitness(y[1]), reverse=True)))
        return self._model_labels, map(lambda x: self.fitness_computer.pass_vector(x), self._get_values_lists())

    ########## STRING OPERATIONS ##########
    def _to_row(self, model_label, strings_list):
        return '{}{}{} {}'.format(model_label,
                                  ' '*(self._max_label_len-len(model_label)),
                                  self._label_separator,
                                  ' '.join(map(lambda x: '{}{}'.format(x[1], ' '*(self._max_col_lens[x[0]] - self._length(x[1]))), enumerate(strings_list))))

    def _to_list_of_strings(self, values_list):
        return map(lambda x: self._to_string(x[0], x[1]), zip(values_list, self.columns_to_render))

    def _to_string(self, value, column_definition):
        _ = '-'
        if value is not None:
            _ = COLUMNS_HASH[ModelReporter._get_hash_key(column_definition)].get('to-string', '{}').format(value)
            self._max_col_lens[self.columns_to_render.index(column_definition)] = max(self._max_col_lens[self.columns_to_render.index(column_definition)], len(_))
        if column_definition in self.fitness_computer.best and value == self.fitness_computer.best[column_definition]:
            return self.highlight_pre_fix + _ + self.highlight_post_fix
        return _

    def _length(self, a_string):
        _ = re.search(r'm(\d+(?:\.\d+)?)', a_string)
        if _: # if string is wrapped arround rendering decorators
            l = len(_.group(1))
        else:
            l = len(a_string)
        return l

    ########## EXTRACTION ##########
    def _get_values_lists(self):
        return map(lambda x: self._extract_all(experimental_results_factory.create_from_json_file(x)), self._result_paths)
    def _extract_all(self, exp_results):
        return map(lambda x: self._extract(x, exp_results), self.columns_to_render)
    def _extract(self, column_definition, exp_results):
        tokens, parameters = ModelReporter._parse_column_definition(column_definition)
        return COLUMNS_HASH['-'.join(tokens)]['extractor'](*list([exp_results] + parameters))

    ########## COLUMNS DEFINITIONS ##########
    def _determine_maximal_column_for_rendering(self, results_paths):
        return list(reduce(lambda i,j: i.union(j), map(lambda x: set(self._get_all_columns(experimental_results_factory.create_from_json_file(x))), results_paths)))
    def _get_all_columns(self, exp_results):
        return reduce(lambda i,j: i+j, map(lambda x: COLUMNS_HASH[x]['definitions'](exp_results) if x in ModelReporter.DYNAMIC_COLUMNS else [x], ModelReporter.DEFAULT_COLUMNS))

    ########## STATIC ##########
    @staticmethod
    def _get_column_definitions(columns, column_definitions):
        if not isinstance(columns, Iterable):
            raise InvalidColumnsException("Input columns are of type '{}' instead of iterable".format(type(columns)))
        invalid_columns = [_ for _ in columns if _ not in ModelReporter.DEFAULT_COLUMNS]
        if invalid_columns:
            raise InvalidColumnsException('Input columns [{}] are not valid'.format(', '.join(invalid_columns)))
        return reduce(lambda i,j: i+j, map(lambda x: sorted([_ for _ in column_definitions if _.startswith(x)]), columns))

    @staticmethod
    def _get_invalid_column_definitions(column_defs):
        return [_ for _ in column_defs if _ not in ModelReporter._get_column_definitions(ModelReporter.DEFAULT_COLUMNS, column_defs)]

    @staticmethod
    def _get_label(json_path):
        return re.search('/([\w\-.]+)\.json$', json_path).group(1)

    @staticmethod
    def _get_hash_key(column_definition):
        return '-'.join([_ for _ in column_definition.split('-') if ModelReporter._is_token(_)])

    @staticmethod
    def _parse_column_definition(definition):
        return map(lambda y: list(filter(None, y)),
                   zip(*map(lambda x: (x, None) if ModelReporter._is_token(x) else (None, x), definition.split('-'))))

    @staticmethod
    def _is_token(definition_element):
        try:
            _ = float(definition_element)
            return False
        except ValueError:
            if definition_element[0] == '@' or len(definition_element) == 1:
                return False
            return True


class InvalidColumnsException(Exception):
    def __init__(self, msg):
        super(InvalidColumnsException, self).__init__(msg)

class InvalidMetricException(Exception):
    def __init__(self, msg):
        super(InvalidMetricException, self).__init__(msg)
