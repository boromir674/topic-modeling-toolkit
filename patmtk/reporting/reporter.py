import re
import os
import sys
from glob import glob
from collections import Iterable

from .fitness import FitnessCalculator
from . import results_handler
from functools import reduce


class ModelReporter:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

    def __init__(self, collections_root_path, results_dir_name='results'):
        self._collections_dir = collections_root_path
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

    def get_formatted_string(self, collection_name, columns=None, metric='', verbose=True):
        """
        :param str collection_name:
        :param list columns:
        :param str metric:
        :param bool verbose:
        :return:
        :rtype: str
        """
        if verbose:
            print('REPORTING ON:')
        self._initialize(collection_name, columns=columns, metric=metric, verbose=verbose)
        body = '\n'.join(self._compute_rows())
        head = '{}{} {} {}'.format(' '*self._max_label_len,
                                   ' '*len(self._label_separator),
                                   ' '.join(['{}{}'.format(x[1], ' '*(self._max_col_lens[x[0]] - len(x[1]))) for x in enumerate(self._columns_titles[:-1])]),
                                   self._columns_titles[-1])
        return head + '\n' + body

    def _initialize(self, collection_name, columns=None, metric='', verbose=False):
        self._collection_name = collection_name
        self._result_paths = glob('{}/*.json'.format(os.path.join(self._collections_dir, collection_name, self._results_dir_name)))
        self._model_labels = [ModelReporter._get_label(x) for x in self._result_paths]
        if not self._model_labels:
            print("Either wrong dataset label '{}' was given or the collection/dataset has no trained models".format(collection_name))
            sys.exit(1)
        self._max_label_len = max([len(x) for x in self._model_labels])
        self._columns_to_render, self._columns_failed = [], []
        self._maximal_renderable_columns = self._get_maximal_renderable_columns()

        if not columns:
            self.columns_to_render = self._maximal_renderable_columns
        else:
            self.columns_to_render, self._columns_failed = ModelReporter._get_renderable(self._maximal_renderable_columns, columns)

        if metric and metric not in self.columns_to_render:
            raise InvalidMetricException("Metric '{}' is not recognized within [{}]".format(metric, ', '.join(self.columns_to_render)))
        self._metric = metric
        print("Input metric to sort on: '{}'".format(self._metric))
        if verbose:
            print('Using: [{}]'.format(', '.join(self.columns_to_render)))
            print('Ommiting: [{}]'.format(', '.join({_ for _ in self._maximal_renderable_columns if _ not in self.columns_to_render})))
            print('Failed: [{}]'.format(', '.join(self._columns_failed)))
        self._columns_titles = results_handler.get_titles(self.columns_to_render)
        self._max_col_lens = [len(x) for x in self._columns_titles]
        self.fitness_computer.highlightable_columns = [_ for _ in self.columns_to_render if ModelReporter._get_hash_key(_) in self._column_keys_to_highlight]

    @property
    def columns_to_render(self):
        return self._columns_to_render

    @columns_to_render.setter
    def columns_to_render(self, column_definitions):
        if not isinstance(column_definitions, Iterable):
            raise InvalidColumnsException(
                "Input column definitions are of type '{}' instead of iterable".format(type(column_definitions)))
        if not column_definitions:
            raise InvalidColumnsException('Input column definitions evaluates to None')
        invalid_columns = ModelReporter._get_invalid_column_definitions(column_definitions,
                                                                        self._maximal_renderable_columns)
        if invalid_columns:
            raise InvalidColumnsException(
                'Input column definitions [{}] are not valid'.format(', '.join(invalid_columns)))
        self._columns_to_render = column_definitions

    ########## COLUMNS DEFINITIONS ##########
    def _get_maximal_renderable_columns(self):
        """Call this method to get a list of all the inferred columns allowed to render."""
        return ModelReporter._get_column_definitions(results_handler.DEFAULT_COLUMNS,
                                                     ModelReporter.determine_maximal_set_of_renderable_columns(results_handler.get_experimental_results(self._collection_name)))

    @staticmethod
    def determine_maximal_set_of_renderable_columns(exp_results_list):
        return reduce(lambda i, j: i.union(j),
                      [set(results_handler.get_all_columns(x, results_handler.DEFAULT_COLUMNS)) for x in
                       exp_results_list])

    def _compute_rows(self):
        self._model_labels, values_lists = self._get_labels_n_values()
        return [self._to_row(y[0], y[1]) for y in zip(self._model_labels, [self._to_list_of_strings(x) for x in values_lists])]

    def _get_labels_n_values(self):
        """Call this method to get a list of model labels and a list of lists of reportable values that correspond to each label
        Fitness_computer finds the maximum values per eligible column definition that need to be highlighted."""
        if self._metric:
            self.fitness_computer.__init__(single_metric=self._metric, column_definitions=self.columns_to_render)
            return [list(t) for t in zip(*sorted(zip(self._model_labels, self._results_value_vectors), key=lambda y: self.fitness_computer(y[1]), reverse=True))]
        return self._model_labels, [self.fitness_computer.pass_vector(x) for x in self._results_value_vectors]

    ########## STRING OPERATIONS ##########
    def _to_row(self, model_label, strings_list):
        return '{}{}{} {}'.format(model_label,
                                  ' '*(self._max_label_len-len(model_label)),
                                  self._label_separator,
                                  ' '.join(['{}{}'.format(x[1], ' '*(self._max_col_lens[x[0]] - self._length(x[1]))) for x in enumerate(strings_list)]))

    def _to_list_of_strings(self, values_list):
        return [self._to_string(x[0], x[1]) for x in zip(values_list, self.columns_to_render)]

    def _to_string(self, value, column_definition):
        _ = '-'
        if value is not None:
            _ = results_handler.stringnify(column_definition, value)
            self._max_col_lens[self.columns_to_render.index(column_definition)] = max(self._max_col_lens[self.columns_to_render.index(column_definition)], len(_))
        if column_definition in self.fitness_computer.best and value == self.fitness_computer.best[column_definition]:
            return self.highlight_pre_fix + _ + self.highlight_post_fix
        return _

    def _length(self, a_string):
        _ = re.search(r'm(\d+(?:\.\d+)?)', a_string)  # !! this regex assumes that the string represents a number (eg will fail if input belongs to 'regularizers' column)
        if _: # if string is wrapped arround rendering decorators
            return len(_.group(1))
        return len(a_string)

    ########## EXTRACTION ##########
    @property
    def _results_value_vectors(self):
        return [self._extract_all(x) for x in results_handler.get_experimental_results(self._collection_name, selection='all')]

    def _extract_all(self, exp_results):  # get a list (vector) of extracted values; it shall contain integers, floats, Nones
        # (for metrics not tracked for the specific model) a single string for representing the regularization specifications and nan for the 'sparsity-phi-i' metric
        return [results_handler.extract(exp_results, x, 'last') for x in self.columns_to_render]

    ########## STATIC ##########
    @staticmethod
    def _get_renderable(allowed_renderable, columns):
        """
        Call this method to get the list of valid renderable columns and the list of invalid ones. The renderable columns
        are inferred from the selected 'columns' and the 'allowed' ones.\n
        :param list allowed_renderable:
        :param list columns:
        :return: 1st list: renderable inferred, 2nd list list: invalid requested columns to render
        :rtype: tuple of two lists
        """
        return [[_f for _f in reduce(lambda i,j: i+j, x) if _f]
                for x in zip(*[list(z)
                               for z in [ModelReporter._build_renderable(y, allowed_renderable)
                                         for y in columns]])]

    @staticmethod
    def _build_renderable(requested_column, allowed_renderable):
        if requested_column in allowed_renderable:
            return [requested_column], [None]
        elif requested_column in results_handler.DYNAMIC_COLUMNS:
            return sorted([_ for _ in allowed_renderable if _.startswith(requested_column)]), [None]
        elif requested_column in results_handler.DEFAULT_COLUMNS:  # if c is one of the columns that map to exactly one column to render; ie 'perplexity'
            return [requested_column], [None]
        else: # requested column is invalid: is not on of the allowed renderable columns
            return [None], [requested_column]

    @staticmethod
    def _get_column_definitions(columns, column_definitions):
        """Given a list of allowed column definitions, returns a sublist of it based on the selected columns. The returned list is ordered
         based on the given columns."""
        return reduce(lambda i,j: i+j, [sorted([_ for _ in column_definitions if _.startswith(x)]) for x in columns])

    @staticmethod
    def _get_invalid_column_definitions(column_defs, allowed_renderable):
        return [_ for _ in column_defs if _ not in allowed_renderable]

    @staticmethod
    def _get_label(json_path):
        try:
            return re.search('/([\w\-\.\+@]+)\.json$', json_path).group(1)
        except AttributeError as e:
            print('PATH', json_path)
            raise e

    @staticmethod
    def _get_hash_key(column_definition):
        return '-'.join([_ for _ in column_definition.split('-') if ModelReporter._is_token(_)])

    @staticmethod
    def _parse_column_definition(definition):
        return [list([_f for _f in y if _f]) for y in zip(*[(x, None) if ModelReporter._is_token(x) else (None, x) for x in definition.split('-')])]

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
