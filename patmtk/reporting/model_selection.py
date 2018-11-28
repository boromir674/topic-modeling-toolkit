import os
import re
from glob import glob
from collections import Counter

from patm.modeling import ExperimentalResults
from .fitness import FitnessFunction


KERNEL_SUB_ENTITIES = ('coherence', 'contrast', 'purity')

def _get_kernel_sub_hash(entity):
    assert entity in KERNEL_SUB_ENTITIES
    return {'kernel-'+entity: {'scalar-extractor': lambda x,y: getattr(x.tracked, 'kernel'+y[2:]).average.last if hasattr(x.tracked, 'kernel'+y[2:]) else None,
                               'list-extractor': lambda x, y: getattr(x.tracked, 'kernel' + y[2:]).average.all if hasattr(x.tracked, 'kernel' + y[2:]) else None,
                               'column-title': lambda x: 'k'+entity[:3:2]+'.'+str(x)[2:],
                               'to-string': '{:.4f}',
                               'definitions': lambda x: map(lambda z: 'kernel-{}-{:.2f}'.format(entity, z), x.tracked.kernel_thresholds)}}

COLUMNS_HASH = {
    'nb-topics': {'scalar-extractor': lambda x: x.scalars.nb_topics,
                  'column-title': lambda: 'tpcs'},
    'collection-passes': {'scalar-extractor': lambda x: x.scalars.dataset_iterations,
                          'column-title': lambda: 'col-i'},
    'document-passes': {'scalar-extractor': lambda x: x.scalars.document_passes,
                        'column-title': lambda: 'doc-i'},
    'total-phi-updates': {'scalar-extractor': lambda x: x.scalars.dataset_iterations * x.scalars.document_passes,
                          'column-title': lambda: 'phi-u'},
    'perplexity': {'scalar-extractor': lambda x: x.tracked.perplexity.last,
                   'list-extractor': lambda x: x.tracked.perplexity.all,
                   'column-title': lambda: 'prpl',
                   'to-string': '{:.1f}',
                   'min-value': float('inf')},
    'top-tokens-coherence': {'scalar-extractor': lambda x,y: getattr(x.tracked, 'top'+str(y)).average_coherence.last if hasattr(x.tracked, 'top'+str(y)) else None,
                             'list-extractor': lambda x,y: getattr(x.tracked, 'top'+str(y)).average_coherence.all if hasattr(x.tracked, 'top'+str(y)) else None,
                             'column-title': lambda x: 'top'+str(x)+'ch',
                             'to-string': '{:.4f}',
                             'definitions': lambda x: map(lambda y: 'top-tokens-coherence-'+str(y), x.tracked.top_tokens_cardinalities)},
    'sparsity-phi': {'scalar-extractor': lambda x,y: getattr(x.tracked, 'sparsity_phi_'+y).last if hasattr(x.tracked, 'sparsity_phi_'+y) else None,
                     'list-extractor': lambda x,y: getattr(x.tracked, 'sparsity_phi_'+y).all if hasattr(x.tracked, 'sparsity_phi_'+y) else None,
                     'column-title': lambda y: 'spp@'+y,
                     'to-string': '{:.2f}',
                     'definitions': lambda x: map(lambda z: 'sparsity-phi-{}'.format(z), x.tracked.modalities_initials)},
    'sparsity-theta': {'scalar-extractor': lambda x: x.tracked.sparsity_theta.last,
                       'list-extractor': lambda x: x.tracked.sparsity_theta.all,
                       'column-title': lambda: 'spt',
                       'to-string': '{:.2f}'},
    'background-tokens-ratio': {'scalar-extractor': lambda x,y: getattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]).last if hasattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]) else None,
                                'list-extractor': lambda x,y: getattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]).all if hasattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]) else None,
                                'column-title': lambda x: 'btr.'+str(x)[2:],
                                'to-string': '{:.2f}',
                                'definitions': lambda x: map(lambda y: 'background-tokens-ratio-{:.2f}'.format(y), x.tracked.background_tokens_thresholds)},
    'regularizers': {'scalar-extractor': lambda x: '[{}]'.format(', '.join(x.regularizers)),
                     'column-title': lambda: 'regs'}
}

COLUMNS_HASH = reduce(lambda x, y: dict(y, **x), [COLUMNS_HASH] + map(lambda z: _get_kernel_sub_hash(z), KERNEL_SUB_ENTITIES))


class ResultsHandler(object):
    _QUANTITY_2_EXTRACTOR = {'last': 'scalar', 'all': 'list'}
    DYNAMIC_COLUMNS = ['kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi', 'background-tokens-ratio']
    DEFAULT_COLUMNS = ['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity'] +\
                      DYNAMIC_COLUMNS[:-1] + ['sparsity-theta'] + [DYNAMIC_COLUMNS[-1]] + ['regularizers']

    def __init__(self, collection_root_path, results_dir_name='results'):
        self._collection_root_path = collection_root_path
        self._results_dir_name = results_dir_name
        self._results_hash = {}
        self._collection = ''
        # self._fitness_function_builder = FitnessFunctionBuilder()
        self._fitness_function_hash = {}

    def get_experimental_results(self, collection_name, top='all', sort=''):
        """
        Call this method to get a list of experimental result objects from topic models trained on the given collection.
        :param str collection_name:
        :param str or int top:
        :param str sort:
        :return:
        """
        self._collection = collection_name
        result_paths = glob('{}/*.json'.format(os.path.join(self._collection_root_path, collection_name, self._results_dir_name)))
        return self._get_experimental_results(result_paths, top=top, callable_metric=self._get_metric(sort))

    def _get_experimental_results(self, results_paths, top='all', callable_metric=None):
        if top == 'all':
            top = len(results_paths)
        assert type(top) == int and top > 0
        if callable_metric:
            return sorted(map(lambda x: self._process_result_path(x), results_paths), key=callable_metric, reverse=True)[:top]
        return map(lambda x: self._process_result_path(x), results_paths)[:top]

    def _process_result_path(self, result_path):
        if result_path not in self._results_hash:
            self._results_hash[result_path] = ExperimentalResults.create_from_json_file(result_path)
        return self._results_hash[result_path]

    def _get_metric(self, metric):
        if not metric:
            return None
        if metric not in self._fitness_function_hash:
            self._fitness_function_hash[metric] = FitnessFunction.single_metric(metric)
        return lambda x: self._fitness_function_hash[metric].compute([ResultsHandler.extract(x, metric, 'last')])

    @staticmethod
    def get_titles(column_definitions):
        return map(lambda x: ResultsHandler.get_abbreviation(x), column_definitions)

    @staticmethod
    def get_abbreviation(definition):
        tokens, parameters = ResultsHandler._parse_column_definition(definition)
        return COLUMNS_HASH['-'.join(tokens)]['column-title'](*parameters)

    @staticmethod
    def stringnify(column, value):
        """
        :param str column: key or definition; example values: 'perplexity', 'kernel-coherence', 'kernel-coherence-0.80'
        :param value:
        :return:
        :rtype str
        """
        return COLUMNS_HASH.get(column, COLUMNS_HASH[ResultsHandler._get_hash_key(column)]).get('to-string', '{}').format(value)

    @staticmethod
    def get_tau_trajectory(exp_results, matrix_name):
        """

        :param patm.modeling.experimental_results.ExperimentalResults exp_results:
        :param matrix_name:
        :return:
        """
        return getattr(exp_results.tracked.tau_trajectories, matrix_name).all

    @staticmethod
    def extract(exp_results, column_definition, quantity):
        """
        Call this method to query the given experimental results object about a specific metric. Supports requesting all
        values tracked along the training process.
        :param patm.modeling.experimental_results.ExperimentalResults exp_results:
        :param str column_definition:
        :param str quantity: must be one of {'last', 'all'}
        :return:
        """
        tokens, parameters = ResultsHandler._parse_column_definition(column_definition)
        return COLUMNS_HASH['-'.join(tokens)][ResultsHandler._QUANTITY_2_EXTRACTOR[quantity]+'-extractor'](*list([exp_results] + parameters))

    @staticmethod
    def get_all_columns(exp_results):
        return reduce(lambda i, j: i + j, map(lambda x: COLUMNS_HASH[x]['definitions'](exp_results) if x in ResultsHandler.DYNAMIC_COLUMNS else [x], ResultsHandler.DEFAULT_COLUMNS))

    ###### UTILITY FUNCTIONS ######
    @staticmethod
    def _parse_column_definition(definition):
        return map(lambda y: list(filter(None, y)),
                   zip(*map(lambda x: (x, None) if ResultsHandler._is_token(x) else (None, x), definition.split('-'))))

    @staticmethod
    def _get_hash_key(column_definition):
        return '-'.join([_ for _ in column_definition.split('-') if ResultsHandler._is_token(_)])

    @staticmethod
    def _is_token(definition_element):
        try:
            _ = float(definition_element)
            return False
        except ValueError:
            if definition_element[0] == '@' or len(definition_element) == 1:
                return False
            return True

    @staticmethod
    def determine_metrics_usable_for_comparison(exp_results_list):
        c = Counter()
        c.update(map(lambda i,j: i+j, map(lambda x: ResultsHandler.get_all_columns(x), exp_results_list)))
        return [k for k, v in c.items() if v > 1]

    @staticmethod
    def determine_maximal_set_of_renderable_columns(exp_results_list):
        return reduce(lambda i, j: i.union(j), map(lambda x: set(ResultsHandler.get_all_columns(x)), exp_results_list))

# class ModelSelector(object):
#
#     def __init__(self, collection_results_dir_path=''):
#         self._working_dir = ''
#         self._results_paths, self._available_model_labels = [], []
#
#         if collection_results_dir_path:
#             self.working_dir = collection_results_dir_path
#
#     @property
#     def working_dir(self):
#         return self._working_dir
#
#     @working_dir.setter
#     def working_dir(self, results_dir_path):
#         self._working_dir = results_dir_path
#         self._results_paths = glob('{}/*.json'.format(self._working_dir))
#         if not self._results_paths:
#             raise NoTrainingResultsFoundException("No saved training results found in '{}' directory.".format(collection_results_dir_path))
#         self._available_model_labels = list(map(lambda x: re.search('/([\w\-.]+)\.json$', x).group(1), self._results_paths))
#
#     def select_n_get(self, expression='all'):
#         if expression == 'all':
#             return self._available_model_labels
#         return []

# class NoTrainingResultsFoundException(Exception):
#     def __init__(self, msg):
#         super(NoTrainingResultsFoundException, self).__init__(msg)


if __name__ == '__main__':
    ms = ModelSelector(collection_results_dir_path='/data/thesis/data/collections/dd/results')
