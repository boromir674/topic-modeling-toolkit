import os
import re
import abc
import glob
import argparse
from collections import Counter
from patm.definitions import results_root, models_root, collections_dir
from patm.utils import load_results
from collections import defaultdict

def load_reportables1(results_path):
    d = load_results(results_path)
    return {'model_label': d['model_label'],
            'nb_topics': d['model_parameters']['nb_topics'][-1][1],
            'collection_passes': sum(d['collection_passes']),
            'document_passes': d['model_parameters']['document_passes'][-1][1],
            'reg_names': sorted(map(lambda x: x[1]['name'], d['reg_parameters'][-1][1].items())),
            'perplexity': float(d['trackables']['perplexity']['value'][-1]),
            'sparsity-theta': float(d['trackables']['sparsity-theta']['value'][-1]),
            'sparsity-phi': float(d['trackables']['sparsity-phi']['value'][-1]),
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
        self._transf = {'perplexity': 'prplx', 'kernel-coherence': 'coher', 'kernel-contrast': 'contr', 'kernel-purity': 'purity'}
        self._to_string_defs = {'perplexity': '{:.1f}', 'kernel-coherence': '{:.4f}', 'kernel-contrast': '{:.4f}', 'kernel-purity': '{:.4f}'}
        self.result_paths = glob.glob('{}/*-train.json'.format(results_dir))
        self._biggest_len = max(map(lambda x: len(os.path.basename(x).replace('-train.json', '')), self.result_paths))
        self._detail_switch = {False: lambda x: x['model_label'], True: self.to_string}
        self.fitness_computer = None
        self._infos = []
        self._insertions = []

        # stripped_phi_names = [os.path.basename(phi_path).replace('.phi', '') for phi_path in glob.glob('{}/*-train.phi'.format(models_dir))]
        # if stripped_phi_names != stripped_result_names:
        #     print '{} phi matrices do not correspond to resuls jsons'.format([_ for _ in stripped_phi_names if _ not in stripped_result_names], )
        #     print '{} results jsons do not correspond to phi matrices'.format([_ for _ in stripped_result_names if _ not in stripped_phi_names])

    def get_info_list(self, sort_function=False, add_fitness_info=False):
        if sort_function:
            self.fitness_computer = FitnessCalculator(sort_function)
            self._infos = sorted(map(lambda x: load_reportables1(x), self.result_paths), key=lambda x: self.fitness_computer.compute_fitness(x, add_info=add_fitness_info), reverse=True)
        else:
            self._infos = map(lambda x: load_reportables1(x), self.result_paths)
        self._insertions = [[] for _ in range(len(self._infos))]
        return self._infos

    def get_infos(self, details=False, sort_function=False, add_fitness_info=False):
        if sort_function:
            self.fitness_computer = FitnessCalculator(sort_function)
            return map(self._detail_switch[details], sorted(map(lambda x: load_reportables1(x), self.result_paths), key=lambda x: self.fitness_computer.compute_fitness(x, add_info=add_fitness_info), reverse=True))
        return map(self._detail_switch[details], map(lambda x: load_reportables1(x), self.result_paths))

    def to_string(self, reportables):
        r = reportables

        b = 'tpcs:{} col_iters:{} doc_iters:{} total:{}, '.format(r['nb_topics'], r['collection_passes'], r['document_passes'], r['collection_passes'] * r['document_passes'])
        # _ = 'prplx:{' + self._to_string_defs['perplexity'] + '}' + 'coher:{' + self._to_string_defs['kernel-coherence'] + ']' + 'contr:{' + self._to_string_defs['kernel-contrast'] + '}' + 'purity:{' + self._to_string_defs['kernel-purity'] + '}' + ' regs: [{}]'.format(', '.join(str(_) for _ in r['reg_names']))
        _ = 'prplx:' + self._to_string_defs['perplexity'] + ' coher:' + self._to_string_defs['kernel-coherence'] + ' contr:' + self._to_string_defs['kernel-contrast'] + ' purity:' + self._to_string_defs['kernel-purity'] + ' regs: [{}]'.format(', '.join(str(_) for _ in r['reg_names']))
        _ = _.format(r['perplexity'], r['kernel-coherence'], r['kernel-contrast'], r['kernel-purity'])
        b += _
        if 'fitness_function' and 'fitness_value' in r:
            b = '{}={:.4f} '.format(r['fitness_function'], r['fitness_value']) + b
        return r['model_label'] + ' '*(self._biggest_len - len(r['model_label'])) +': '+ b

    def get_highlighted_detailed_string(self, sorting_function, highlight_type):

        il = self.get_info_list(sort_function=sorting_function, add_fitness_info=True)
        b = ''
        best = self.fitness_computer.get_best()
        for k, v in best.items():
            self._get_match_indices(k, v)
        for z, repor in enumerate(il):
            sb = self.to_string(repor)
            accum = 0
            for ddef in self._insertions[z]:
                sb = self._insert_highlights(sb, ddef[0]+accum, ddef[1], highlight_type, self.ENDC)
                accum += len(highlight_type) + len(self.ENDC)
            b += '{}\n'.format(sb)
        return b

    def _get_match_indices(self, reportable_key, value):
        for j, i in enumerate(self._infos):
            m = re.match(r'[\w\+ :,\*=\.\[\]\n\-]+(' + self._transf[reportable_key] + ':' + self._to_string_defs[reportable_key].format(value).replace('.', '\.') + ')', self.to_string(i))
            if m:
                self._insertions[j].append((self.to_string(i).index(m.group(1)), len(m.group(1))))
        # return map(lambda x: (x[0], [x[1].string.index(x[1].group(1)), len(x[1].group(1))]) if x[1] else (x[1], []), map(lambda x: (self.to_string(x), re.match(r'[\w\+ :,\*=\.\[\]\n\-]+('+self._transf[reportable_key]+':'+self._to_string_defs[reportable_key].format(value).replace('.', '\.')+')', self.to_string(x))), self._infos))

    def _insert_highlights(self, b, start_index, length, first_insert, second_insert):

        return b[:start_index] + first_insert + b[start_index:start_index+length] + second_insert + b[start_index+length:]

def getg(key):
    return lambda x: x[key]

class FitnessFunction:
    def __init__(self, extractors, coefficients, names, ordering='natural'):
        self._extr = extractors
        self._coeff = coefficients
        self._names = names
        assert ordering in ('natural', 'reversed')
        self.order = ordering
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
        self._extractors.append(getg(name))
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

    def get_metric_function(self, name):
        return self._functions.get(name, fitness_function_builder.new(ordering=_orders[name]).coefficient(name, 1).build())

    def linear_combination_function(self, function_expression):
        pass
        # FUTURE WORK to support custom linnear combination of evaluation metrics

fitness_function_factory = FitnessFunctionFactory()

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

class FitnessCalculator(object):
    def __init__(self, calculating_function):
        self.func = calculating_function
        self._best = {'perplexity': float('inf'), 'kernel-coherence': 0, 'kernel-purity': 0, 'kernel-contrast': 0}

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

    def get_best(self):
        return self._best


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on existing saved model topic model instances developed on the specified dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('--details', '-d', default=False, action='store_true', help='Switch to show details about the models')
    parser.add_argument('--sort', '-s', default='', help='Whether to sort the found experiments by checking the desired metric against the corresponding models')
    # parser.add_argument('--sort-formula', '-sf', help='Whether to sort the found experiments by measuring the fitness of the models with the custom provided function')
    return parser.parse_args()

if __name__ == '__main__':
    collections_root_dir = '/data/thesis/data/collections'
    args = get_cli_arguments()
    if args.sort:
        fitness_function = fitness_function_factory.get_metric_function(args.sort)
    else:
        fitness_function = False
    model_reporter = ModelReporter(args.dataset)
    b = model_reporter.get_highlighted_detailed_string(fitness_function, model_reporter.UNDERLINE)
    print b
