import re
import os
from math import ceil
from functools import reduce

from operator import itemgetter
import warnings

from topic_modeling_toolkit.results import ExperimentalResults


def parse_sort(sort_def):
    _ = re.search(r'^([a-zA-Z]+)-?(?:(0\.)?(\d\d?))?', sort_def).groups()
    return {'type': _[0], 'threshold': _threshold(_[1], _[2])}

def parse_tokens_type(tokens_type_def):
    _ = re.search(r'^((?:[a-zA-Z\-]+)*(?:[a-zA-Z]+))-?(?:(0\.)?(\d\d?))?', tokens_type_def).groups()
    return {'type': _[0], 'threshold': _threshold(_[1], _[2])}


def _threshold(el1, el2):
    if el1 is None and el2 is None:
        return None
    if len(el2) == 2:
        return '0.' + el2
    return '0.' + el2 + '0'

class Metric(object):
    def __new__(cls, *args, **kwargs):
        x = super(Metric, cls).__new__(cls)
        x._attr_getter = None
        x._reverse = True
        return x
    def __call__(self, *args, **kwargs):
        return self._attr_getter(args[0])
    def sort(self, topics):
        return sorted(topics, key=self._attr_getter, reverse=self._reverse)
class AlphabeticalOrder(Metric):
    def __init__(self):
        self._attr_getter = lambda x: x.name
        self._reverse = False

class KernelMetric(Metric):
    def __init__(self, threshold, metric_attribute, str_format='{:.1f}'):
        assert 0 < threshold < 1
        self._th, self._kernel_metric_attribute, self._frt = threshold, metric_attribute, str_format
        self._attr_getter = lambda x: getattr(getattr(x, 'kernel{}'.format(self._threshold(str(self._th)[2:4]))), self._kernel_metric_attribute)
    def pformat(self, topic):
        return '{}%'.format(self._frt.format(self(topic)*100))

    @staticmethod
    def _threshold(threshold):
        if len(threshold) == 1:
            return threshold + '0'
        return threshold

class MetricsContainer(object):
    abbrvs = {'coh': 'coherence', 'con': 'contrast', 'pur': 'purity'}
    def __init__(self):
        self._ab = AlphabeticalOrder()
    def __iter__(self):
        return iter(('coherence', 'contrast', 'purity'))
    def __call__(self, *args, **kwargs):
        return getattr(self, self.abbrvs.get(args[0], args[0]))(*args[1:])
    def name(self, *args):
        return self._ab
    def coherence(self, threshold):
        return KernelMetric(threshold, 'coherence')
    def contrast(self, threshold):
        return KernelMetric(threshold, 'contrast')
    def purity(self, threshold):
        return KernelMetric(threshold, 'purity')
    def __contains__(self, item):
        return item in iter(self) or item == 'name' or item in ['coh', 'con', 'pur']

metrics_container = MetricsContainer()


class TopicsHandler(object):
    sep = ' |'
    headers_sep = ' ' * len(sep)
    token_extractor = {'top-tokens': lambda x: x[0].top_tokens, # a topic and any other parameter that is not used
                       'kernel': lambda x: getattr(x[0], 'kernel'+x[1][2:]).tokens} # assumes topic_obj and threshold ie '0.90' tuple
    length_extractor = {'top-tokens': lambda x: len(x[0].top_tokens),
                        'kernel': lambda x: len(getattr(x[0], 'kernel'+ x[1][2:]))}  # assumes topic_obj and threshold ie '0.90' tuple
    metrics = metrics_container

    def __init__(self, collections_root_dir, results_dir_name='results'):
        self._cols = collections_root_dir
        self._res_dir_name = results_dir_name
        self._res, self._top_tokens_def = None, ''
        self._warn = []
        self._groups_type = ''
        self._model_label, self._threshold = '', ''
        # self._path2res = {}
        self._max_token_length_per_column = []
        self._model_topics_hash = {}
        self._max_tokens_per_row = []
        self._row_count = 0
        self._line_count = 0
        self._columns = 5
        self._topic_headers = []
        self._max_headers = []

    def _result_path(self, *args):
        """If 1 argument is given it is assumed to be a full path to results json file. If 2 arguments are given it is
        assumed that the 1st is a dataset label and the second a model label (eg 'plsa_1_3' not 'plsa_1_3.json')"""
        if os.path.isfile(args[0]):
            self._model_label = os.path.basename(args[0]).replace('.json', '')
            return args[0]
        self._model_label = args[1]
        return os.path.join(self._cols, args[0], 'results', '{}.json'.format(args[1]))

    def _model_topics(self, results_path):
        if results_path not in self._model_topics_hash:
            self._model_topics_hash[results_path] = ModelTopics(ExperimentalResults.create_from_json_file(results_path))
        return self._model_topics_hash[results_path]

    def pformat_background(self, model_results_path, columns=6, nb_tokens=100, show_title=False):
        """
        :param list model_results_path: if one element it should be a full path to a json file holding experimental results. If 2 elements, 1st i a dataset label and 2nd a model label (ie 'clda_1_2')
        :param int columns:
        :param int nb_tokens:
        :param bool show_title:
        :return:
        :rtype: str
        """
        _ = ExperimentalResults.create_from_json_file(self._result_path(*model_results_path))
        all_bg_tokens = _.final.background_tokens
        return self._pformat2(all_bg_tokens[:nb_tokens], columns=columns, title=(lambda x: _.scalars.model_label+' background tokens\n\n' if x else '')(show_title))

    @classmethod
    def _pformat2(cls, elements_list, columns=8, lines=20, title=''):
        columns = min(columns, int(ceil(len(elements_list) / lines)))
        max_token_p_col = [max(map(len, cls.column_slice(c, elements_list, lines, columns))) for c in range(columns)]
        b = title
        for r in range(int(ceil(len(elements_list) / (columns*lines)))):
            for l in range(lines):
                b += '{}\n'.format(cls.sep.join('{}{}'.format(tok_el, (max_token_p_col[i] - len(tok_el))*' ')
                             for i, tok_el in enumerate(cls.line_slice(r, l, elements_list, lines, columns))))
            b += '\n'
        return b

    @staticmethod
    def column_slice(column_index, elements_list, nb_lines, nb_columns):
        ent = range(int(ceil(len(elements_list) / (nb_columns*nb_lines))))  # number of "rows" each row has 'nb_lines' lines
        _ = [elements_list[nb_lines*(r*nb_columns + column_index):nb_lines*(r*nb_columns + column_index + 1)] for r in ent]
        return reduce(lambda i,j: i+j, _)

    @staticmethod
    def line_slice(r_index, l_index, all_bg_tokens, lines, columns):
        return all_bg_tokens[(r_index*lines*columns)+l_index:(r_index+1)*lines*columns:lines]

    def pformat(self, model_results_path, topics_set, tokens_type, sort, nb_tokens, columns, trim=0, topic_info=True, show_title=False):
        """
        :param list model_results_path:
        :param str topics_set:
        :param str tokens_type: accepts 'top-tokens' or 'kernel-|kernel' plus a threshold like '0.80', '0.6', '0.1234', '75', '5', '4321'
        :param str sort: {'name' (alphabetical), 'coherence-th', 'contrast-th', 'purity-th'} th i a \d\d pattern corresponding to kernel threshold
        :param int nb_tokens:
        :param int columns:
        :param bool show_title:
        :return:
        :rtype: str
        """
        self._columns = columns
        tokens_info = parse_tokens_type(tokens_type)
        sort_info = parse_sort(sort)
        assert tokens_info['type'] in ('top-tokens', 'kernel')
        if sort_info['type'] not in self.metrics:
            raise RuntimeError("Metric '{}' not supported. Use from [{}]".format(sort_info['type'], ', '.join("'{}'".format(x) for x in self.metrics)))
        if sort_info['threshold'] is None:
            if sort_info['type'] == 'name':
                if tokens_info['type'] == 'kernel' and tokens_info['threshold'] is None:
                    raise RuntimeError("Requested to show the kernel tokens per topic, but no threshold information was found in either the 'tokens_type' or the 'sort' parameters, which is required to target a specific kernel.")
            elif tokens_info['threshold'] is None:
                raise RuntimeError("Requested to sort topics according to a metric but no thresold was found in either the 'tokens_type' or the 'sort' parameters, which is required to target a specific kernel.")
            elif tokens_info['type'] == 'top-tokens':
                raise RuntimeError("Requested to sort topics according to a metric but no thresold was found in either the 'tokens_type' or the 'sort' parameters, which is required to target a specific kernel.")
        elif tokens_info['type'] == 'kernel' and tokens_info['threshold'] is not None and tokens_info['threshold'] != sort_info['threshold']:
            raise RuntimeError("The input token-type and sort-metric thresholds, {} and {} respectively, differ".format(tokens_info['threshold'], sort_info['threshold']))

        self._threshold = (lambda x: tokens_info['threshold'] if x is None else x)(sort_info['threshold'])

        model_topics = self._model_topics(self._result_path(*model_results_path))
        assert topics_set == 'domain'
        topics_set = getattr(model_topics, topics_set)  # either domain or background topics
        # self._available_thresolds = topics_set.thresholds

        th = self._threshold
        if th is not None:
            th = float(self._threshold)
        callable_metric = self.metrics(sort_info['type'], th)
        topics = callable_metric.sort(list(topics_set))
        if len(topics) < columns:
            self._columns = len(topics)

        self._max_token_length_per_column = [max([max(len(topic.name),
                                                      max(map(len,
                                                              list(self.token_extractor[tokens_info['type']]([topic, self._threshold]))[:nb_tokens or None])))
                                                  for topic in topics[i::self._columns]]) for i in range(self._columns)]
        self._topic_headers = list(self._gen_headers(topics, prefix='t', topic_info=topic_info))
        assert len(self._topic_headers) == len(topics)
        self._max_headers = [max(max(len(_) for _ in thd) for thd in self._topic_headers[c::self._columns]) for c in range(self._columns)]

        self._max_token_length_per_column = [max(x[0], x[1]) for x in zip(self._max_token_length_per_column, self._max_headers)]
        self._max_tokens_per_row = [max(self.length_extractor[tokens_info['type']]([t, self._threshold])
                                        for t in topics[k*self._columns:(k+1)*self._columns]) for k in range(int(ceil(len(topics) / self._columns)))]
        return self._pformat(topics, tokens_info['type'], nb_tokens, title=(lambda x: 'model: {}, tokens:{}, sort:{}\nrespective metrics: coherence, contrast, purity'.format(self._model_label, tokens_type, sort) + '\n\n' if x else '')(show_title))

    def _pformat(self, topics, tokens_type, nb_tokens, title=''):
        b = title
        for r in range(int(ceil(len(topics) / self._columns))):
            row_topics = topics[r*self._columns:(r+1)*self._columns]
            b += self._build_headers(r*self._columns, (r+1)*self._columns) + \
                 '\n'.join(self._line(l, tokens_type, row_topics) for l in range(min(self._max_tokens_per_row[r], nb_tokens))) + '\n'
            b += '\n'
        return b

    def _gen_headers(self, topics, prefix='t', topic_info=True):
        it = iter([prefix + re.search('^\w+(\d{2,3})$', t.name).group(1)] for t in topics)
        if topic_info and self._threshold is not None: # strictly must check against None
                return iter([next(it)[0], self._topic_metrics_header(t, float(self._threshold))] for t in topics)
        return it

    def _build_headers(self, start, stop):
        it = iter([r +' ' * (self._max_token_length_per_column[i] - len(r)) for r in item] for i, item in enumerate(self._topic_headers[slice(start, stop)]))
        e = reduce(lambda i, j: [ie + self.headers_sep + je for ie, je in zip(i, j)], it)
        return '{}\n'.format('\n'.join(_ for _ in e))

    @classmethod
    def _topic_metrics_header(cls, topic, threshold):
        """
        :param float threshold:
        """
        return ' '.join(getattr(cls.metrics, m)(threshold).pformat(topic) for m in cls.metrics)

    def _line(self, l_index, tokens_type, topics):
        return self.sep.join('{}{}'.format(tok_el, (self._max_token_length_per_column[i] - len(tok_el))*' ')
                             for i, tok_el in enumerate(self._gen_line_elements(l_index, tokens_type, topics)))

    def _gen_line_elements(self, l_index, tokens_type, topics):
        return iter(self._token(t, tokens_type, l_index) for t in topics)

    def _token(self, topic, tokens_type, l_index):
        if len(self._get_tokens_list(topic, tokens_type, threshold=self._threshold)) <= l_index:
            return ''
        return self._get_tokens_list(topic, tokens_type, threshold=self._threshold)[l_index]

    @classmethod
    def _get_tokens_list(cls, topic, tokens_type, threshold=''):
        return list(cls.token_extractor[tokens_type]([topic, threshold]))


class ModelTopics(object):

    @property
    def domain(self):
        return self._domain_topic_set
    @property
    def background_tokens(self):
        return self._background_tokens

    def __init__(self, experimental_results):
        """

        :param results.experimental_results.ExperimentalResults experimental_results:
        """
        max_top_tokens = max([int(_.split('-')[-1]) for _ in experimental_results.final.top_defs])
        if max_top_tokens < 10:
            warnings.warn("You probably wouldn't want to track less than the 10 most probable [p(w|t)] tokens per inferred topic. "
                          "This is achieved by enabling a score tracker: add a line such as 'top-tokens-10 = tops10' or "
                          "'top-tokens-100 = t100' under the [scores] section.")
        self._top_tokens_def = 'top{}'.format(max_top_tokens)

        self._domain_topic_set = self._create_topics_set('domain', experimental_results)

        self._background_tokens = experimental_results.final.background_tokens

    def _create_topics_set(self, topic_set_name, exp_res):
        topic_names_list = getattr(exp_res.scalars, '{}_topics'.format(topic_set_name))
        return TopicsSet(topic_set_name, [
            Topic(tn,
                  [{'threshold': float(th),
                    'tokens': self._get_tokens(getattr(exp_res.final, 'kernel{}'.format(th[2:])), tn),
                    'metrics': {m: getattr(getattr(getattr(exp_res.tracked, 'kernel{}'.format(th[2:])), tn), m).last for m in metrics_container}}
                    # 'metrics': {m: getattr(getattr(getattr(exp_res.tracked, 'kernel{}'.format(th)), tn), m).last for m in metrics_container}}
                   for th in exp_res.tracked.kernel_thresholds],
                  self._get_tokens(getattr(exp_res.final, self._top_tokens_def), tn))
            for tn in topic_names_list])

    def _get_tokens(self, final_obj, topic_name):
        return getattr(final_obj, topic_name).tokens


class TopicsSet(object):
    def __init__(self, name, topics):
        """

        :param str name:
        :param list topics:
        """
        self._name = name
        self._topics = {t.name: t for t in topics}
        if not all([topics[0].kernel_thresholds == x.kernel_thresholds for x in topics]):
            raise RuntimeError("Unexpectedly topics with different kernels defined found; the thresholds differ.")
        self._thresholds = topics[0].kernel_thresholds

    def __str__(self):
        return "{} Topics(nb_topics={})".format(self._name, len(self._topics))
    def __len__(self):
        return len(self._topics)
    def __getitem__(self, item):
        return self._topics[item]
    def __getattr__(self, item):
        return self._topics[item]
    def __contains__(self, item):
        return item in self._topics
    def __iter__(self):
        return iter(self._topics.values())
    @property
    def thresholds(self):
        return self._thresholds
    @property
    def name(self):
        return self._name
    @property
    def topic_names(self):
        return list(self._topics.keys())
    @property
    def topics_dict(self):
        return self._topics


class Topic(object):
    """
    Can be queried for its tokens with {'top_tokens' property to get a list of tokens sorted on p(w|t).\n
    Can be queried per defined kernel: eg topic_object.kernel60.tokens, topic_object.kernel80.tokens
                                       eg topic_object.kernel60.coherence, topic_object.kernel25.purity
    """
    class Kernel(object):
        def __init__(self, threshold, tokens, metrics):
            """
            :param float threshold:
            :param list tokens:
            :param dict metrics:
            """
            assert 0 < threshold <= 1
            self.threshold = threshold
            self._tokens = tokens
            self.metrics = metrics
        @property
        def coherence(self):
            return self.metrics['coherence']
        @property
        def contrast(self):
            return self.metrics['contrast']
        @property
        def purity(self):
            return self.metrics['purity']
        @property
        def tokens(self):
            return self._tokens
        def __len__(self):
            return len(self._tokens)

    def __init__(self, name, kernel_defs, top_tokens):
        """
        :param str name:
        :param list of Kernel kernel_defs: list of dicts like {'threshold': float, 'tokens': list, metrics: dict}
        :param list top_tokens:
        """
        self._name = name
        # the dict below has keys like ['60', '80', '25'] correponding to kernels with thresholds [0.6, 0.8, 0.25]
        self._kernel_info = {self._threshold_key(kr['threshold']): self.Kernel(kr['threshold'], kr['tokens'], kr['metrics']) for kr in kernel_defs}
        self._top_tokens = top_tokens

    def __str__(self):
        return "'{}': [{}]".format(self._name, ', '.join(['tt: '+str(len(self.top_tokens))] + ['{}: {}'.format(ko.threshold, len(ko.tokens)) for ko in self.kernel_objects]))

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _threshold_key(threshold):
        s = str(threshold)
        assert s.startswith('0.')
        if len(s[2:4]) < 2:
            return s[2:4] + '0'
        return s[2:4]

    @property
    def name(self):
        return self._name

    @property
    def top_tokens(self):
        return self._top_tokens

    def __getattr__(self, item):
        _ = item.replace('kernel', '')
        if _ not in self._kernel_info:
            raise AttributeError("Requested kernel with threshold 0.{} ({}), but the available ones are [{}].".format(_, item, ', '.join(self.kernel_names)))
        return self._kernel_info[_]

    @property
    def kernel_thresholds(self):
        return sorted(self._kernel_info.keys())

    @property
    def kernel_names(self):
        return map('kernel{}'.format, self.kernel_thresholds)

    @property
    def kernel_objects(self):
        return [self._kernel_info[th.replace('kernel', '')] for th in self.kernel_names]
