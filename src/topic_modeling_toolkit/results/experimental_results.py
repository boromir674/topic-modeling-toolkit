import sys
import json
from math import ceil
import attr
from functools import reduce

from collections import OrderedDict
import re

import logging
logger = logging.getLogger(__name__)



@attr.s(cmp=True, repr=True, str=True)
class ExperimentalResults(object):
    scalars = attr.ib(init=True, cmp=True, repr=True)
    tracked = attr.ib(init=True, cmp=True, repr=True)
    final = attr.ib(init=True, cmp=True, repr=True)
    regularizers = attr.ib(init=True, converter=lambda x: sorted(x), cmp=True, repr=True)
    reg_defs = attr.ib(init=True, cmp=True, repr=True)
    score_defs = attr.ib(init=True, cmp=True, repr=True)

    def __str__(self):
        return 'Scalars:\n{}\nTracked:\n{}\nFinal:\n{}\nRegularizers: {}'.format(self.scalars, self.tracked, self.final, ', '.join(self.regularizers))

    def __eq__(self, other):
        return True  # [self.scalars, self.regularizers, self.reg_defs, self.score_defs] == [other.scalars, other.regularizers, other.reg_defs, other.score_defs]

    @property
    def tracked_kernels(self):
        return [getattr(self.tracked, 'kernel'+str(x)[2:]) for x in self.tracked.kernel_thresholds]

    @property
    def tracked_top_tokens(self):
        return [getattr(self.tracked, 'top{}'.format(x)) for x in self.tracked.top_tokens_cardinalities]

    @property
    def phi_sparsities(self):
        return [getattr(self.tracked, 'sparsity_phi_'.format(x)) for x in self.tracked.modalities_initials]

    def to_json(self, human_redable=True):
        if human_redable:
            indent = 2
        else:
            indent = None
        return json.dumps(self, cls=RoundTripEncoder, indent=indent)

    def save_as_json(self, file_path, human_redable=True, debug=False):
        if human_redable:
            indent = 2
        else:
            indent = None
        import os
        if not os.path.isdir(os.path.dirname(file_path)):
            raise FileNotFoundError("Directory in path '{}' has not been created.".format(os.path.dirname(file_path)))
        with open(file_path, 'w') as fp:
            json.dump(self, fp, cls=RoundTripEncoder, indent=indent)

    @classmethod
    def _legacy(cls, key, data, factory):
        if key not in data:
            logger.info("'{}' not in experimental results dict with keys [{}]. Perhaps object is of the old format".format(key, ', '.join(data.keys())))
            return factory(key, data)
        return data[key]

    @classmethod
    def create_from_json_file(cls, file_path):
        with open(file_path, 'r') as fp:
            res = json.load(fp, cls=RoundTripDecoder)
        return cls.from_dict(res)

    @classmethod
    def from_dict(cls, data):
        return ExperimentalResults(SteadyTrackedItems.from_dict(data),
                                   ValueTracker.from_dict(data),
                                   FinalStateEntities.from_dict(data),
                                   data['regularizers'],
                                   # data.get('reg_defs', {'t'+str(i): v[:v.index('|')] for i, v in enumerate(data['regularizers'])}),
                                   cls._legacy('reg_defs', data, lambda x, y: {'type-'+str(i): v[:v.index('|')] for i, v in enumerate(y['regularizers'])}),
                                   cls._legacy('score_defs', data,
                                               lambda x, y: {k: 'name-'+str(i) for i, k in
                                                             enumerate(_ for _ in sorted(y['tracked'].keys()) if _ not in ['tau-trajectories', 'regularization-dynamic-parameters', 'collection-passes'])}))

    @classmethod
    def create_from_experiment(cls, experiment):
        """
        :param patm.modeling.experiment.Experiment experiment:
        :return:
        """
        return ExperimentalResults(SteadyTrackedItems.from_experiment(experiment),
                                   ValueTracker.from_experiment(experiment),
                                   FinalStateEntities.from_experiment(experiment),
                                   [x.label for x in experiment.topic_model.regularizer_wrappers],
                                   {k: v for k, v in zip(experiment.topic_model.regularizer_unique_types, experiment.topic_model.regularizer_names)},
                                   experiment.topic_model.definition2evaluator_name)


############### PARSER #################
@attr.s(slots=False)
class StringToDictParser(object):
    """Parses a string (if '-' found in string they are converted to '_'; any '@' is removed) trying various regexes at runtime and returns the one capturing the most information (this is simply measured by the number of entities captured"""
    regs = attr.ib(init=False, default={'score-word': r'[a-zA-Z]+',
                                        'score-sep': r'(?:-|_)',
                                        'numerical-argument': r'(?: \d+\. )? \d+',
                                        'modality-argument': r'[a-zA-Z]',
                                        'modality-argum1ent': r'@?[a-zA-Z]c?'}, converter=lambda x: dict(x, **{'score-type': r'{score-word}(?:{score-sep}{score-word})*'.format(**x)}), cmp=True, repr=True)
    normalizer = attr.ib(init=False, default={
            'kernel': 'topic-kernel',
            'top': 'top-tokens',
            'btr': 'background-tokens-ratio'
    }, cmp=False, repr=True)

    def __call__(self, *args, **kwargs):
        self.string = args[0]
        self.design = kwargs.get('design', [
            r'({score-type}) {score-sep} ({numerical-argument})',
            r'({score-type}) {score-sep} @?({modality-argument})c?$',
            r'({score-type}) ({numerical-argument})',
            r'({score-type})'])
        if kwargs.get('debug', False):
            dd = []
            for d in self.design:
                dd.append(self.search_debug(d, args[0]))
            if kwargs.get('encode', False):
                return self.encode(max(dd, key=lambda x: len(x)))
            return max(dd, key=lambda x: len(x))
        if kwargs.get('encode', False):
            return self.encode(max([self.search_n_dict(r, args[0]) for r in self.design],
                       key=lambda x: len(x)))
        return max([self.search_n_dict(r, args[0]) for r in self.design], key=lambda x: len(x))

    def search_n_dict(self, design_line, string):
        return OrderedDict([(k, v) for k, v in zip(self._entities(design_line), list(getattr(re.compile(design_line.format(**self.regs), re.X).match(string), 'groups', lambda: len(self._entities(design_line)) * [''])())) if v])

    def search_debug(self, design_line, string):
        reg = re.compile(design_line.format(**self.regs), re.X)
        res = reg.search(string)
        if res:
            ls = res.groups()
        else:
            ls = len(self._entities(design_line))*['']
        return OrderedDict([(k, v) for k , v in zip(self._entities(design_line), list(ls)) if v])

    def encode(self, ord_d):
        # try:
        #
        # except KeyError:
        #     raise KeyError("String '{}' could not be parsed. Dict {}".format(self.string, ord_d))
        ord_d['score-type'] = self.normalizer.get(ord_d['score-type'].replace('_', '-'), ord_d['score-type'].replace('_', '-'))
        if 'modality-argument' in ord_d:
            ord_d['modality-argument'] = '@{}c'.format(ord_d['modality-argument'].lower())
        if 'numerical-argument' in ord_d:
            ord_d['numerical-argument'] = self._norm_numerical(ord_d['score-type'], ord_d['numerical-argument'])
            # if ord_d['score-type'] in ['topic_kernel', 'background_tokens_ratio']:
            #     # integer = int(ord_d['numerical-argument'])
            #     a_float = float(ord_d['numerical-argument'])
            #     if int(a_float) == a_float:
            #         ord_d['numerical-argument'] = ord_d['numerical-argument'][:2]
            #     else:
            #         ord_d['numerical-argument'] = '{:.2f}'.format(a_float)[2:]
        return ord_d
        #
        #     integer = int(ord_d['numerical-argument'])
        #     a_float = float(ord_d['numerical-argument'])
        #     if integer == a_float:
        #
        #         return key, str(int(value))
        #     return key, '{:.2f}'.format(value)
        #     ord_d['numerical-argument'] = ord_d['numerical-argument'].replace('@', '').lower()
        #     return key, value.replace('@', '').lower()
        # if ord_d['score-type'] in ['top_kernel', 'background_tokens_ratio']:
        #
        # if key == 'score-type':
        #     return key, value.replace('-', '_')
        # if key == 'modality-argument':
        #     return key, value.replace('@', '').lower()
        # if key == 'numerical-argument':
        #     integer = int(value)
        #     a_float = float(value)
        #     if integer == a_float:
        #         return key, str(int(value))
        #     return key, '{:.2f}'.format(value)
        #
        # return key, value

    def _norm_numerical(self, score_type, value):
        if score_type in ['topic-kernel', 'background-tokens-ratio']:
            a_float = float(value)
            if int(a_float) == a_float:
                value = '0.' + str(int(a_float))
                # return value[:2]  # keep fist 2 characters only as the real part
            return '{:.2f}'.format(float(value))
        return value

    def _entities(self, design_line):
        return self._post_process(re.findall(r''.join([r'\({', r'(?:{score-type}|{numerical-argument}|{modality-argument})'.format(**self.regs), r'}\)']), design_line))

    def _post_process(self, entities):
        return [x[2:-2] for x in entities]


################### VALUE TRACKER ###################

@attr.s(cmp=True, repr=True, str=True)
class ValueTracker(object):
    _tracked = attr.ib(init=True, cmp=True, repr=True)
    parser = attr.ib(init=False, factory=StringToDictParser, cmp=False, repr=False)

    def __attrs_post_init__(self):
        self.scores = {}
        self._rest = {}

        for tracked_definition, v in list(self._tracked.items()):  # assumes maximum depth is 2

            d = self.parser(tracked_definition, encode=True, debug=False)
            key = '-'.join(d.values())
            # print("DEF: {}, d: {}, key: {}".format(tracked_definition, d, key))
            try:
                tracked_type = d['score-type']
            except KeyError:
                raise KeyError("String '{}' was not parsed successfully. d = {}".format(tracked_definition, d))
            if tracked_type == 'topic-kernel':
                self.scores[tracked_definition] = TrackedKernel(*v)
            elif tracked_type == 'top-tokens':
                self.scores[key] = TrackedTopTokens(*v)
            elif tracked_type in ['perplexity', 'sparsity-theta', 'background-tokens-ratio']:
                self.scores[key] = TrackedEntity(tracked_type, v)
            elif tracked_type in ['sparsity-phi']:
                self.scores[key] = TrackedEntity(tracked_type, v)
            elif tracked_type == 'tau-trajectories':
                self._rest[key] = TrackedTrajectories(v)
            elif tracked_type == 'regularization-dynamic-parameters':
                self._rest[key] = TrackedEvolvingRegParams(v)
            else:
                self._rest[key] = TrackedEntity(tracked_type, v)

    @classmethod
    def from_dict(cls, data):
        tracked = data['tracked']
        d = {'perplexity': tracked['perplexity'],
             'sparsity-theta': tracked['sparsity-theta'],
             'collection-passes': tracked['collection-passes'],
             'tau-trajectories': tracked['tau-trajectories'],
             }
        if 'regularization-dynamic-parameters' in tracked:
            d['regularization-dynamic-parameters'] = tracked['regularization-dynamic-parameters']
        else:
            logger.info("Did not find 'regularization-dynamic-parameters' in tracked values. Probably, reading from legacy formatted object")
        return ValueTracker(reduce(lambda x, y: dict(x, **y), [
            d,
            {key: v for key, v in list(tracked.items()) if key.startswith('background-tokens-ratio')},
            {'topic-kernel-' + key: [[value['avg_coh'], value['avg_con'], value['avg_pur'], value['size']],
                                     {t_name: t_data for t_name, t_data in list(value['topics'].items())}]
             for key, value in list(tracked['topic-kernel'].items())},
            {'top-tokens-' + key: [value['avg_coh'], {t_name: t_data for t_name, t_data in list(value['topics'].items())}]
                 for key, value in list(tracked['top-tokens'].items())},
            {key: v for key, v in list(tracked.items()) if key.startswith('sparsity-phi-@')}
        ]))

        # except KeyError as e:
        #     raise TypeError("Error {}, all: [{}], scalars: [{}], tracked: [{}], final: [{}]".format(e,
        #         ', '.join(sorted(data.keys())),
        #         ', '.join(sorted(data['scalars'].keys())),
        #         ', '.join(sorted(data['tracked'].keys())),
        #         ', '.join(sorted(data['final'].keys())),
        #     ))

    @classmethod
    def from_experiment(cls, experiment):
        return ValueTracker(reduce(lambda x, y: dict(x, **y), [
            {'perplexity': experiment.trackables['perplexity'],
             'sparsity-theta': experiment.trackables['sparsity-theta'],
             'collection-passes': experiment.collection_passes,
             'tau-trajectories': {matrix_name: experiment.regularizers_dynamic_parameters.get('sparse-' + matrix_name, {}).get('tau', []) for matrix_name in ['theta', 'phi']},
             'regularization-dynamic-parameters': experiment.regularizers_dynamic_parameters},
            {key: v for key, v in list(experiment.trackables.items()) if key.startswith('background-tokens-ratio')},
            {kernel_definition: [[value[0], value[1], value[2], value[3]], value[4]] for kernel_definition, value in
                 list(experiment.trackables.items()) if kernel_definition.startswith('topic-kernel')},
            {top_tokens_definition: [value[0], value[1]] for top_tokens_definition, value in
                 list(experiment.trackables.items()) if top_tokens_definition.startswith('top-tokens-')},
            {key: v for key, v in list(experiment.trackables.items()) if key.startswith('sparsity-phi-@')},
        ]))

    def __dir__(self):
        return sorted(list(self.scores.keys()) + list(self._rest.keys()))

    @property
    def top_tokens_cardinalities(self):
        return sorted(int(_.split('-')[-1]) for _ in list(self.scores.keys()) if _.startswith('top-tokens'))

    @property
    def kernel_thresholds(self):
        """
        :return: list of strings eg ['0.60', '0.80', '0.25']
        """
        return sorted(_.split('-')[-1] for _ in list(self.scores.keys()) if _.startswith('topic-kernel'))

    @property
    def modalities_initials(self):
        return sorted(_.split('-')[-1][1] for _ in list(self.scores.keys()) if _.startswith('sparsity-phi'))

    @property
    def tracked_entity_names(self):
        return sorted(list(self.scores.keys()) + list(self._rest.keys()))

    @property
    def background_tokens_thresholds(self):
        return sorted(_.split('-')[-1] for _ in list(self.scores.keys()) if _.startswith('background-tokens-ratio'))

    @property
    def tau_trajectory_matrices_names(self):
        try:
            return self._rest['tau-trajectories'].matrices_names
        except KeyError:
            raise KeyError("Key 'tau-trajectories' was not found in scores [{}]. ALL: [{}]".format(
                ', '.join(sorted(self.scores.keys())), ', '.join(self.tracked_entity_names)))

    def __getitem__(self, item):
        d = self.parser(item, encode=True, debug=False)
        key = '-'.join(d.values())
        if key in self.scores:
            return self.scores[key]
        elif key in self._rest:
            return self._rest[key]
        raise KeyError(
            "Requested item '{}', converted to '{}' but it was not found either in ValueTracker.scores [{}] nor in ValueTracker._rest [{}]".format(
                item, key, ', '.join(sorted(self.scores.keys())), ', '.join(sorted(self._rest.keys()))))

    def __getattr__(self, item):
        d = self.parser(item, encode=True, debug=False)
        key = '-'.join(d.values())
        if key in self.scores:
            return self.scores[key]
        elif key in self._rest:
            return self._rest[key]
        raise AttributeError(
            "Requested item '{}', converted to '{}' after parsed as {}. It was not found either in ValueTracker.scores [{}] nor in ValueTracker._rest [{}]".format(
                item, key, d, ', '.join(sorted(self.scores.keys())), ', '.join(sorted(self._rest.keys()))))

##############################################################


@attr.s(cmp=True, repr=True, str=True, slots=True)
class TrackedEvolvingRegParams(object):
    """Holds regularizers_parameters data which is a dictionary: keys should be the unique regularizer definitions
    (as in train.cfg; eg 'label-regularization-phi-dom-cls'). Keys should map to dictionaries, which map strings to lists.
    These dictionaries keys should correspond to one of the supported dynamic parameters (eg 'tau', 'gamma') and each
    list should have length equal to the number of collection iterations and hold the evolution/trajectory of the parameter values
    """
    _evolved = attr.ib(init=True, converter=lambda regularizers_params: {reg: {param: TrackedEntity(param, values_list) for param, values_list in reg_params.items()} for
                     reg, reg_params in regularizers_params.items()}, cmp=True)
    regularizers_definitions = attr.ib(init=False, default=attr.Factory(lambda self: sorted(self._evolved.keys()), takes_self=True), cmp=False, repr=True)

    def __iter__(self):
        return ((k, v) for k, v in self._evolved.items())

    def __getattr__(self, item):
        return self._evolved[item]

@attr.s(cmp=True, repr=True, str=True, slots=True)
class TrackedKernel(object):
    """data = [avg_coherence_list, avg_contrast_list, avg_purity_list, sizes_list, topic_name2elements_hash]"""
    average = attr.ib(init=True, cmp=True, converter=lambda x: KernelSubGroup(x[0], x[1], x[2], x[3]))
    _topics_data = attr.ib(converter=lambda topic_name2elements_hash: {key: KernelSubGroup(val['coherence'], val['contrast'], val['purity'], key) for key, val in list(topic_name2elements_hash.items())}, init=True, cmp=True)
    topics = attr.ib(default=attr.Factory(lambda self: sorted(self._topics_data.keys()), takes_self=True))

    def __getattr__(self, item):
        if item in self._topics_data:
            return self._topics_data[item]
        raise AttributeError("Topic '{}' is not registered as tracked".format(item))

@attr.s(cmp=True, repr=True, str=True, slots=True)
class TrackedTopTokens(object):
    average_coherence = attr.ib(init=True, converter=lambda x: TrackedEntity('average_coherence', x), cmp=True, repr=True)
    _topics_data = attr.ib(init=True, converter=lambda topic_name2coherence: {key: TrackedEntity(key, val) for key, val in list(topic_name2coherence.items())}, cmp=True, repr=True)
    topics = attr.ib(init=False, default=attr.Factory(lambda self: sorted(self._topics_data.keys()), takes_self=True), cmp=False, repr=True)

    def __getattr__(self, item):
        if item in self._topics_data:
            return self._topics_data[item]
        raise AttributeError("Topic '{}' is not registered as tracked".format(item))


@attr.s(cmp=True, repr=True, str=True, slots=True)
class KernelSubGroup(object):
    """Containes averages over topics for metrics computed on lexical kernels defined with a specific threshold [coherence, contrast, purity] and the average kernel size"""
    coherence = attr.ib(init=True, converter=lambda x: TrackedEntity('coherence', x), cmp=True, repr=True)
    contrast = attr.ib(init=True, converter=lambda x: TrackedEntity('contrast', x), cmp=True, repr=True)
    purity = attr.ib(init=True, converter=lambda x: TrackedEntity('purity', x), cmp=True, repr=True)
    size = attr.ib(init=True, converter=lambda x: TrackedEntity('size', x), cmp=True, repr=True)
    name = attr.ib(init=True, default='', cmp=True, repr=True)


@attr.s(cmp=True, repr=True, str=True, slots=True)
class TrackedTrajectories(object):
    _trajs = attr.ib(init=True, converter=lambda matrix_name2elements_hash: {k: TrackedEntity(k, v) for k, v in list(matrix_name2elements_hash.items())}, cmp=True, repr=True)
    trajectories = attr.ib(init=False, default=attr.Factory(lambda self: sorted(list(self._trajs.items()), key=lambda x: x[0]), takes_self=True), cmp=False, repr=False)
    matrices_names = attr.ib(init=False, default=attr.Factory(lambda self: sorted(self._trajs.keys()), takes_self=True), cmp=False, repr=True)

    @property
    def phi(self):
        if 'phi' in self._trajs:
            return self._trajs['phi']
        raise AttributeError

    @property
    def theta(self):
        if 'theta' in self._trajs:
            return self._trajs['theta']
        raise AttributeError

    def __str__(self):
        return str(self.matrices_names)


@attr.s(cmp=True, repr=True, str=True, slots=True)
class TrackedEntity(object):
    name = attr.ib(init=True, converter=str, cmp=True, repr=True)
    all = attr.ib(init=True, converter=list, cmp=True, repr=True)  # list of elements (values tracked)

    @property
    def last(self):
        return self.all[-1]

    def __len__(self):
        return len(self.all)

    def __getitem__(self, item):
        if item == 'all':
            return self.all
        try:
            return self.all[item]
        except KeyError:
            raise KeyError("self: {} type of self._elements: {}".format(self, type(self.all).__name__))

######### CONSTANTS during training and between subsequent fit calls

def _check_topics(self, attribute, value):
    if len(value) != len(set(value)):
        raise RuntimeError("Detected duplicates in input topics: [{}]".format(', '.join(str(x) for x in value)))

def _non_overlapping(self, attribute, value):
    if any(x in value for x in self.background_topics):
        raise RuntimeError("Detected overlapping domain topics [{}] with background topics [{}]".format(', '.join(str(x) for x in value), ', '.join(str(x) for x in self.background_topics)))

def background_n_domain_topics_soundness(instance, attribute, value):
    if instance.nb_topics != len(instance.background_topics) + len(value):
        raise ValueError("nb_topics should be equal to len(background_topics) + len(domain_topics). Instead, {} != {} + {}".format(instance.nb_topics, len(instance.background_topics), len(value)))


@attr.s(repr=True, cmp=True, str=True, slots=True)
class SteadyTrackedItems(object):
    """Supportes only one tokens list for a specific btr threshold."""
    dir = attr.ib(init=True, converter=str, cmp=True, repr=True)  # T.O.D.O. rename to 'dataset_dir'
    model_label = attr.ib(init=True, converter=str, cmp=True, repr=True)
    dataset_iterations = attr.ib(init=True, converter=int, cmp=True, repr=True)
    nb_topics = attr.ib(init=True, converter=int, cmp=True, repr=True)
    document_passes = attr.ib(init=True, converter=int, cmp=True, repr=True)
    background_topics = attr.ib(init=True, converter=list, cmp=True, repr=True, validator=_check_topics)
    domain_topics = attr.ib(init=True, converter=list, cmp=True, repr=True, validator=[_check_topics, _non_overlapping, background_n_domain_topics_soundness])
    background_tokens_threshold = attr.ib(init=True, converter=float, cmp=True, repr=True)
    modalities = attr.ib(init=True, converter=dict, cmp=True, repr=True)

    parser = StringToDictParser()
    def __dir__(self):
        return ['dir', 'model_label', 'dataset_iterations', 'nb_topics', 'document_passes', 'background_topics', 'domain_topics', 'background_tokens_threshold', 'modalities']
    # def __str__(self):
    #     return '\n'.join(['{}: {}'.format(x, getattr(self, x)) for x in dir(self)])

    @classmethod
    def from_dict(cls, data):
        steady = data['scalars']
        background_tokens_threshold = 0  # no distinction between background and "domain" tokens
        for eval_def in data['tracked']:
            if eval_def.startswith('background-tokens-ratio'):
                background_tokens_threshold = max(background_tokens_threshold, float(eval_def.split('-')[-1]))
        return SteadyTrackedItems(steady['dir'], steady['label'], data['scalars']['dataset_iterations'], steady['nb_topics'],
                                  steady['document_passes'], steady['background_topics'], steady['domain_topics'], background_tokens_threshold, steady['modalities'])

    @classmethod
    def from_experiment(cls, experiment):
        """
        Uses the maximum threshold found amongst the defined 'background-tokens-ratio' scores.\n
        :param patm.modeling.experiment.Experiment experiment:
        :return:
        """
        ds = [d for d in experiment.topic_model.evaluator_definitions if d.startswith('background-tokens-ratio-')]
        m = 0
        for d in (_.split('-')[-1] for _ in ds):
            if str(d).startswith('0.'):
                m = max(m, float(d))
            else:
                m = max(m, float('0.' + str(d)))

        return SteadyTrackedItems(experiment.current_root_dir, experiment.topic_model.label, experiment.dataset_iterations,
                                  experiment.topic_model.nb_topics, experiment.topic_model.document_passes, experiment.topic_model.background_topics,
                                  experiment.topic_model.domain_topics, m, experiment.topic_model.modalities_dictionary)


@attr.s(repr=True, str=True, cmp=True, slots=True)
class TokensList(object):
    tokens = attr.ib(init=True, converter=list, repr=True, cmp=True)
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, item):
        return self.tokens[item]
    def __iter__(self):
        return iter(self.tokens)
    def __contains__(self, item):
        return item in self.tokens


def kernel_def2_kernel(kernel_def):
    return 'kernel'+kernel_def.split('.')[-1]

def top_tokens_def2_top(top_def):
    return 'top'+top_def.split('-')[-1]

@attr.s(repr=True, str=True, cmp=True, slots=True)
class FinalStateEntities(object):
    kernel_hash = attr.ib(init=True, converter=lambda x: {kernel_def: TopicsTokens(data) for kernel_def, data in x.items()}, cmp=True, repr=False)
    top_hash = attr.ib(init=True, converter=lambda x: {top_tokens_def: TopicsTokens(data) for top_tokens_def, data in x.items()}, cmp=True, repr=False)
    _bg_tokens = attr.ib(init=True, converter=TokensList, cmp=True, repr=False)
    kernel_defs = attr.ib(init=False, default=attr.Factory(lambda self: sorted(self.kernel_hash.keys()), takes_self=True))
    kernels = attr.ib(init=False, default=attr.Factory(lambda self: [kernel_def2_kernel(x) for x in self.kernel_defs], takes_self=True))
    top_defs = attr.ib(init=False, default=attr.Factory(lambda self: sorted(self.top_hash.keys()), takes_self=True))
    top = attr.ib(init=False, default=attr.Factory(lambda self: [top_tokens_def2_top(x) for x in self.top_defs], takes_self=True))
    background_tokens = attr.ib(init=False, default=attr.Factory(lambda self: sorted(self._bg_tokens.tokens), takes_self=True))

    parse = attr.ib(init=False, factory=StringToDictParser, cmp=False, repr=False)

    def __getattr__(self, item):
        d = self.parse(item, encode=True, debug=False)
        key = '-'.join(d.values())
        if key in self.kernel_hash:
            return self.kernel_hash[key]
        elif key in self.top_hash:
            return self.top_hash[key]
        raise KeyError(
            "Requested item '{}', converted to '{}' after parsed as {}. It was not found either in kernel thresholds [{}] nor in top-toknes cardinalities [{}]".format(
                item, key, d, ', '.join(sorted(self.kernel_hash.keys())), ', '.join(sorted(self.top_hash.keys()))))

    def __str__(self):
        return 'bg-tokens: {}\n'.format(len(self._bg_tokens)) + '\n'.join(['final state: {}\n{}'.format(y, getattr(self, y)) for y in self.top + self.kernels])

    @classmethod
    def from_dict(cls, data):
        return FinalStateEntities(
            {'topic-kernel-' + threshold: tokens_hash for threshold, tokens_hash in list(data['final']['topic-kernel'].items())},
            {'top-tokens-' + nb_tokens: tokens_hash for nb_tokens, tokens_hash in list(data['final']['top-tokens'].items())},
            data['final']['background-tokens']
        )

    @classmethod
    def from_experiment(cls, experiment):
        return FinalStateEntities(experiment.final_tokens['topic-kernel'],experiment.final_tokens['top-tokens'], experiment.final_tokens['background-tokens'])

@attr.s(repr=True, cmp=True, slots=True)
class TopicsTokens(object):
    _tokens = attr.ib(init=True, converter=lambda topic_name2tokens: {topic_name: TokensList(tokens_list) for topic_name, tokens_list in topic_name2tokens.items()}, cmp=True, repr=False)
    _nb_columns = attr.ib(init=False, default=10, cmp=False, repr=False)
    _nb_rows = attr.ib(init=False, default=attr.Factory(lambda self: int(ceil((float(len(self._tokens)) / self._nb_columns))), takes_self=True), repr=False, cmp=False)
    topics = attr.ib(init=False, default=attr.Factory(lambda self: sorted(self._tokens.keys()), takes_self=True), repr=True, cmp=False)
    __lens = attr.ib(init=False, default=attr.Factory(lambda self: {k : {'string': len(k), 'list': len(str(len(v)))} for k, v in self._tokens.items()}, takes_self=True), cmp=False, repr=False)

    def __iter__(self):
        return ((k,v) for k,v in self._tokens.items())

    def __getattr__(self, item):
        if item in self._tokens:
            return self._tokens[item]
        raise AttributeError

    def __str__(self):
        return '\n'.join([self._get_row_string(x) for x in self._gen_rows()])

    def _gen_rows(self):
        i, j = 0, 0
        while i<len(self._tokens):
            _ = self._index(j)
            j += 1
            i += len(_)
            yield _

    def _index(self, row_nb):
        return [_f for _f in [self.topics[x] if x<len(self._tokens) else None for x in range(self._nb_columns * row_nb, self._nb_columns * row_nb + self._nb_columns)] if _f]

    def _get_row_string(self, row_topic_names):
        return '  '.join(['{}{}: {}{}'.format(
            x[1], (self._max(x[0], 'string')-len(x[1]))*' ', (self._max(x[0], 'list')-self.__lens[x[1]]['list'])*' ', len(self._tokens[x[1]])) for x in enumerate(row_topic_names)])

    def _max(self, column_index, entity):
        return max([self.__lens[self.topics[x]][entity] for x in self._range(column_index)])

    def _range(self, column_index):
        return list(range(column_index, self._get_last_index_in_column(column_index), self._nb_columns))

    def _get_last_index_in_column(self, column_index):
        return self._nb_rows * self._nb_columns - self._nb_columns + column_index


class RoundTripEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, TokensList):
            return obj.tokens
        if isinstance(obj, TopicsTokens):
            return {topic_name: list(getattr(obj, topic_name)) for topic_name in obj.topics}
        if isinstance(obj, FinalStateEntities):
            return {'topic-kernel': {kernel_def.split('-')[-1]: getattr(obj, kernel_def2_kernel(kernel_def)) for kernel_def in obj.kernel_defs},
                    'top-tokens': {top_def.split('-')[-1]: getattr(obj, top_tokens_def2_top(top_def)) for top_def in obj.top_defs},
                    'background-tokens': obj.background_tokens}
        if isinstance(obj, SteadyTrackedItems):
            return {'dir': obj.dir,
                    'label': obj.model_label,
                    'dataset_iterations': obj.dataset_iterations,
                    'nb_topics': obj.nb_topics,
                    'document_passes': obj.document_passes,
                    'background_topics': obj.background_topics,
                    'domain_topics': obj.domain_topics,
                    'modalities': obj.modalities}
        if isinstance(obj, TrackedEntity):
            return obj.all
        if isinstance(obj, TrackedTrajectories):
            return {matrix_name: tau_elements for matrix_name, tau_elements in obj.trajectories}
        if isinstance(obj, TrackedEvolvingRegParams):
            return dict(obj)
        if isinstance(obj, TrackedTopTokens):
            return {'avg_coh': obj.average_coherence,
                    'topics': {topic_name: getattr(obj, topic_name).all for topic_name in obj.topics}}
        if isinstance(obj, TrackedKernel):
            return {'avg_coh': obj.average.coherence,
                    'avg_con': obj.average.contrast,
                    'avg_pur': obj.average.purity,
                    'size': obj.average.size,
                    'topics': {topic_name: {'coherence': getattr(obj, topic_name).coherence.all,
                                            'contrast': getattr(obj, topic_name).contrast.all,
                                            'purity': getattr(obj, topic_name).purity.all} for topic_name in obj.topics}}
        if isinstance(obj, ValueTracker):
            _ = {name: tracked_entity for name, tracked_entity in list(obj.scores.items()) if name not in ['tau-trajectories', 'regularization-dynamic-parameters'] or
                 all(name.startswith(x) for x in ['top-tokens', 'topic-kernel'])}
            # _ = {name: tracked_entity for name, tracked_entity in list(obj._flat.items())}
            _['top-tokens'] = {k: obj.scores['top-tokens-' + str(k)] for k in obj.top_tokens_cardinalities}
            _['topic-kernel'] = {k: obj.scores['topic-kernel-' + str(k)] for k in obj.kernel_thresholds}
            _['tau-trajectories'] = {k: getattr(obj.tau_trajectories, k) for k in obj.tau_trajectory_matrices_names}
            _['collection-passes'] = obj.collection_passes,
            _['regularization-dynamic-parameters'] = obj.regularization_dynamic_parameters
            return _
        if isinstance(obj, ExperimentalResults):
            return {'scalars': obj.scalars,
                    'tracked': obj.tracked,
                    'final': obj.final,
                    'regularizers': obj.regularizers,
                    'reg_defs': obj.reg_defs,
                    'score_defs': obj.score_defs}
        return super(RoundTripEncoder, self).default(obj)


class RoundTripDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        return obj
