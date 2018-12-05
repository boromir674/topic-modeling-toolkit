from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import sys
from math import ceil
import json
from abc import ABCMeta, abstractproperty
from future.utils import with_metaclass
from functools import reduce


class ExperimentalResults(object):
    def __init__(self, root_dir, model_label, nb_topics, document_passes, background_topics, domain_topics, modalities,
                 tracked, kernel_tokens, top_tokens_defs, background_tokens, regularizers_labels):
        """
        Examples:
            4 elements: kernel_data = [[1,2], [3,4], [5,6], {'t01': {'coherence': [1,2,3],
                                                                 'contrast': [6, 3],
                                                                 'purity': [1, 8]},
                                                         't00': {'coherence': [10,2,3],
                                                                 'contrast': [67, 36],
                                                                 'purity': [12, 89]},
                                                         't02': {'coherence': [10,11],
                                                                 'contrast': [656, 32],
                                                                 'purity': [17, 856]}
                                                         }]

            top10_data = [[1, 2], {'t01': [1,2,3], 't00': [10,2,3], 't02': [10,11]}]
            tau_trajectories_data = {'phi': [1,2,3,4,5], 'theta': [5,6,7,8,9]} \n
            tracked = {'perplexity': [1, 2, 3], 'kernel': kernel_data, 'top10': top10_data, 'tau_trajectories': tau_trajectories_data}

        :param str root_dir:
        :param str model_label:
        :param int nb_topics:
        :param int document_passes:
        :param list of str background_topics:
        :param list of str domain_topics:
        :param dict modalities: class_name => weight , ie {'@default_class': 1.0, '@ideology_class': 5.0}
        :param tracked: ie kernel_data = [[1,2], [3,4], [5,6], {'t01': {'coherence': [1,2,3],\n
                                                                 'contrast': [6, 3],\n
                                                                 'purity': [1, 8]},\n
                                                         't00': {'coherence': [10,2,3],\n
                                                                 'contrast': [67, 36],\n
                                                                 'purity': [12, 89]},\n
                                                         't02': {'coherence': [10,11],\n
                                                                 'contrast': [656, 32],\n
                                                                 'purity': [17, 856]}\n
                                                         }]\n

            top10_data = [[1, 2], {'t01': [1,2,3], 't00': [10,2,3], 't02': [10,11]}]
            tau_trajectories_data = {'phi': [1,2,3,4,5], 'theta': [5,6,7,8,9]}
        :param dict kernel_tokens: topic_name => list_of_tokens
        :param dict top_tokens_defs:
        :param list background_tokens: list of background tokens, p(t) and p(t | w) distributions \mathrm{KL}(p(t) | | p(t | w)) (or vice versa)
            for each token and counts the part of tokens that have this value greater than a given (non-negative) delta_threshold.
        """
        assert len(background_topics) + len(domain_topics) == nb_topics
        assert sum(tracked['collection_passes']) == len(tracked['perplexity'])
        background_tokens_threshold = 0 # no distinction between background and "domain" tokens
        for eval_def in tracked:
            if eval_def.startswith('background-tokens-ratio'):
                background_tokens_threshold = float(eval_def.split('-')[-1])
        self._steady_container = SteadyTrackedItems(root_dir, model_label, sum(tracked['collection_passes']), nb_topics, document_passes,
                                                    background_topics, domain_topics, background_tokens_threshold, modalities)
        self._tracker = ValueTracker(tracked)
        self._final_state_items = FinalStateEntities(dict([(x[0], dict([(y[0], list(y[1])) for y in list(x[1].items())])) for x in list(kernel_tokens.items())]),
                                                     dict([(x[0], dict([(y[0], list(y[1])) for y in list(x[1].items())])) for x in list(top_tokens_defs.items())]),
                                                     background_tokens)
        self._regularizers = regularizers_labels

    def __str__(self):
        return 'Scalars:\n{}\nTracked:\n{}\nFinal:\n{}\nRegularizers: {}'.format(self._steady_container, self._tracker, self._final_state_items, ', '.join(self._regularizers))

    @property
    def regularizers(self):
        return sorted(self._regularizers)

    @property
    def scalars(self):
        return self._steady_container

    @property
    def tracked(self):
        return self._tracker

    @property
    def final(self):
        return self._final_state_items

    @property
    def tracked_kernels(self):
        return [getattr(self._tracker, 'kernel'+str(x)[2:]) for x in self._tracker.kernel_thresholds]

    @property
    def tracked_top_tokens(self):
        return [getattr(self._tracker, 'top{}'.format(x)) for x in self._tracker.top_tokens_cardinalities]

    @property
    def phi_sparsities(self):
        return [getattr(self._tracker, 'sparsity_phi_'+x) for x in self._tracker.modalities_initials]

    def to_json(self, human_redable=True):
        if human_redable:
            indent = 2
        else:
            indent = None
        return json.dumps(self, cls=RoundTripEncoder, indent=indent)

    def save_as_json(self, file_path, human_redable=True):
        if human_redable:
            indent = 2
        else:
            indent = None
        with open(file_path, 'w') as fp:
            json.dump(self, fp, cls=RoundTripEncoder, indent=indent)

    @classmethod
    def create_from_json_file(cls, file_path):
        with open(file_path, 'r') as fp:
            res = json.load(fp, cls=RoundTripDecoder)
        data = [{'perplexity': res['tracked']['perplexity'],
                     'sparsity-theta': res['tracked']['sparsity-theta'],
                     'collection_passes': res['tracked']['collection-passes']},
                {key: v for key, v in list(res['tracked'].items()) if key.startswith('background-tokens-ratio')},
                {'tau-trajectories': res['tracked']['tau-trajectories']},
                {'topic-kernel-' + key: [value['avg_coh'], value['avg_con'], value['avg_pur'],
                                         {t_name: t_data for t_name, t_data in
                                          list(value['topics'].items())}] for key, value in
                 list(res['tracked']['topic-kernel'].items())},
                {'top-tokens-' + key: [value['avg_coh'], {t_name: t_data for t_name, t_data in list(value['topics'].items())}]
                 for key, value in list(res['tracked']['top-tokens'].items())},
                {key: v for key, v in list(res['tracked'].items()) if key.startswith('sparsity-phi-@')}]

        # data = list({'perplexity': res['tracked']['perplexity'],
        #              'sparsity-theta': res['tracked']['sparsity-theta'],
        #              'collection_passes': res['tracked']['collection-passes']})
        # data.append({key: v for key, v in res['tracked'].items() if key.startswith('background-tokens-ratio')})
        # data.append({'tau-trajectories': res['tracked']['tau-trajectories']})
        # data.append({'topic-kernel-' + key: [value['avg_coh'], value['avg_con'], value['avg_pur'],
        #                                                           {t_name: t_data for t_name, t_data in
        #                                                            value['topics'].items()}] for key, value in
        #                                   res['tracked']['topic-kernel'].items()})
        # self._data.append({'top-tokens-' + key: [value['avg_coh'], {t_name: t_data for t_name, t_data in value['topics'].items()}] for key, value in res['tracked']['top-tokens'].items()})
        # self._data.append({key: v for key, v in res['tracked'].items() if key.startswith('sparsity-phi-@')})
        return ExperimentalResults(res['scalars']['dir'],
                                   res['scalars']['label'],
                                   res['scalars']['nb_topics'],
                                   res['scalars']['document_passes'],
                                   res['scalars']['background_topics'],
                                   res['scalars']['domain_topics'],
                                   res['scalars']['modalities'],
                                   reduce(lambda x, y: dict(x, **y), data),
                                   {'topic-kernel-'+threshold: tokens_hash for threshold, tokens_hash in list(res['final']['topic-kernel'].items())},
                                   {'top-tokens-'+nb_tokens: tokens_hash for nb_tokens, tokens_hash in list(res['final']['top-tokens'].items())},
                                   res['final']['background-tokens'],
                                   res['regularizers'])

    @classmethod
    def create_from_experiment(cls, experiment):
        data = [{'perplexity': experiment.trackables['perplexity'],
                       'sparsity-theta': experiment.trackables['sparsity-theta'],
                       'collection_passes': experiment.collection_passes},
                {kernel_definition: [value[0], value[1], value[2], value[3]] for kernel_definition, value in
                 list(experiment.trackables.items()) if kernel_definition.startswith('topic-kernel')},
                {top_tokens_definition: [value[0], value[1]] for top_tokens_definition, value in
                 list(experiment.trackables.items()) if top_tokens_definition.startswith('top-tokens-')},
                {'tau-trajectories': {matrix_name: experiment.reg_params['sparse-' + matrix_name]['tau']} for
                 matrix_name in ['phi', 'theta']},
                {key: v for key, v in list(experiment.trackables.items()) if key.startswith('sparsity-phi-@')},
                {key: v for key, v in list(experiment.trackables.items()) if key.startswith('background-tokens-ratio')}]
        final_kernel_tokens = {eval_def: cls._get_final_tokens(experiment, eval_def) for eval_def in experiment.topic_model.evaluator_definitions if eval_def.startswith('topic-kernel-')}
        final_top_tokens = {eval_def: cls._get_final_tokens(experiment, eval_def) for eval_def in experiment.topic_model.evaluator_definitions if eval_def.startswith('top-tokens-')}
        return ExperimentalResults(experiment.current_root_dir,
                                   experiment.topic_model.label,
                                   experiment.topic_model.nb_topics,
                                   experiment.topic_model.document_passes,
                                   experiment.topic_model.background_topics,
                                   experiment.topic_model.domain_topics,
                                   experiment.topic_model.modalities_dictionary,
                                   reduce(lambda x, y: dict(x, **y), data),
                                   final_kernel_tokens,
                                   final_top_tokens,
                                   experiment.topic_model.background_tokens,
                                   [x.label for x in experiment.topic_model.regularizer_wrappers])

    @staticmethod
    def _get_final_tokens(experiment, evaluation_definition):
        return experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[evaluation_definition]].tokens[-1]

    @staticmethod
    def _get_evolved_tokens(experiment, evaluation_definition):
        """
        :param experiment:
        :param evaluation_definition:
        :return:
        :rtype: list of dicts
        """
        return experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[evaluation_definition]].tokens


# class ExperimentalResultsFactory(object):
#     def __init__(self):
#         self._data = []
#
#     def create_from_json_file(self, file_path):
#         with open(file_path, 'r') as fp:
#             res = json.load(fp, cls=RoundTripDecoder)
#         self._data =[{'perplexity': res['tracked']['perplexity'],
#                       'sparsity-theta': res['tracked']['sparsity-theta'],
#                       'collection_passes': res['tracked']['collection-passes']}]
#         self._data.append({key: v for key, v in res['tracked'].items() if key.startswith('background-tokens-ratio')})
#         self._data.append({'tau-trajectories': res['tracked']['tau-trajectories']})
#         self._data.append({'topic-kernel-' + key: [value['avg_coh'], value['avg_con'], value['avg_pur'],
#                                                                   {t_name: t_data for t_name, t_data in
#                                                                    value['topics'].items()}] for key, value in
#                                           res['tracked']['topic-kernel'].items()})
#         self._data.append({'top-tokens-' + key: [value['avg_coh'], {t_name: t_data for t_name, t_data in value['topics'].items()}] for key, value in res['tracked']['top-tokens'].items()})
#         self._data.append({key: v for key, v in res['tracked'].items() if key.startswith('sparsity-phi-@')})
#         return ExperimentalResults(res['scalars']['dir'],
#                                    res['scalars']['label'],
#                                    res['scalars']['nb_topics'],
#                                    res['scalars']['document_passes'],
#                                    res['scalars']['background_topics'],
#                                    res['scalars']['domain_topics'],
#                                    res['scalars']['modalities'],
#                                    reduce(lambda x, y: dict(x, **y), self._data),
#                                    {'topic-kernel-'+threshold: tokens_hash for threshold, tokens_hash in res['final']['topic-kernel'].items()},
#                                    {'top-tokens-'+nb_tokens: tokens_hash for nb_tokens, tokens_hash in res['final']['top-tokens'].items()},
#                                    res['final']['background-tokens'],
#                                    res['regularizers'])
#
#     def create_from_experiment(self, experiment):
#         self._data = [{'perplexity': experiment.trackables['perplexity'],
#                        'sparsity-theta': experiment.trackables['sparsity-theta'],
#                        'collection_passes': experiment.collection_passes}]
#         self._data.append({kernel_definition: [value[0], value[1], value[2], value[3]] for kernel_definition, value in experiment.trackables.items() if kernel_definition.startswith('topic-kernel')})
#         self._data.append({top_tokens_definition: [value[0], value[1]] for top_tokens_definition, value in experiment.trackables.items() if top_tokens_definition.startswith('top-tokens-')})
#         self._data.append({'tau-trajectories': {matrix_name: experiment.reg_params['sparse-'+matrix_name]['tau']} for matrix_name in ['phi', 'theta']})
#         self._data.append({key: v for key, v in experiment.trackables.items() if key.startswith('sparsity-phi-@')})
#         self._data.append({key: v for key, v in experiment.trackables.items() if key.startswith('background-tokens-ratio')})
#         final_kernel_tokens = {eval_def: self._get_final_tokens(experiment, eval_def) for eval_def in experiment.topic_model.evaluator_definitions if eval_def.startswith('topic-kernel-')}
#         final_top_tokens = {eval_def: self._get_final_tokens(experiment, eval_def) for eval_def in experiment.topic_model.evaluator_definitions if eval_def.startswith('top-tokens-')}
#         return ExperimentalResults(experiment.current_root_dir,
#                                    experiment.topic_model.label,
#                                    experiment.topic_model.nb_topics,
#                                    experiment.topic_model.document_passes,
#                                    experiment.topic_model.background_topics,
#                                    experiment.topic_model.domain_topics,
#                                    experiment.topic_model.modalities_dictionary,
#                                    reduce(lambda x, y: dict(x, **y), self._data),
#                                    final_kernel_tokens,
#                                    final_top_tokens,
#                                    experiment.topic_model.background_tokens,
#                                    map(lambda x: x.label, experiment.topic_model.regularizer_wrappers))
#
#     def _get_final_tokens(self, experiment, evaluation_definition):
#         return experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[evaluation_definition]].tokens[-1]
#
#     def _get_evolved_tokens(self, experiment, evaluation_definition):
#         """
#         :param experiment:
#         :param evaluation_definition:
#         :return:
#         :rtype: list of dicts
#         """
#         return experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[evaluation_definition]].tokens
#
#
# experimental_results_factory = ExperimentalResultsFactory()


class AbstractValueTracker(with_metaclass(ABCMeta, object)):
    def __dir__(self):
        return [_f for _f in [self._trans('perplexity'), self._trans('sparsity_theta')] + ['sparsity_phi_' + x for x in self.modalities_initials] + \
               ['kernel'+str(x)[2:] for x in self.kernel_thresholds] + \
               ['top' + str(x) for x in self.top_tokens_cardinalities] + ['tau_trajectories'] + \
               ['background_tokens_ratio_'+str(x)[2:] for x in self.background_tokens_thresholds] + [self._trans('collection_passes')] if _f]

    def _trans(self, dash_splitable):
        if dash_splitable not in self._flat: return None
        return dash_splitable.replace('-', '_')
    def __repr__(self):
        return '[{}]'.format(', '.join(dir(self)))
    def __str__(self):
        return '[{}]'.format(', '.join(sorted(self._flat.keys()) + sorted(self._groups.keys())))

    def __init__(self, tracked):
        self._flat = {}
        self._groups = {}
        for evaluator_definition, v in list(tracked.items()):  # assumes maximum depth is 2
            score_type = _strip_parameters(evaluator_definition.replace('_', '-'))
            if score_type == 'topic-kernel':
                self._groups[evaluator_definition] = TrackedKernel(*v)
            elif score_type == 'top-tokens':
                self._groups[evaluator_definition] = TrackedTopTokens(*v)
            elif score_type == 'tau-trajectories':
                self._groups[score_type] = TrackedTrajectories(v)
            else:
                self._flat[evaluator_definition.replace('_', '-')] = TrackedEntity(score_type, v)

    @property
    def top_tokens_cardinalities(self):
        return sorted(int(_.split('-')[-1]) for _ in list(self._groups.keys()) if _.startswith('top-tokens'))

    @property
    def kernel_thresholds(self):
        return sorted(float(_.split('-')[-1]) for _ in list(self._groups.keys()) if _.startswith('topic-kernel'))

    @property
    def modalities_initials(self):
        return sorted(_.split('-')[-1][1] for _ in list(self._flat.keys()) if _.startswith('sparsity-phi'))

    @property
    def tracked_entity_names(self):
        return sorted(list(self._flat.keys()) + list(self._groups.keys()))

    @property
    def background_tokens_thresholds(self):
        return sorted(float(_.split('-')[-1]) for _ in list(self._flat.keys()) if _.startswith('background-tokens-ratio-0.'))

    def _query_metrics(self):
        _ = self._decode(sys._getframe(1).f_code.co_name).replace('_', '-')
        if _ in self._flat:
            return self._flat[_]
        if _ in self._groups:
            return self._groups[_]
        raise AttributeError("'{}' not found within [{}]".format(_, ', '.join(list(self._flat.keys()) + list(self._groups))))

    def _decode(self, method_name):
        if method_name.startswith('top'):
            return 'top-tokens-' + method_name[3:]
        return method_name

    @abstractproperty
    def collection_passes(self):
        raise NotImplemented
    @abstractproperty
    def perplexity(self):
        raise NotImplemented
    @abstractproperty
    def sparsity_phi(self):
        raise NotImplemented
    @abstractproperty
    def sparsity_theta(self):
        raise NotImplemented
    @abstractproperty
    def background_tokens_ratio(self):
        raise NotImplemented
    @abstractproperty
    def kernel(self):
        raise NotImplemented
    @abstractproperty
    def top10(self):
        raise NotImplemented
    @abstractproperty
    def top100(self):
        raise NotImplemented
    @abstractproperty
    def tau_trajectories(self):
        raise NotImplemented


def _strip_parameters(score_definition):
    tokens = []
    for el in score_definition.split('-'):
        try:
            _ = float(el)
        except ValueError:
            if el[0] != '@':
                tokens.append(el)
    return '-'.join(tokens)


class ValueTracker(AbstractValueTracker):

    def __init__(self, tracked):
        super(ValueTracker, self).__init__(tracked)
        self._index = None

    def __getattr__(self, item):
        if item.startswith('kernel'):
            return self._groups['topic-kernel-0.' + item[6:]]
        if item.startswith('sparsity_phi_'):
            try:
                _ = self._flat['sparsity-phi-@{}c'.format(item[13:])]
            except KeyError:
                raise KeyError('{} not in {}. Maybe in [{}] ??'.format(item, list(self._flat.keys()), list(self._groups.keys())))
            return self._flat['sparsity-phi-@{}c'.format(item[13:])]
        if item.startswith('background_tokens_ratio_'):
            return self._flat['background-tokens-ratio-0.' + item.split('_')[-1]]
        return None

    @property
    def collection_passes(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def top100(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def top10(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def kernel(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def background_tokens_ratio(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def sparsity_phi(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def sparsity_theta(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def perplexity(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def tau_trajectories(self):
        return super(ValueTracker, self)._query_metrics()

    @property
    def tau_trajectory_matrices_names(self):
        return self._groups['tau-trajectories'].matrices_names


class KernelSubGroup(object):
    def __init__(self, coherence_list, contrast_list, purity_list):
        self.ll = [TrackedCoherence(coherence_list), TrackedContrast(contrast_list), TrackedPurity(purity_list)]
        # super(KernelSubGroup, self).__init__([TrackedCoherence(coherence_list), TrackedContrast(contrast_list), TrackedPurity(purity_list)])

    @property
    def coherence(self):
        return self.ll[0]

    @property
    def contrast(self):
        return self.ll[1]

    @property
    def purity(self):
        return self.ll[2]

class SingleTopicGroup(KernelSubGroup):
    def __init__(self, topic_name, coherence_list, contrast_list, purity_list):
        self.name = topic_name
        super(SingleTopicGroup, self).__init__(coherence_list, contrast_list, purity_list)


class TrackedTopics(object):
    def __init__(self, topics_data):
        self._data = topics_data

    @property
    def topics(self):
        return sorted(self._data.keys())

    def __getattr__(self, item):
        if item in self._data:
            return self._data[item]
        raise AttributeError("Topic '{}' is not registered as tracked".format(item))

class TrackedKernel(TrackedTopics):
    def __init__(self, avg_coherence_list, avg_contrast_list, avg_purity_list, topic_name2elements_hash):
        """
        :param list avg_coherence:
        :param list avg_contrast:
        :param list avg_purity:
        :param dict topic_name2elements_hash: dict of dicts: inner dicts have lists as values; ie\n
            {'top_00': {'coherence': [1,2], 'contrast': [3,4], 'purity': [8,9]},\n
            'top_01': {'coherence': [0,3], 'contrast': [5,8], 'purity': [6,3]}}
        """
        self._avgs_group = KernelSubGroup(avg_coherence_list, avg_contrast_list, avg_purity_list)
        super(TrackedKernel, self).__init__({key: SingleTopicGroup(key, val['coherence'], val['contrast'], val['purity']) for key, val in list(topic_name2elements_hash.items())})
        # self._topics_groups = map(lambda x: SingleTopicGroup(x[0], x[1]['coherence'], x[1]['contrast'], x[1]['purity']), sorted(topic_name2elements_hash.items(), key=lambda x: x[0]))
        # d = {key: SingleTopicGroup(key, val['coherence'], val['contrast'], val['purity']) for key, val in topic_name2elements_hash.items()}
        # super(TrackedKernel, self).__init__([TrackedCoherence(avg_coherences), TrackedContrast(avg_contrasts), TrackedPurity(avg_purities)])

    @property
    def average(self):
        return self._avgs_group

class TrackedTopTokens(TrackedTopics):
    def __init__(self, avg_coherence, topic_name2coherence):
        """
        :param list avg_coherence:
        :param dict topic_name2coherence: contains coherence per topic
        """
        self._avg_coh = TrackedEntity('average_coherence', avg_coherence)
        super(TrackedTopTokens, self).__init__({key: TrackedEntity(key, val) for key, val in list(topic_name2coherence.items())})

    @property
    def average_coherence(self):
        return self._avg_coh


class TrackedTrajectories(object):
    def __init__(self, matrix_name2elements_hash):
        self._trajs = {k: TrackedEntity(k, v) for k, v in list(matrix_name2elements_hash.items())}

    @property
    def trajectories(self):
        return sorted(list(self._trajs.items()), key=lambda x: x[0])

    @property
    def matrices_names(self):
        return sorted(self._trajs.keys())

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


class TrackedEntity(object):
    def __init__(self, name, elements_list):
        self._name = name
        self._elements = elements_list
    def __len__(self):
        return len(self._elements)
    def __getitem__(self, item):
        if item == 'all':
            return self._elements
        return self._elements[item]
    @property
    def name(self):
        return self._name
    @property
    def all(self):
        return self._elements
    @property
    def last(self):
        return self._elements[-1]

class TrackedCoherence(TrackedEntity):
    def __init__(self, elements_list):
        super(TrackedCoherence, self).__init__('coherence', elements_list)

class TrackedContrast(TrackedEntity):
    def __init__(self, elements_list):
        super(TrackedContrast, self).__init__('contrast', elements_list)

class TrackedPurity(TrackedEntity):
    def __init__(self, elements_list):
        super(TrackedPurity, self).__init__('purity', elements_list)


class SteadyTrackedItems(object):

    def __init__(self, root_dir, model_label, dataset_iterations, nb_topics, document_passes, background_topics, domain_topics, background_tokens_threshold, modalities):
        self._root_dir = root_dir
        self._model_label = model_label
        self._total_collection_passes = dataset_iterations
        self._nb_topics = nb_topics
        self._nb_doc_passes = document_passes
        self._bg_topics = background_topics
        self._dm_topics = domain_topics
        self._bg_tokens_threshold = background_tokens_threshold
        self._modalities = modalities
    def __dir__(self):
        return ['dir', 'model_label', 'dataset_iterations', 'nb_topics', 'document_passes', 'background_topics', 'domain_topics', 'background_tokens_threshold', 'modalities']
    def __str__(self):
        return '\n'.join(['{}: {}'.format(x, getattr(self, x)) for x in dir(self)])
    @property
    def dir(self):
        return self._root_dir
    @property
    def model_label(self):
        return self._model_label
    @property
    def dataset_iterations(self):
        return self._total_collection_passes

    @property
    def nb_topics(self):
        return self._nb_topics

    @property
    def document_passes(self):
        return self._nb_doc_passes

    @property
    def background_topics(self):
        return self._bg_topics

    @property
    def domain_topics(self):
        return self._dm_topics
    @property
    def background_tokens_threshold(self):
        return self._bg_tokens_threshold
    @property
    def modalities(self):
        return self._modalities


def kernel_def2_kernel(kernel_def):
    return 'kernel'+kernel_def.split('.')[-1]

def top_tokens_def2_top(top_def):
    return 'top'+top_def.split('-')[-1]


class FinalStateEntities(object):
    def __init__(self, kernel_def_2tokens_hash, top_def_2_tokens_hash, background_tokens):
        self._kernel_defs_list = sorted(kernel_def_2tokens_hash.keys())
        self._top_defs_list = sorted(top_def_2_tokens_hash.keys())
        self._bg_tokens = TokensList(background_tokens)
        for kernel_def, topic_name2tokens in list(kernel_def_2tokens_hash.items()):
            setattr(self, kernel_def2_kernel(kernel_def), TopicsTokens(topic_name2tokens))
        for top_def, topic_name2tokens in list(top_def_2_tokens_hash.items()):
            setattr(self, top_tokens_def2_top(top_def), TopicsTokens(topic_name2tokens))

    def __str__(self):
        return 'bg-tokens: {}\n'.format(len(self._bg_tokens)) + '\n'.join(['final state: {}\n{}'.format(y, getattr(self, y)) for y in self.top + self.kernels])

    @property
    def top(self):
        return [top_tokens_def2_top(x) for x in self._top_defs_list]

    @property
    def top_defs(self):
        return self._top_defs_list

    @property
    def kernels(self):
        return [kernel_def2_kernel(x) for x in self._kernel_defs_list]

    @property
    def kernel_defs(self):
        return self._kernel_defs_list

    @property
    def background_tokens(self):
        return self._bg_tokens.tokens


class TopicsTokens(object):
    def __init__(self, topic_name2tokens_hash):
        self._nb_columns = 10
        self._tokens = {topic_name: TokensList(tokens_list) for topic_name, tokens_list in list(topic_name2tokens_hash.items())}
        self._nb_topics = len(topic_name2tokens_hash)
        self._nb_rows = int(ceil((old_div(float(self._nb_topics), self._nb_columns))))
        self.__lens = {}
        for k, v in list(self._tokens.items()):
            setattr(self, k, v)
            self.__lens[k] = {'string': len(k), 'list': len(str(len(v)))}

    def __str__(self):
        return '\n'.join([self._get_row_string(x) for x in self._gen_rows()])

    def _get_row_string(self, row_topic_names):
        return '  '.join(['{}{}: {}{}'.format(
            x[1], (self._max(x[0], 'string')-len(x[1]))*' ', (self._max(x[0], 'list')-self.__lens[x[1]]['list'])*' ', len(self._tokens[x[1]])) for x in enumerate(row_topic_names)])

    def _gen_rows(self):
        i, j = 0, 0
        while i<len(self._tokens):
            _ = self._index(j)
            j += 1
            i += len(_)
            yield _

    def _max(self, column_index, entity):
        assert entity in ['string', 'list']
        return max([self.__lens[self.topics[x]][entity] for x in self._range(column_index)])

    def _range(self, column_index):
        # print 'RANGE', column_index, self._get_last_index_in_column(column_index), self._nb_columns
        return list(range(column_index, self._get_last_index_in_column(column_index), self._nb_columns))

    def _index(self, row_nb):
        return [_f for _f in [self.topics[x] if x<len(self._tokens) else None for x in range(self._nb_columns * row_nb, self._nb_columns * row_nb + self._nb_columns)] if _f]

    def _index_by_column(self, column_index):
        return [self.topics[x] for x in self._range(column_index)]

    def _get_last_index_in_column(self, column_index):
        assert column_index < self._nb_columns
        index = self._nb_rows * self._nb_columns - self._nb_columns + column_index
        if index < self._nb_topics <= index:
            index -= self._nb_columns
        return index

    @property
    def topics(self):
        return sorted(self._tokens.keys())

class TokensList(object):
    def __init__(self, tokens_list):
        assert type(tokens_list) == list
        self._tokens = tokens_list
    def __len__(self):
        return len(self._tokens)
    def __getitem__(self, item):
        return self._tokens[item]
    @property
    def tokens(self):
        return self._tokens
    def __str__(self):
        return str(self._tokens)
    def __repr__(self):
        return str(len(self._tokens))
    def __iter__(self):
        return iter(self._tokens)
    def __contains__(self, item):
        return item in self._tokens


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
        if isinstance(obj, TrackedTopTokens):
            return {'avg_coh': obj.average_coherence,
                    'topics': {topic_name: obj.__getattr__(topic_name).all for topic_name in obj.topics}}
        if isinstance(obj, TrackedKernel):
            return {'avg_coh': obj.average.coherence,
                    'avg_con': obj.average.contrast,
                    'avg_pur': obj.average.purity,
                    'topics': {topic_name: {'coherence': obj.__getattr__(topic_name).coherence.all,
                                            'contrast': obj.__getattr__(topic_name).contrast.all,
                                            'purity': obj.__getattr__(topic_name).purity.all} for topic_name in obj.topics}}
        if isinstance(obj, ValueTracker):
            _ = {name: tracked_entity for name, tracked_entity in list(obj._flat.items())}
            _['top-tokens'] = {k: obj._groups['top-tokens-' + str(k)] for k in obj.top_tokens_cardinalities}
            _['topic-kernel'] = {k: obj._groups['topic-kernel-' + str(k)] for k in obj.kernel_thresholds}
            _['tau-trajectories'] = {k: obj.tau_trajectories.__getattribute__(k) for k in obj.tau_trajectory_matrices_names}
            return _
        if isinstance(obj, ExperimentalResults):
            return {'scalars': obj.scalars,
                    'tracked': obj.tracked,
                    'final': obj.final,
                    'regularizers': obj.regularizers}
        return super(RoundTripEncoder, self).default(obj)


class RoundTripDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        return obj
