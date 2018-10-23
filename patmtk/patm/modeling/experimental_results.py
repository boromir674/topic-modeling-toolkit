import sys
import json
from abc import ABCMeta, abstractproperty


import json, datetime
from dateutil import parser




class ExperimentalResults(object):
    def __init__(self, root_dir, model_label, nb_topics, document_passes, background_topics, domain_topics, modalities, tracked):  #, kernel_tokens, top_tokens_defs, background_tokens):
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
        self._steady_container = SteadyTrackedItems(root_dir, model_label, nb_topics, document_passes, background_topics, domain_topics, modalities)
        self._tracker = ValueTracker(tracked)

    @property
    def scalars(self):
        return self._steady_container

    @property
    def tracked(self):
        return self._tracker

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

class ExperimentalResultsFactory(object):
    def __init__(self):
        self._data = []

    def create_from_json_file(self, file_path):
        with open(file_path, 'r') as fp:
            res = json.load(fp, cls=RoundTripDecoder)
        # self._data['modality-phi-sparsities'] = {key: value for key, value in res['tracked'].items() if key.startswith('sparsity-phi-')}
        self._data =[{'perplexity': res['tracked']['perplexity'],
                      'sparsity-theta': res['tracked']['sparsity-theta']}]
        self._data.append({'tau-trajectories': res['tracked']['tau-trajectories']})
        self._data.append({'topic-kernel-' + key: [value['avg_coh'], value['avg_con'], value['avg_pur'],
                                                                  {t_name: t_data for t_name, t_data in
                                                                   value['topics'].items()}] for key, value in
                                          res['tracked']['topic-kernel'].items()})
        self._data.append({'top-tokens-' + key: [value['avg_coh'], {t_name: t_data for t_name, t_data in value['topics'].items()}] for key, value in res['tracked']['top-tokens'].items()})
        self._data.append({key: v for key, v in res['tracked'].items() if key.startswith('sparsity-phi-@')})
        return ExperimentalResults(res['scalars']['dir'],
                                   res['scalars']['label'],
                                   res['scalars']['nb_topics'],
                                   res['scalars']['document_passes'],
                                   res['scalars']['background_topics'],
                                   res['scalars']['domain_topics'],
                                   res['scalars']['modalities'],
                                   reduce(lambda x, y: dict(x, **y), self._data))

    def create_from_experiment(self, experiment):
        self._data = [{'perplexity': experiment.trackables['perplexity'],
                       'sparsity-theta': experiment.trackables['sparsity-theta']}]
        self._data.append({key: [value['avg_coh'], value['avg_con'], value['avg_pur'], {t_name: t_data for t_name, t_data in value['topics'].items()}] for key, value in experiment.trackables.items() if key.startswith('topic-kernel')})
        self._data.append({top_tokens_definition: [value['avg_coh'], {t_name: t_data for t_name, t_data in value['coherence'].items()}] for top_tokens_definition, value in experiment.trackables.items() if top_tokens_definition.startswith('top-tokens-')})

        #     ['top_tokens_data_hash'] = {
        # 'top-tokens-' + key: [value['avg_coh'], {t_name: t_data for t_name, t_data in value['topics'].items()}] for
        # key, value in res['tracked']['top-tokens'].items()}
        # self._data['tau_values_data_hash'] = res['tracked']['tau-trajectories']

        return ExperimentalResults(experiment.current_root_dir,
                                   experiment.topic_model.label,
                                   experiment.topic_model.nb_topics,
                                   experiment.topic_model.document_passes,
                                   experiment.topic_model.background_topics,
                                   experiment.topic_model.domain_topics,
                                   experiment.topic_model.modalities_dictionary,
                                   reduce(lambda x, y: dict(x, **y), self._data.values()))


class AbstractValueTracker(object):
    __metaclass__ = ABCMeta

    def __str__(self):
        return '[{}]'.format(', '.join(list(self._flat.keys()) + list(self._groups)))

    def __init__(self, tracked):
        self._flat = {}
        self._groups = {}
        for evaluator_definition, v in tracked.items():  # assumes maximum depth is 2
            score_type = _strip_parameters(evaluator_definition.replace('_', '-'))
            if score_type == 'topic-kernel':
                self._groups[evaluator_definition] = TrackedKernel(*v)
            elif score_type == 'top-tokens':
                self._groups[evaluator_definition] = TrackedTopTokens(*v)
            elif score_type == 'tau-trajectories':
                self._groups[score_type] = TrackedTrajectories(v)
            else:
                self._flat[evaluator_definition.replace('_', '-')] = TrackedEntity(score_type, v)
                # self._metrics[score_type.replace('-', '_')] = TrackedEntity(score_type, v)
            # if type(v) == dict:
            #     tracked_entities = []
            #     for ink, inv in v.items():
            #         tracked_entities.append(TrackedEntity(ink.replace('-', '_'), inv))
            #     self._metrics[score_type.replace('-', '_')] = TrackedGroup(tracked_entities)

        print 'Created ValueTracker', self
        print self._groups['tau-trajectories']

    @property
    def top_tokens_cardinalities(self):
        return sorted(int(_.split('-')[-1]) for _ in self._groups.keys() if _.startswith('top-tokens'))

    @property
    def kernel_thresholds(self):
        return sorted(float(_.split('-')[-1]) for _ in self._groups.keys() if _.startswith('topic-kernel'))

    @property
    def tracked_entity_names(self):
        return sorted(self._flat.keys() + self._groups.keys())

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
    def perplexity(self):
        raise NotImplemented
    @abstractproperty
    def sparsity_phi(self):
        raise NotImplemented
    @abstractproperty
    def sparsity_theta(self):
        raise NotImplemented
    @abstractproperty
    def background_token_ratio(self):
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

    def __iter__(self):
        return iter(self._metrics.keys())

    def __getattr__(self, item):
        if item.startswith('kernel'):
            return self._groups['topic-kernel-0.' + item[6:]]
        if item.startswith('sparsity_phi_'):
            try:
                _ = self._flat['sparsity-phi-@{}c'.format(item[13:])]
            except KeyError:
                raise KeyError('{} not in {}. Maybe in [{}] ??'.format(item, self._flat.keys(), self._groups.keys()))
            return self._flat['sparsity-phi-@{}c'.format(item[13:])]
        return None

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
    def background_token_ratio(self):
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
        super(TrackedKernel, self).__init__({key: SingleTopicGroup(key, val['coherence'], val['contrast'], val['purity']) for key, val in topic_name2elements_hash.items()})
        # self._topics_groups = map(lambda x: SingleTopicGroup(x[0], x[1]['coherence'], x[1]['contrast'], x[1]['purity']), sorted(topic_name2elements_hash.items(), key=lambda x: x[0]))
        # d = {key: SingleTopicGroup(key, val['coherence'], val['contrast'], val['purity']) for key, val in topic_name2elements_hash.items()}
        # super(TrackedKernel, self).__init__([TrackedCoherence(avg_coherences), TrackedContrast(avg_contrasts), TrackedPurity(avg_purities)])

    @property
    def average(self):
        return self._avgs_group

    # def __getattr__(self, item):
    #     for topic_group in self._topics_groups:
    #         if topic_group.name == item:
    #             return topic_group
    #     raise AttributeError("Topic named as '{}' is not registered in the kernel tracked entities".format(item))

class TrackedTopTokens(TrackedTopics):
    def __init__(self, avg_coherence, topic_name2coherence):
        """
        :param list avg_coherence:
        :param dict topic_name2coherence: contains coherence per topic
        """
        self._avg_coh = TrackedEntity('average_coherence', avg_coherence)
        # self._topics_coh = map(lambda x: TrackedEntity(x[0], x[1]), topic_name2coherence.items())
        super(TrackedTopTokens, self).__init__({key: TrackedEntity(key, val) for key, val in topic_name2coherence.items()})

    @property
    def average_coherence(self):
        return self._avg_coh

    # def __getattr__(self, item):
    #     for tracked_entity in self._topics_coh:
    #         if tracked_entity.name == item:
    #             return tracked_entity
    #     raise AttributeError("Topic named as '{}' not found as registered".format(item))

class TrackedTrajectories(object):
    def __init__(self, matrix_name2elements_hash):
        self._trajs = {k: TrackedEntity(k, v) for k, v in matrix_name2elements_hash.items()}

    @property
    def trajectories(self):
        return sorted(self._trajs.items(), key=lambda x: x[0])

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

    def __init__(self, root_dir, model_label, nb_topics, document_passes, background_topics, domain_topics, modalities):
        self._root_dir = root_dir
        self._model_label = model_label
        self._nb_topics = nb_topics
        self._nb_doc_passes = document_passes
        self._bg_topics = background_topics
        self._dm_topics = domain_topics
        self._modalities = modalities
    @property
    def dir(self):
        return self._root_dir
    @property
    def model_label(self):
        return self._model_label

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
    def modalities(self):
        return self._modalities

def test():
    _dir = 'dir'
    label = 'gav'
    topics = 5
    doc_passes = 2
    bg_t = ['t0', 't1']
    dm_t = ['t2', 't3', 't4']
    mods = {'dcn': 1, 'icn': 5}

    tr_t = TrackedTopTokens([1,2,3], {'t00': [4,5,6], 't01': [4,5,5]})

    assert tr_t.average_coherence.all == [1,2,3]
    assert tr_t.average_coherence.last == 3
    assert tr_t.t00.all == [4,5,6]
    assert tr_t.t01.last == 5

    kernel_data = [[1,2], [3,4], [5,6], {'t01': {'coherence': [1,2,3],
                                                                 'contrast': [6, 3],
                                                                 'purity': [1, 8]},
                                                         't00': {'coherence': [10,2,3],
                                                                 'contrast': [67, 36],
                                                                 'purity': [12, 89]},
                                                         't02': {'coherence': [10,11],
                                                                 'contrast': [656, 32],
                                                                 'purity': [17, 856]}
                                                         }]

    top10_data = [[1, 2], {'t01': [12,22,3], 't00': [10,2,3], 't02': [10,11]}]
    top100_data = [[10, 20], {'t01': [5, 7, 9], 't00': [12, 32, 3], 't02': [11, 1]}]
    tau_trajectories_data = {'phi': [1,2,3,4,5], 'theta': [5,6,7,8,9]}

    tracked_kernel = TrackedKernel(*kernel_data)
    assert tracked_kernel.average.contrast.all == [3, 4]
    assert tracked_kernel.average.purity.last == 6
    assert tracked_kernel.average.coherence.all == [1, 2]
    assert tracked_kernel.t00.contrast.all == [67, 36]
    assert tracked_kernel.t02.purity.all == [17, 856]
    assert tracked_kernel.t01.coherence.last == 3

    tracked = {'perplexity': [1, 2, 3], 'sparsity_phi': [-2, -4, -6], 'sparsity_theta': [2, 4, 6], 'topic-kernel-0.6': kernel_data, 'top-tokens-10': top10_data, 'top-tokens-100':top100_data, 'tau-trajectories': tau_trajectories_data}
    exp = ExperimentalResults(_dir, label, topics, doc_passes, bg_t, dm_t, mods, tracked)
    #
    assert exp.scalars.nb_topics == 5
    assert exp.scalars.document_passes == 2

    assert exp.tracked.perplexity.last == 3
    assert exp.tracked.perplexity.all == [1, 2, 3]
    assert exp.tracked.sparsity_phi.all == [-2, -4, -6]
    assert exp.tracked.sparsity_theta.last == 6
    assert exp.tracked.kernel6.average.purity.all == [5, 6]
    assert exp.tracked.kernel6.t02.coherence.all == [10, 11]
    assert exp.tracked.kernel6.t02.purity.all == [17, 856]
    assert exp.tracked.kernel6.t02.contrast.last == 32
    assert exp.tracked.kernel6.average.coherence.all == [1, 2]
    assert exp.tracked.top10.t01.all == [12,22,3]
    assert exp.tracked.top10.t00.last == 3
    assert exp.tracked.top10.average_coherence.all == [1, 2]

    assert exp.tracked.tau_trajectories.phi.all == [1,2,3,4,5]
    assert exp.tracked.tau_trajectories.theta.last == 9

    try:
        _ = exp.tracked.tau_trajectories.gav.last
    except AttributeError as e:
        pass


    value_tracker = ValueTracker(tracked)
    tracked_kernel = TrackedKernel(*kernel_data)

    data = {
        # "name": "Silent Bob",
        # "dt": datetime.datetime(2013, 11, 11, 10, 40, 32),
        # 'names': ['alpha', 'beta'],
        # 'steady': SteadyTrackedItems('gav-dir', 'alpha_model', 20, 1, ['t00, t01'], ['t02', 't03', 't04'],
        #                              {'@default_class': 1, '@ideology_class': 5}),
        # 'ent1': TrackedPurity([0, 0, -1, -2, -3]),
        # 'ent3': TrackedEntity('gg-ent', [0, 0, -1, -2, -3]),
        # 'tracked_taus:': TrackedTrajectories({'phi': [0, 0, -1, -2, -3], 'theta': [0, -2, -4, -6, -8]}),
        # 'top10': TrackedTopTokens([1, 2], {'t01': [1, 2, 3], 't00': [10, 2, 3], 't02': [10, 11]}),
        # 'kernel': tracked_kernel,
        # 'value_tracker': value_tracker,
        'exp1': exp
    }

    s = json.dumps(exp, cls=RoundTripEncoder, indent=2)
    print 'S:\n', s

    res = json.loads(s, cls=RoundTripDecoder)
    print 'R: {}\n'.format(type(res)), res

    # assert res['scalars']['dir'] == 'dir'
    # assert res['scalars']['domain_topics'] == ['t2', 't3', 't4']
    # assert res['scalars']['modalities'] == {'dcn': 1, 'icn': 5}
    #
    # assert res['tracked']['perplexity'] == [1, 2, 3]
    # assert res['tracked']['top-tokens']['10']['avg_coh'] == [1, 2]
    # assert res['tracked']['top-tokens']['10']['topics']['t01'] == [12, 22, 3]
    # assert res['tracked']['top-tokens']['10']['topics']['t02'] == [10, 11]
    # assert res['tracked']['topic-kernel']['0.6']['avg_pur'] == [5, 6]
    # assert res['tracked']['topic-kernel']['0.6']['topics']['t00']['purity'] == [12, 89]
    # assert res['tracked']['topic-kernel']['0.6']['topics']['t01']['contrast'] == [6, 3]
    # assert res['tracked']['topic-kernel']['0.6']['topics']['t02']['coherence'] == [10, 11]


    print 'ALL ASSERTIONS SUCCEEDED'

class RoundTripEncoder(json.JSONEncoder):
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:%S"
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return {
                "_type": "datetime",
                "value": obj.strftime("%s %s" % (
                    self.DATE_FORMAT, self.TIME_FORMAT
                ))
            }
        if isinstance(obj, SteadyTrackedItems):
            return {'dir': obj.dir,
                    'label': obj.model_label,
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
            _ = {name: tracked_entity for name, tracked_entity in obj._flat.items()}
            _['top-tokens'] = {k: obj._groups['top-tokens-' + str(k)] for k in obj.top_tokens_cardinalities}
            _['topic-kernel'] = {k: obj._groups['topic-kernel-' + str(k)] for k in obj.kernel_thresholds}
            _['tau-trajectories'] = {k: obj.tau_trajectories.__getattribute__(k) for k in obj.tau_trajectory_matrices_names}
            return _
        if isinstance(obj, ExperimentalResults):
            return {'scalars': obj.scalars,
                    'tracked': obj.tracked}
        return super(RoundTripEncoder, self).default(obj)


class RoundTripDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        self._data = {}
    def object_hook(self, obj):
        return obj


if __name__ == '__main__':
    test()
