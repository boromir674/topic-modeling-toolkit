import sys
from abc import ABCMeta, abstractproperty

class ExperimentalResults(object):
    def __init__(self, root_dir, model_label, nb_topics, document_passes, background_topics, domain_topics, modalities, tracked):
        """
        :param patm.modeling.experiment.Experiment experiment:
        """
        # self._steady_container = SteadyTrackedItems(sum(experiment.collection_passes), experiment.topic_model.label,
        #                                             experiment.topic_model.nb_topics, experiment.topic_model.document_passes,
        #                                             experiment.topic_model.background_topics, experiment.topic_model.domain_topics,
        #                                             experiment.topic_model.modalities_dictionary)
        self._steady_container = SteadyTrackedItems(root_dir, model_label, nb_topics, document_passes, background_topics, domain_topics, modalities)
        self._tracker = ValueTracker(tracked)

    @property
    def scalars(self):
        return self._steady_container

    @property
    def tracked(self):
        return self._tracker


class AbstractValueTracker(object):
    __metaclass__ = ABCMeta
    def __init__(self, tracked):
        self._flat = {}
        self._groups = {}
        self._metrics = {}
        for k, v in tracked.items():  # assumes maximum depth is 2
            if k == 'kernel':
                self._groups[k] = TrackedKernel(*v)
            elif k in ('top10', 'top100'):
                self._groups[k] = TrackedTopTokens(*v)
            elif k == 'tau_trajectories':
                self._groups[k] = TrackedTrajectories(v)
            # if type(v) == dict:
            #     tracked_entities = []
            #     for ink, inv in v.items():
            #         tracked_entities.append(TrackedEntity(ink.replace('-', '_'), inv))
            #     self._metrics[k.replace('-', '_')] = TrackedGroup(tracked_entities)
            else:
                self._flat[k.replace('-', '_')] = TrackedEntity(k, v)
                # self._metrics[k.replace('-', '_')] = TrackedEntity(k, v)

    def _query_metrics(self):
        _ = sys._getframe(1).f_code.co_name
        if _ in self._flat:
            return self._flat[_]
        if _ in self._groups:
            return self._groups[_]
        raise AttributeError

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


class ValueTracker(AbstractValueTracker):

    def __init__(self, tracked):
        super(ValueTracker, self).__init__(tracked)
        self._index = None

    def __iter__(self):
        return iter(self._metrics.keys())

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

# class TrackedGroup(object):
#     def __init__(self, tracked_entities_list):
#         self._tracked_entities = tracked_entities_list
#     def __getattr__(self, item):
#         for tracked_entity in self._tracked_entities:
#             if item == tracked_entity.name:
#                 return tracked_entity
#         raise AttributeError
#     def __iter__(self):
#         for tr_ent in self._tracked_entities:
#             yield tr_ent.name
#     def __dir__(self):
#         return map(lambda x: x.name, self._tracked_entities)


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


class TrackedKernel(object):
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
        self._topics_groups = map(lambda x: SingleTopicGroup(x[0], x[1]['coherence'], x[1]['contrast'], x[1]['purity']), sorted(topic_name2elements_hash.items(), key=lambda x: x[0]))
        # super(TrackedKernel, self).__init__([TrackedCoherence(avg_coherences), TrackedContrast(avg_contrasts), TrackedPurity(avg_purities)])

    @property
    def average(self):
        return self._avgs_group

    def __getattr__(self, item):
        for topic_group in self._topics_groups:
            if topic_group.name == item:
                return topic_group
        raise AttributeError("Topic named as '{}' is not registered in the kernel tracked entities".format(item))

class TrackedTopTokens(object):
    def __init__(self, avg_coherence, topic_name2coherence):
        """
        :param list avg_coherence:
        :param dict topic_name2coherence: contains coherence per topic
        """
        self._avg_coh = TrackedEntity('average_coherence', avg_coherence)
        self._topics_coh = map(lambda x: TrackedEntity(x[0], x[1]), sorted(topic_name2coherence.items(), key=lambda x: x[0]))

    @property
    def average_coherence(self):
        return self._avg_coh

    def __getattr__(self, item):
        for tracked_entity in self._topics_coh:
            if tracked_entity.name == item:
                return tracked_entity
        raise AttributeError("Topic named as '{}' not found as registered".format(item))

class TrackedTrajectories(object):
    def __init__(self, matrix_name2elements_hash):
        self._trajs = {k: TrackedEntity(k, v) for k, v in matrix_name2elements_hash.items()}

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

    top10_data = [[1, 2], {'t01': [1,2,3], 't00': [10,2,3], 't02': [10,11]}]
    tau_trajectories_data = {'phi': [1,2,3,4,5], 'theta': [5,6,7,8,9]}

    tracked_kernel = TrackedKernel(*kernel_data)
    assert tracked_kernel.average.contrast.all == [3, 4]
    assert tracked_kernel.average.purity.last == 6
    assert tracked_kernel.average.coherence.all == [1, 2]
    assert tracked_kernel.t00.contrast.all == [67, 36]
    assert tracked_kernel.t02.purity.all == [17, 856]
    assert tracked_kernel.t01.coherence.last == 3

    tracked = {'perplexity': [1, 2, 3], 'kernel': kernel_data, 'top10': top10_data, 'tau_trajectories': tau_trajectories_data}
    exp = ExperimentalResults(_dir, label, topics, doc_passes, bg_t, dm_t, mods, tracked)

    assert exp.scalars.nb_topics == 5
    assert exp.scalars.document_passes == 2

    assert exp.tracked.perplexity.last == 3
    assert exp.tracked.perplexity.all == [1, 2, 3]
    assert exp.tracked.kernel.average.purity.all == [5, 6]
    assert exp.tracked.kernel.t02.coherence.all == [10, 11]
    assert exp.tracked.kernel.t02.purity.all == [17, 856]
    assert exp.tracked.kernel.t02.contrast.last == 32
    assert exp.tracked.kernel.average.coherence.all == [1, 2]
    assert exp.tracked.top10.t01.all == [1, 2, 3]
    assert exp.tracked.top10.t00.last == 3
    assert exp.tracked.top10.average_coherence.all == [1, 2]

    assert exp.tracked.tau_trajectories.phi.all == [1,2,3,4,5]
    assert exp.tracked.tau_trajectories.theta.last == 9

    try:
        _ = exp.tracked.tau_trajectories.gav.last
    except AttributeError as e:
        pass

    print 'ALL ASSERTIONS SUCCEEDED'

if __name__ == '__main__':
    test()
