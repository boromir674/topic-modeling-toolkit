import os
import json
import unittest
from random import randint
# import patm
from patm.modeling.experimental_results import ExperimentalResults, TrackedKernel, ExperimentalResultsFactory, RoundTripEncoder, RoundTripDecoder
from patm.modeling import Experiment
from patm.definitions import COLLECTIONS_DIR_PATH, DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME
from patm.modeling import trainer_factory

# Random order for tests runs. (Original is: -1 if x<y, 0 if x==y, 1 if x>y).
unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)

# unittest.TestLoader.sortTestMethodsUsing = None

# CONSTANTS
my_dir = os.path.dirname(os.path.realpath(__file__))

test_json_path = os.path.join(my_dir, 'test-file')
test_collection = 'test-dataset'
test_model = 'test-model'
test_collection_root = os.path.join(COLLECTIONS_DIR_PATH, test_collection)


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

kernel_data1 = [[10,20], [30,40], [50,6], {'t01': {'coherence': [3, 9],
                                                             'contrast': [96, 3],
                                                             'purity': [1, 98]},
                                                     't00': {'coherence': [19,2,93],
                                                             'contrast': [7, 3],
                                                             'purity': [2, 89]},
                                                     't02': {'coherence': [0,11],
                                                             'contrast': [66, 32],
                                                             'purity': [17, 85]}
                                                     }]

top10_data = [[1, 2], {'t01': [12,22,3], 't00': [10,2,3], 't02': [10,11]}]
top100_data = [[10, 20], {'t01': [5,7,9], 't00': [12,32,3], 't02': [11,1]}]
tau_trajectories_data = {'phi': [1,2,3,4,5], 'theta': [5,6,7,8,9]}
tracked = {'perplexity': [1, 2, 3],
           'sparsity-phi-@dc': [-2, -4, -6],
           'sparsity-phi-@ic': [-56, -12, -32],
           'sparsity-theta': [2, 4, 6],
           'background-tokens-ratio-0.3': [0.4, 0.3, 0.2],
           'topic-kernel-0.6': kernel_data,
           'topic-kernel-0.8': kernel_data1,
           'top-tokens-10': top10_data,
           'top-tokens-100': top100_data,
           'tau-trajectories': tau_trajectories_data,
           'collection_passes': [3]}


kernel_tokens = {'topic-kernel-0.6': {'t00': ['a', 'b', 'c'],
                                      't01': ['d', 'e', 'f'],
                                      't02': ['g', 'h', 'i']},
                 'topic-kernel-0.8': {'t00': ['j', 'k', 'l'],
                                      't01': ['m', 'n', 'o'],
                                      't02': ['p', 'q', 'r']}}

top_tokens = {'top-tokens-10': {'t00': ['s', 't', 'u'],
                                't01': ['v', 'x', 'y'],
                                't02': ['z', 'a1', 'b1']},
              'top-tokens-100': {'t00': ['c1', 'd1', 'e1'],
                                 't01': ['f1', 'g1', 'h1'],
                                 't02': ['i1', 'j1', 'k1']}}

background_tokens = ['l1', 'm1', 'n1']
regs = ['reg1', 'reg2']


exp1 = ExperimentalResults(_dir, label, topics, doc_passes, bg_t, dm_t, mods, tracked, kernel_tokens, top_tokens, background_tokens, regs)
results_factory = ExperimentalResultsFactory()


def setUpModule():
    exp1.save_as_json(test_json_path)


def tearDownModule():
    os.remove(test_json_path)


class TestExperimentalResults(unittest.TestCase):
    """Unittest."""

    maxDiff, __slots__ = None, ()

    def setUp(self):
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        self.tracked_kernel = TrackedKernel(*kernel_data)
        self.eval_def2eval_name = {'perplexity': 'per',
                                   'sparsity-phi-@dc': 'sppd',
                                   'sparsity-phi-@ic': 'sppi',
                                   'sparsity-theta': 'spt',
                                   'topic-kernel-0.6': 'tk1',
                                   'topic-kernel-0.8': 'tk2',
                                   'top-tokens-10': 'tt10',
                                   'top-tokens-100': 'tt100',
                                   'background-tokens-ratio-0.3': 'btr'}
        self.reg_type2name = {'sparse-phi': 'spp', 'smooth-phi': 'smp', 'sparse-theta': 'spt', 'smooth-theta': 'smt'}
        self.nb_topics = 5
        self.collection_passes = 20

    def tearDown(self):
        """Method to tear down the test fixture. Run AFTER the test methods."""
        pass

    def addCleanup(self, function, *args, **kwargs):
        """Function called AFTER tearDown() to clean resources used on test."""
        pass

    @classmethod
    def setUpClass(cls):
        """Class method called BEFORE tests in an individual class run. """
        pass  # Probably you may not use this one. See setUp().

    @classmethod
    def tearDownClass(cls):
        """Class method called AFTER tests in an individual class run. """
        pass  # Probably you may not use this one. See tearDown().

    def test_tracked_kernel(self):
        assert self.tracked_kernel.average.contrast.all == [3, 4]
        assert self.tracked_kernel.average.purity.last == 6
        assert self.tracked_kernel.average.coherence.all == [1, 2]
        assert self.tracked_kernel.t00.contrast.all == [67, 36]
        assert self.tracked_kernel.t02.purity.all == [17, 856]
        assert self.tracked_kernel.t01.coherence.last == 3

    def test_constructor(self):
        self.exp1 = exp1
        self._assert_experimental_results_creation()

    def test_json_construction(self):
        json_string = exp1.to_json()
        res = json.loads(json_string, cls=RoundTripDecoder)
        assert res['scalars']['dir'] == 'dir'
        assert res['scalars']['domain_topics'] == ['t2', 't3', 't4']
        assert res['scalars']['modalities'] == {'dcn': 1, 'icn': 5}
        assert res['tracked']['perplexity'] == [1, 2, 3]
        assert res['tracked']['top-tokens']['10']['avg_coh'] == [1, 2]
        assert res['tracked']['top-tokens']['10']['topics']['t01'] == [12, 22, 3]
        assert res['tracked']['top-tokens']['10']['topics']['t02'] == [10, 11]
        assert res['tracked']['topic-kernel']['0.6']['avg_pur'] == [5, 6]
        assert res['tracked']['topic-kernel']['0.6']['topics']['t00']['purity'] == [12, 89]
        assert res['tracked']['topic-kernel']['0.6']['topics']['t01']['contrast'] == [6, 3]
        assert res['tracked']['topic-kernel']['0.6']['topics']['t02']['coherence'] == [10, 11]
        assert res['tracked']['background-tokens-ratio-0.3'] == [0.4, 0.3, 0.2]
        assert res['tracked']['tau-trajectories']['phi'] == [1, 2, 3, 4, 5]
        assert res['tracked']['tau-trajectories']['theta'] == [5, 6, 7, 8, 9]
        assert res['final']['topic-kernel']['0.6']['t00'] == ['a', 'b', 'c']
        assert res['final']['topic-kernel']['0.6']['t02'] == ['g', 'h', 'i']
        assert res['final']['topic-kernel']['0.8']['t01'] == ['m', 'n', 'o']
        assert res['final']['top-tokens']['10']['t02'] == ['z', 'a1', 'b1']
        assert res['final']['top-tokens']['100']['t00'] == ['c1', 'd1', 'e1']
        assert res['final']['top-tokens']['100']['t01'] == ['f1', 'g1', 'h1']
        assert res['final']['background-tokens'] == ['l1', 'm1', 'n1']

    def test_creation_from_json(self):
        self.exp1 = results_factory.create_from_json_file(test_json_path)
        # print self.exp1.to_json(human_redable=True)  # ExperimentalResults(_dir, label, topics, doc_passes, bg_t, dm_t, mods, tracked)
        self._assert_experimental_results_creation()

    def test_creation_from_experiment(self):
        model_trainer = trainer_factory.create_trainer(test_collection, exploit_ideology_labels=True, force_new_batches=False)
        experiment = Experiment(test_collection_root, model_trainer.cooc_dicts)
        model_trainer.register(experiment)  # when the model_trainer trains, the experiment object listens to changes
        topic_model = model_trainer.model_factory.construct_model(test_model, self.nb_topics, self.collection_passes, 1, 0.25, {IDEOLOGY_CLASS_NAME: 5, DEFAULT_CLASS_NAME: 1}, self.eval_def2eval_name, self.reg_type2name)
        train_specs = model_trainer.model_factory.create_train_specs()
        experiment.init_empty_trackables(topic_model)
        model_trainer.train(topic_model, train_specs, effects=False)
        exp1 = results_factory.create_from_experiment(experiment)
        dom = topic_model.domain_topics

        assert hasattr(exp1, 'tracked')
        assert hasattr(exp1, 'scalars')
        assert exp1.scalars.nb_topics == self.nb_topics
        assert exp1.scalars.model_label == 'test-model'
        assert all(map(lambda x: len(x) == self.collection_passes, [exp1.tracked.perplexity, exp1.tracked.sparsity_theta, exp1.tracked.sparsity_phi_d, exp1.tracked.sparsity_phi_i]))
        for reg_def in (_ for _ in self.eval_def2eval_name if _.startswith('topic-kernel-')):
            tr_kernel = getattr(exp1.tracked, 'kernel'+reg_def.split('-')[-1][2:])
            assert all(map(lambda x: len(getattr(tr_kernel.average, x)) == self.collection_passes, ['coherence', 'contrast', 'purity']))
        assert all(map(lambda x: len(x.average_coherence) == self.collection_passes,
                       (getattr(exp1.tracked, 'top' + _.split('-')[-1]) for _ in self.eval_def2eval_name if _.startswith('top-tokens-'))))
        assert all(map(lambda x: len(x.all) == self.collection_passes,
                       (getattr(exp1.tracked, 'sparsity_phi_' + _.split('-')[-1][1]) for _ in self.eval_def2eval_name if
                        _.startswith('sparsity-phi-@'))))
        # for ed, en in (_ for _ in self.eval_def2eval_name.items() if _[0].startswith('top-tokens-')):
        #     print ed, en, '[{}]'.format(', '.join(map(lambda x: str(x), getattr(exp1.tracked, 'top'+ed.split('-')[-1]).average_coherence)))
        #
        # for ed, en in (_ for _ in self.eval_def2eval_name.items() if _[0].startswith('topic-kernel-')):
        #     print ed, en, '[{}]'.format(', '.join(map(lambda x: str(x), getattr(exp1.tracked, 'kernel'+ed.split('-')[-1][2:]).average.coherence)))

        for reg_def in (_ for _ in self.eval_def2eval_name if _.startswith('top-tokens-')):
            tr_top = getattr(exp1.tracked, 'top' + reg_def.split('-')[-1])
            assert len(tr_top.average_coherence) == self.collection_passes

    #         # assert abs(tr_kernel.average.coherence.last - sum(map(lambda x: getattr(tr_kernel, x).coherence.last, dom)) / float(self.nb_topics)) < 0.001
    #
        for kernel, kernel_def in zip(exp1.final.kernels, exp1.final.kernel_defs):
            for topic_name in exp1.scalars.domain_topics:
                # print dir(getattr(exp1.final, kernel))
                # print getattr(getattr(exp1.final, kernel), topic_name).tokens
                # print experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[kernel_def]].tokens[-1]
                assert getattr(getattr(exp1.final, kernel), topic_name).tokens == experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[kernel_def]].tokens[-1][topic_name]
    #
    #     for top, top_def in zip(exp1.final.top, exp1.final.top_defs):
    #         for topic_name in exp1.scalars.domain_topics:
    #             print dir(getattr(exp1.final, top))
    #             # print experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[kernel_def]].tokens[-1]
    #             assert getattr(getattr(exp1.final, top), topic_name).tokens == experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[top_def]].tokens[-1][topic_name]

    def _assert_experimental_results_creation(self):
        assert self.exp1.scalars.nb_topics == 5
        assert self.exp1.scalars.document_passes == 2
        assert self.exp1.tracked.perplexity.last == 3
        assert self.exp1.tracked.perplexity.all == [1, 2, 3]
        assert self.exp1.tracked.sparsity_phi_d.all == [-2, -4, -6]
        assert self.exp1.tracked.sparsity_phi_i.all == [-56, -12, -32]
        assert self.exp1.tracked.sparsity_theta.last == 6

        assert self.exp1.tracked.kernel6.average.purity.all == [5, 6]
        assert self.exp1.tracked.kernel6.t02.coherence.all == [10, 11]
        assert self.exp1.tracked.kernel6.t02.purity.all == [17, 856]
        assert self.exp1.tracked.kernel6.t02.contrast.last == 32
        assert self.exp1.tracked.kernel6.average.coherence.all == [1, 2]
        assert self.exp1.tracked.kernel8.average.purity.all == [50, 6]
        assert self.exp1.tracked.kernel8.t02.coherence.last == 11
        assert self.exp1.tracked.kernel8.t02.purity.all == [17, 85]
        assert self.exp1.tracked.kernel8.t00.contrast.last == 3
        assert self.exp1.tracked.kernel8.average.purity.last == 6

        assert self.exp1.tracked.top10.t01.all == [12, 22, 3]
        assert self.exp1.tracked.top10.t00.last == 3
        assert self.exp1.tracked.top10.average_coherence.all == [1, 2]

        assert self.exp1.tracked.tau_trajectories.phi.all == [1, 2, 3, 4, 5]
        assert self.exp1.tracked.tau_trajectories.theta.last == 9

        assert self.exp1.final.kernel6.t00.tokens == ['a', 'b', 'c']
        assert self.exp1.final.kernel6.t02.tokens == ['g', 'h', 'i']
        assert self.exp1.final.kernel8.topics == ['t00', 't01', 't02']
        assert self.exp1.final.top10.t02.tokens == ['z', 'a1', 'b1']
        assert self.exp1.final.top100.t00.tokens == ['c1', 'd1', 'e1']
        assert self.exp1.final.top100.t01.tokens == ['f1', 'g1', 'h1']
        assert len(self.exp1.final.top100.t02) == 3
        assert 'j1' in self.exp1.final.top100.t02
        assert self.exp1.final.top_defs == ['top-tokens-10', 'top-tokens-100']
        assert self.exp1.final.kernels == ['kernel6', 'kernel8']
        assert self.exp1.final.background_tokens == ['l1', 'm1', 'n1']

        assert self.exp1.tracked.background_tokens_ratio_3.all ==  [0.4, 0.3, 0.2]
        assert self.exp1.tracked.background_tokens_ratio_3.last == 0.2

if __name__.__contains__("__main__"):

    unittest.main()
    # Run just 1 test.
    # unittest.main(defaultTest='TestExperimentalResults.test_tracked_kernel', warnings='ignore')
