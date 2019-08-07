import os
import json
import pytest


from results import ExperimentalResults
from results.experimental_results import TrackedKernel, RoundTripDecoder

from patm.modeling import Experiment




class TestRQ1ExperimentalResults(object):

    #
    # def test_constructor(self, exp_res_obj1):
    #     self._assert_experimental_results_creation(exp_res_obj1)
    #
    # def test_json_construction(self, exp_res_obj1):
    #     json_string = exp_res_obj1.to_json()
    #     res = json.loads(json_string, cls=RoundTripDecoder)
    #     assert res['scalars']['dir'] == 'a-dataset-dir'
    #     assert res['scalars']['domain_topics'] == ['t2', 't3', 't4']
    #     assert res['scalars']['modalities'] == {'dcn': 1, 'icn': 5}
    #     assert res['tracked']['perplexity'] == [1, 2, 3]
    #     assert res['tracked']['top-tokens']['10']['avg_coh'] == [5, 6, 7]
    #     assert res['tracked']['top-tokens']['10']['topics']['t01'] == [12, 22, 3]
    #     assert res['tracked']['top-tokens']['10']['topics']['t02'] == [10, 11]
    #     assert res['tracked']['topic-kernel']['0.60']['avg_pur'] == [5, 6]
    #     assert res['tracked']['topic-kernel']['0.60']['topics']['t00']['purity'] == [12, 89]
    #     assert res['tracked']['topic-kernel']['0.60']['topics']['t01']['contrast'] == [6, 3]
    #     assert res['tracked']['topic-kernel']['0.60']['topics']['t02']['coherence'] == [10, 11]
    #     assert res['tracked']['background-tokens-ratio-0.30'] == [0.4, 0.3, 0.2]
    #     assert res['tracked']['tau-trajectories']['phi'] == [1, 2, 3]
    #     assert res['tracked']['tau-trajectories']['theta'] == [5, 6, 7]
    #     assert res['final']['topic-kernel']['0.60']['t00'] == ['a', 'b', 'c']
    #     assert res['final']['topic-kernel']['0.60']['t02'] == ['g', 'h', 'i']
    #     assert res['final']['topic-kernel']['0.80']['t01'] == ['m', 'n', 'o']
    #     assert res['final']['top-tokens']['10']['t02'] == ['z', 'a1', 'b1']
    #     assert res['final']['top-tokens']['100']['t00'] == ['c1', 'd1', 'e1']
    #     assert res['final']['top-tokens']['100']['t01'] == ['f1', 'g1', 'h1']
    #     assert res['final']['background-tokens'] == ['l1', 'm1', 'n1']

    def test_creation_from_json(self):
        exp1 = ExperimentalResults.create_from_json_file('/data/thesis/data/collections/p29-5_a.10.0.0/results/cplsa1000_0.2_1.json')
        # self._assert_experimental_results_creation(exp1)

    # def test_creation_from_experiment(self, trained_model_n_experiment, train_settings):
    #     model, experiment = trained_model_n_experiment
    #     exp_res = ExperimentalResults.create_from_experiment(experiment)
    #     self._exp_res_obj_assertions(exp_res, experiment, train_settings)
    #     self._assert_tokens_loading_correctness(experiment, exp_res)
    #
    # # @pytest.mark.skip()
    # def test_loaded_model(self, loaded_model_n_experiment, train_settings):
    #     model, experiment = loaded_model_n_experiment
    #     exp_res = ExperimentalResults.create_from_experiment(experiment)
    #     self._exp_res_obj_assertions(exp_res, experiment, train_settings)
    #
    # def _exp_res_obj_assertions(self, exp_res, experiment, training_settings):
    #     col_passes = int(training_settings['learning']['collection_passes'])
    #
    #     assert hasattr(exp_res, 'tracked')
    #     assert hasattr(exp_res, 'scalars')
    #     assert exp_res.scalars.nb_topics == int(training_settings['learning']['nb_topics'])
    #     assert exp_res.scalars.model_label == experiment.topic_model.label
    #     assert all(map(lambda x: len(x) == col_passes,
    #                    [exp_res.tracked.perplexity, exp_res.tracked.sparsity_theta, exp_res.tracked.sparsity_phi_d,
    #                     exp_res.tracked.sparsity_phi_i]))
    #     for reg_def in (_ for _ in training_settings['scores'] if _.startswith('topic-kernel-')):
    #         tr_kernel = getattr(exp_res.tracked, 'kernel' + reg_def.split('-')[-1][2:])
    #         assert all(
    #             map(lambda x: len(getattr(tr_kernel.average, x)) == col_passes, ['coherence', 'contrast', 'purity']))
    #     assert all(map(lambda x: len(x.average_coherence) == col_passes,
    #                    (getattr(exp_res.tracked, 'top' + _.split('-')[-1]) for _ in training_settings['scores'] if
    #                     _.startswith('top-tokens-'))))
    #     assert all(map(lambda x: len(x.all) == col_passes,
    #                    (getattr(exp_res.tracked, 'sparsity_phi_' + _.split('-')[-1][1]) for _ in
    #                     training_settings['scores'] if
    #                     _.startswith('sparsity-phi-@'))))
    #
    #     for reg_def in (_ for _ in training_settings['scores'] if _.startswith('top-tokens-')):
    #         tr_top = getattr(exp_res.tracked, 'top' + reg_def.split('-')[-1])
    #         assert len(tr_top.average_coherence) == col_passes
    #
    # def _assert_tokens_loading_correctness(self, experiment, exp_res):
    #     """Requires a a model that has been trained (not just loaded) so that artm_model has correct state from score_tracker"""
    #     for kernel, kernel_def in zip(exp_res.final.kernels, exp_res.final.kernel_defs):
    #         for topic_name in exp_res.scalars.domain_topics:
    #             assert getattr(getattr(exp_res.final, kernel), topic_name).tokens == \
    #                    experiment.topic_model.artm_model.score_tracker[experiment.topic_model.definition2evaluator_name[kernel_def]].tokens[-1][topic_name]
    #
    # def _assert_experimental_results_creation(self, exp_obj):
    #     assert exp_obj.scalars.nb_topics == 40
    #     assert exp_obj.scalars.document_passes == 2
    #     assert exp_obj.tracked.perplexity.last == 3
    #     assert exp_obj.tracked.perplexity.all == [1, 2, 3]
    #     assert exp_obj.tracked.sparsity_phi_d.all == [-2, -4, -6]
    #     assert exp_obj.tracked.sparsity_phi_i.all == [-56, -12, -32]
    #     assert exp_obj.tracked.sparsity_theta.last == 6
    #
    #     assert exp_obj.tracked.kernel6.average.purity.all == [5, 6]
    #     assert exp_obj.tracked.kernel6.t02.coherence.all == [10, 11]
    #     assert exp_obj.tracked.kernel6.t02.purity.all == [17, 856]
    #     assert exp_obj.tracked.kernel6.t02.contrast.last == 32
    #     assert exp_obj.tracked.kernel6.average.coherence.all == [1, 2]
    #     assert exp_obj.tracked.kernel8.average.purity.all == [50, 6]
    #     assert exp_obj.tracked.kernel8.t02.coherence.last == 11
    #     assert exp_obj.tracked.kernel8.t02.purity.all == [17, 85]
    #     assert exp_obj.tracked.kernel8.t00.contrast.last == 3
    #     assert exp_obj.tracked.kernel8.average.purity.last == 6
    #
    #     assert exp_obj.tracked.top10.t01.all == [12, 22, 3]
    #     assert exp_obj.tracked.top10.t00.last == 3
    #     assert exp_obj.tracked.top10.average_coherence.all == [5, 6, 7]
    #
    #     assert exp_obj.tracked.tau_trajectories.phi.all == [1, 2, 3]
    #     assert exp_obj.tracked.tau_trajectories.theta.last == 7
    #
    #     assert exp_obj.final.kernel6.t00.tokens == ['a', 'b', 'c']
    #     assert exp_obj.final.kernel6.t02.tokens == ['g', 'h', 'i']
    #     assert exp_obj.final.kernel8.topics == ['t00', 't01', 't02']
    #     assert exp_obj.final.top10.t02.tokens == ['z', 'a1', 'b1']
    #     assert exp_obj.final.top100.t00.tokens == ['c1', 'd1', 'e1']
    #     assert exp_obj.final.top100.t01.tokens == ['f1', 'g1', 'h1']
    #     assert len(exp_obj.final.top100.t02) == 3
    #     assert 'j1' in exp_obj.final.top100.t02
    #     assert exp_obj.final.top_defs == ['top-tokens-10', 'top-tokens-100']
    #     assert exp_obj.final.kernels == ['kernel60', 'kernel80']
    #     assert exp_obj.final.background_tokens == ['l1', 'm1', 'n1']
    #
    #     assert exp_obj.tracked.background_tokens_ratio_3.all ==  [0.4, 0.3, 0.2]
    #     assert exp_obj.tracked.background_tokens_ratio_3.last == 0.2
