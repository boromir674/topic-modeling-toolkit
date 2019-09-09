import os
import pytest

from topic_modeling_toolkit.reporting import ModelReporter

@pytest.fixture(scope='module')
def reporter(collections_root_dir):
    return ModelReporter(collections_root_dir, results_dir_name='results')


class TestReporterFunctionality(object):

    def test_experimental_results_retrieval(self, reporter, test_dataset, tuner_obj):
        results_handler = reporter.results_handler
        exp_res_objs = results_handler.get_experimental_results(test_dataset.name)
        assert 1 < len(exp_res_objs)
        # exp_res_obj1 = results_handler.get_experimental_results(test_dataset.name)
        reporter._collection_name = test_dataset.name
        # reporter_exp_res =
        assert exp_res_objs == results_handler.get_experimental_results(test_dataset.name) == reporter.exp_results

    def test_string(self, reporter, test_dataset, tuner_obj):
        COLUMNS = [
            # 'nb-topics',
            # 'collection-passes',
            # 'document-passes',
            # 'total-phi-updates',
            'perplexity',
            'kernel-size',
            'kernel-coherence',
            'kernel-contrast',
            'kernel-purity',
            'top-tokens-coherence',
            'sparsity-phi',
            'sparsity-theta',
            'background-tokens-ratio',
            # 'regularizers'
        ]

        s = reporter.get_formatted_string(test_dataset.name, columns=COLUMNS, metric='perplexity', verbose=False)
        assert 5 == len(s.strip().split('\n'))
