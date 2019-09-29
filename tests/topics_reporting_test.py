import pytest
import os
from topic_modeling_toolkit.reporting import TopicsHandler


@pytest.fixture(scope='module')
def topics_handler(collections_root_dir, exp_res_obj1, trained_model_n_experiment, tuner_obj):
    return TopicsHandler(collections_root_dir, results_dir_name='results')


@pytest.fixture(scope='module')
def domain_string_first_token_line():
    return '65.7% 92.6% 27.3%     51.8% 91.9% 27.4%     50.0% 92.3% 27.8%  44.4% 88.6% 23.1%  43.4% 91.3% 24.9%  39.3% 90.1% 29.5%'


@pytest.fixture(scope='module')
def background_string_first_token_line():
    return 'able          '


@pytest.fixture(scope='module')
def test_model_string():
    return 'unittest_10_0.2_4_1_1'

# 'unittest_4_1_0.2_1_1_10
class TestTopicsHandler(object):
    def test_domain_strings(self, test_collection_dir, topics_handler, test_collection_name, domain_string_first_token_line, test_model_string):
        print(os.listdir(os.path.join(test_collection_dir, 'results')))
        b = topics_handler.pformat([test_collection_name, test_model_string],
                                   'domain',
                                   'top-tokens',
                                   'coh-80',
                                   10,
                                   6,
                                   topic_info=True,
                                   show_title=False)
        b1 = b.split('\n')[1]
        assert b1 == domain_string_first_token_line

    def test_background_strings(self, topics_handler, test_collection_name, background_string_first_token_line, test_model_string):
        b = topics_handler.pformat_background([test_collection_name, test_model_string],
                                             columns=8,
                                             nb_tokens=20,
                                             show_title=True)

        b1 = b.split('\n')
        assert '+'.join(x for x in b1[2:3]) == background_string_first_token_line
