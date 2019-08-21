import pytest
from reporting.topics import TopicsHandler


@pytest.fixture(scope='module')
def topics_handler(collections_root_dir, exp_res_obj1, trained_model_n_experiment, tuner_obj):
    return TopicsHandler(collections_root_dir, results_dir_name='results')


@pytest.fixture(scope='module')
def domain_string_first_token_line():
    return '46.1% 94.7% 49.0%     37.0% 95.6% 46.8%     36.3% 93.1% 51.2%  33.0% 95.2% 52.0%  32.8% 93.9% 53.5%  25.7% 92.7% 51.6%'


@pytest.fixture(scope='module')
def background_string_first_token_line():
    return 'able          '


class TestTopicsHandler(object):
    def test_domain_strings(self, topics_handler, test_collection_name, domain_string_first_token_line):
        b = topics_handler.pformat([test_collection_name, 'unittest_1_10_0.2_5_1'],
                                   'domain',
                                   'top-tokens',
                                   'coh-80',
                                   10,
                                   6,
                                   topic_info=True,
                                   show_title=False)
        b1 = b.split('\n')[1]
        assert b1 == domain_string_first_token_line

    def test_background_strings(self, topics_handler, test_collection_name, background_string_first_token_line):
        b = topics_handler.pformat_background([test_collection_name, 'unittest_1_10_0.2_5_1'],
                                             columns=8,
                                             nb_tokens=20,
                                             show_title=True)

        b1 = b.split('\n')
        assert '+'.join(x for x in b1[2:3]) == background_string_first_token_line
