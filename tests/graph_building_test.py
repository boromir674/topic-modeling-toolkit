import os
import sys
import pytest


@pytest.fixture(scope='module')
def model_labels(test_collection_dir, graphs):
    return sorted([os.path.basename(x).replace('.json', '') for x in os.listdir(os.path.join(test_collection_dir, 'results'))])


@pytest.fixture(scope='module')
def graph_paths(test_collection_dir, model_labels, graphs_parameters):
    # s = ['a-model-label', 'unittest_0_10_0.2_5_1', 'unittest_1_10_0.2_5_1'] # , 'unittest_0_12_0.2_5_1', 'unittest_1_12_0.2_5_1']
    return [os.path.join(test_collection_dir, 'graphs', '{}-{}'.format('+'.join(m for m in model_labels[:graphs_parameters['selection']]), x)) for x in [
        'btr.30_v01.png',
        # 'kch.60_v01.png',
        'kch.80_v01.png',
        # 'kcn.60_v01.png',
        # 'kcn.80_v01.png',
        # 'kpr.60_v01.png',
        # 'kpr.80_v01.png',
        # 'ksz.60_v01.png',
        # 'ksz.80_v01.png',
        # 'prpl_v01.png',
        # 'spp@d_v01.png',
        'spt_v01.png',
        'top10ch_v01.png',
        # 'top100ch_v01.png',
        # 'tau_phi_v01.png',
        # 'tau_theta_v01.png'
    ]]


# @pytest.fixture(scope='module')
# def model_labels():  # number of lines in cooc and ppmi files (771 in python2, 759 in python3)
#     python3 = {True: ['a-model-label', 'unittest_0_12_0.2_5_1', 'unittest_1_12_0.2_5_1'],  # Dirty code to support python 2 backwards compatibility
#                False: ['a-model-label', 'unittest_0_12_0.2_5_1', 'unittest_1_12_0.2_5_1']}
#     return python3[2 < sys.version_info[0]]


def test_strings(graphs, graph_paths):
    assert sorted(graphs) == sorted(graph_paths)
    # assert sorted(graphs) == []
