import pytest

from patm.tuning import Tuner
from patm.tuning.building import tuner_definition_builder as tdb
from os import path

@pytest.fixture(scope='module')
def tuner(collections_root_dir, test_dataset):
    return Tuner(path.join(collections_root_dir, test_dataset.name), evaluation_definitions={
        'perplexity': 'per',
        'sparsity-phi-@dc': 'sppd',
        'sparsity-theta': 'spt',
        'topic-kernel-0.60': 'tk60',
        'topic-kernel-0.80': 'tk80',
        'top-tokens-10': 'top10',
        'top-tokens-100': 'top100',
        'background-tokens-ratio-0.3': 'btr3',
        'background-tokens-ratio-0.2': 'btr2'
    }, verbose=0)

def test_tuner(tuner):

    tuning_definition = tdb.initialize()\
        .nb_topics(10, 12)\
        .collection_passes(5)\
        .document_passes(1)\
        .background_topics_pct(0.2) \
        .ideology_class_weight(0, 1) \
        .build()

        # .sparse_phi()\
        #     .deactivate(8)\
        #     .kind('linear')\
        #     .start(-1)\
        #     .end(-10, -100)\
        # .sparse_theta()\
        #     .deactivate(10)\
        #     .kind('linear')\
        #     .start(-1)\
        #     .end(-10, -100)\

    tuner.active_regularizers = [
        # 'smooth-phi',
        # 'smooth-theta',
        'label-regularization-phi-dom-cls',
        'decorrelate-phi-dom-def',
    ]
    tuner.tune(tuning_definition,
               prefix_label='unittest',
               append_explorables=True,
               append_static=True,
               force_overwrite=True,
               verbose=False)