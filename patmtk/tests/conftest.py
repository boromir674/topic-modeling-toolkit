import os
import sys
from configparser import ConfigParser
import pytest

from patm.build_coherence import CoherenceFilesBuilder
from patm.modeling.trainer import TrainerFactory
from patm.modeling import Experiment
from patm import Tuner
from patm import political_spectrum as political_spectrum_manager


from processors import Pipeline, PipeHandler
from reporting import ResultsHandler
from reporting.dataset_reporter import DatasetReporter
from reporting.graph_builder import GraphMaker
from results.experimental_results import ExperimentalResults
from reporting.topics import TopicsHandler


####################3
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(MODULE_DIR, 'data')

TRAIN_CFG = os.path.join(MODULE_DIR, 'test-train.cfg')
REGS_CFG = os.path.join(MODULE_DIR, 'test-regularizers.cfg')


TEST_COLLECTIONS_ROOT_DIR_NAME = 'unittests-collections'

TEST_COLLECTION = 'unittest-dataset'
MODEL_1_LABEL = 'test-model-1'
#####################

@pytest.fixture(scope='session')
def collections_root_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp(TEST_COLLECTIONS_ROOT_DIR_NAME))

@pytest.fixture(scope='session')
def test_collection_name():
    return TEST_COLLECTION

#
@pytest.fixture(scope='session')
def rq1_cplsa_results_json():
    """These are the results gathered for a cplsa trained model"""
    return os.path.join(DATA_DIR, 'cplsa100000_0.2_0.json')


@pytest.fixture(scope='session')
def test_collection_dir(collections_root_dir, test_collection_name, tmpdir_factory):
    if not os.path.isdir(os.path.join(collections_root_dir, test_collection_name)):
        os.mkdir(os.path.join(collections_root_dir, test_collection_name))
    return os.path.join(collections_root_dir, test_collection_name)
    # return str(tmpdir_factory.mktemp(os.path.join(collections_root_dir, test_collection_name)))
    # return os.path.join(collections_root_dir, TEST_COLLECTION)

@pytest.fixture(scope='session')
def results_handler(collections_root_dir):
    return ResultsHandler(collections_root_dir, results_dir_name='results')


@pytest.fixture(scope='session')
def pairs_file_nb_lines():  # number of lines in cooc and ppmi files (771 in python2, 759 in python3)
    python3 = {True: 759,  # Dirty code to support python 2 backwards compatibility
               False: 771}
    return python3[2 < sys.version_info[0]]

@pytest.fixture(scope='session')
def pipe_n_quantities(test_collection_dir, pairs_file_nb_lines):
    return {'unittest-pipeline-cfg': os.path.join(MODULE_DIR, 'test-pipeline.cfg'),
            'unittest-collection-dir': test_collection_dir,
            'category': 'posts',
            'sample': 100,
            'resulting-nb-docs': 100,
            'nb-bows': 1297,
            'word-vocabulary-length': 833,
            'nb-all-modalities-terms': 834,  # corresponds to the number of lines in the vocabulary file created (must call persist of PipeHandler with add_class_labels_to_vocab=True, which is the default).
            # the above probably will fail in case no second modality is used (only the @default_class is enabled)
            'nb-lines-cooc-n-ppmi-files': pairs_file_nb_lines
            }

@pytest.fixture(scope='session')
def political_spectrum():
    return political_spectrum_manager


#### OPERATIONS ARTIFACTS
@pytest.fixture(scope='session')
def preprocess_phase(pipe_n_quantities):
    pipe_handler = PipeHandler()
    pipe_handler.process(pipe_n_quantities['unittest-pipeline-cfg'], pipe_n_quantities['category'], sample=pipe_n_quantities['sample'])
    return pipe_handler

@pytest.fixture(scope='session')
def test_dataset(preprocess_phase, political_spectrum, test_collection_dir):
    """A dataset ready to be used for topic modeling training. Depends on the input document sample size to take and resulting actual size"""
    text_dataset = preprocess_phase.persist(test_collection_dir, political_spectrum.poster_id2ideology_label, political_spectrum.class_names, add_class_labels_to_vocab=True)
    coh_builder = CoherenceFilesBuilder(test_collection_dir)
    coh_builder.create_files(cooc_window=10, min_tf=0, min_df=0, apply_zero_index=False)
    return text_dataset

# PARSE UNITTEST CFG FILES

def parse_cfg(cfg):
    config = ConfigParser()
    config.read(cfg)
    return {section: dict(config.items(section)) for section in config.sections()}


@pytest.fixture(scope='session')
def train_settings():
    """These settings (learning, reg components, score components, etc) are used to train the model in 'trained_model' fixture. A dictionary of cfg sections mapping to dictionaries with settings names-values pairs."""
    _ = parse_cfg(TRAIN_CFG)
    _['regularizers'] = {k: v for k, v in _['regularizers'].items() if v}
    _['scores'] = {k: v for k, v in _['scores'].items() if v}
    return _


@pytest.fixture(scope='session')
def trainer(collections_root_dir, test_dataset):
    return TrainerFactory().create_trainer(os.path.join(collections_root_dir, test_dataset.name), exploit_ideology_labels=True, force_new_batches=True)


@pytest.fixture(scope='session')
def trained_model_n_experiment(collections_root_dir, test_dataset, trainer):
    experiment = Experiment(os.path.join(collections_root_dir, test_dataset.name))
    topic_model = trainer.model_factory.create_model(MODEL_1_LABEL, TRAIN_CFG, reg_cfg=REGS_CFG, show_progress_bars=False)
    train_specs = trainer.model_factory.create_train_specs()
    trainer.register(experiment)
    experiment.init_empty_trackables(topic_model)
    trainer.train(topic_model, train_specs, effects=False, cache_theta=True)
    experiment.save_experiment(save_phi=True)
    return topic_model, experiment


@pytest.fixture(scope='session')
def loaded_model_n_experiment(collections_root_dir, test_dataset, trainer, trained_model_n_experiment):
    model, experiment = trained_model_n_experiment
    experiment.save_experiment(save_phi=True)
    new_exp_obj = Experiment(os.path.join(collections_root_dir, test_dataset.name))
    trainer.register(new_exp_obj)
    loaded_model = new_exp_obj.load_experiment(model.label)
    return loaded_model, new_exp_obj


    # if args.load:
    #     topic_model = experiment.load_experiment(args.label)
    #     print '\nLoaded experiment and model state'
    #     settings = cfg2model_settings(args.config)
    #     train_specs = TrainSpecs(15, [], [])


@pytest.fixture(scope='session')
def tuner_obj(collections_root_dir, test_dataset):
    from patm.tuning.building import tuner_definition_builder as tdb
    tuner = Tuner(os.path.join(collections_root_dir, test_dataset.name), evaluation_definitions={
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
    return tuner


@pytest.fixture(scope='session')
def dataset_reporter(tuner_obj):
    return DatasetReporter(os.path.dirname(tuner_obj._dir))


@pytest.fixture(scope='session')
def graphs(exp_res_obj1, trained_model_n_experiment, tuner_obj):
    graph_maker = GraphMaker(os.path.dirname(tuner_obj._dir))
    sparser_regularizers_tau_coefficients_trajectories = False
    graph_maker.build_graphs_from_collection(os.path.basename(tuner_obj._dir), 3,  # use a maximal number of 8 models to compare together
                                             metric='perplexity',  # explicit set to force alphabetical ordering of models
                                             score_definitions=['background-tokens-ratio-0.30', 'kernel-coherence-0.80', 'sparsity-theta', 'top-tokens-coherence-10'],
                                             tau_trajectories=(lambda x: 'all' if x else '')(sparser_regularizers_tau_coefficients_trajectories))
    return graph_maker.saved_figures



############################################
@pytest.fixture(scope='session')
def json_path(collections_root_dir, test_collection_name):
    return os.path.join(collections_root_dir, test_collection_name, 'results', 'exp-results-test-model_1.json')



@pytest.fixture(scope='session')
def kernel_data_0():
    return [
        [[1, 2], [3, 4], [5, 6], [120, 100]],
        {'t01': {'coherence': [1, 2, 3],
                 'contrast': [6, 3],
                 'purity': [1, 8]},
         't00': {'coherence': [10, 2, 3],
                 'contrast': [67, 36],
                 'purity': [12, 89]},
         't02': {'coherence': [10, 11],
                 'contrast': [656, 32],
                 'purity': [17, 856]}}
    ]


@pytest.fixture(scope='session')
def kernel_data_1():
    return [[[10,20], [30,40], [50,6], [80, 90]], {'t01': {'coherence': [3, 9],
                                                             'contrast': [96, 3],
                                                             'purity': [1, 98]},
                                                     't00': {'coherence': [19,2,93],
                                                             'contrast': [7, 3],
                                                             'purity': [2, 89]},
                                                     't02': {'coherence': [0,11],
                                                             'contrast': [66, 32],
                                                             'purity': [17, 85]}
                                                     }]


@pytest.fixture(scope='session')
def exp_res_obj1(kernel_data_0, kernel_data_1, json_path, test_collection_dir):

    exp = ExperimentalResults.from_dict({
        'scalars': {
            'dir': 'a-dataset-dir',
            'label': 'a-model-label',
            'dataset_iterations': 3,  # LEGACY '_' (underscore) usage
            'nb_topics': 5,  # LEGACY '_' (underscore) usage
            'document_passes': 2,  # LEGACY '_' (underscore) usage
            'background_topics': ['t0', 't1'],  # LEGACY '_' (underscore) usage
            'domain_topics': ['t2', 't3', 't4'],  # LEGACY '_' (underscore) usage
            'modalities': {'dcn': 1, 'icn': 5}
        },
        'tracked': {
            'perplexity': [1, 2, 3],
            'sparsity-phi-@dc': [-2, -4, -6],
            'sparsity-phi-@ic': [-56, -12, -32],
            'sparsity-theta': [2, 4, 6],
            'background-tokens-ratio-0.3': [0.4, 0.3, 0.2],
            'topic-kernel': {
                '0.60': {
                    'avg_coh': kernel_data_0[0][0],
                    'avg_con': kernel_data_0[0][1],
                    'avg_pur': kernel_data_0[0][2],
                    'size': kernel_data_0[0][3],
                    'topics': kernel_data_0[1]
                },
                '0.80': {
                    'avg_coh': kernel_data_1[0][0],
                    'avg_con': kernel_data_1[0][1],
                    'avg_pur': kernel_data_1[0][2],
                    'size': kernel_data_1[0][3],
                    'topics': kernel_data_1[1]
                }
            },
            'top-tokens': {
                '10': {
                    'avg_coh': [5, 6, 7],
                    'topics': {'t01': [12, 22, 3], 't00': [10, 2, 3], 't02': [10, 11]}
                },
                '100': {
                    'avg_coh': [10, 20, 30],
                    'topics': {'t01': [5, 7, 9], 't00': [12, 32, 3], 't02': [11, 1]}
                }
            },
            'tau-trajectories': {'phi': [1, 2, 3], 'theta': [5, 6, 7]},
            'regularization-dynamic-parameters': {'type-a': {'tau': [1, 2, 3]},
                                                  'type-b': {'tau': [-1, -1, -2], 'alpha': [1, 1.2]}},
            'collection-passes': [3]
        },
        'final': {
            'topic-kernel': {
                '0.60': {'t00': ['a', 'b', 'c'],
                         't01': ['d', 'e', 'f'],
                         't02': ['g', 'h', 'i']},
                '0.80': {'t00': ['j', 'k', 'l'],
                         't01': ['m', 'n', 'o'],
                         't02': ['p', 'q', 'r']}
            },
            'top-tokens': {
                '10': {
                    't00': ['s', 't', 'u'],
                    't01': ['v', 'x', 'y'],
                    't02': ['z', 'a1', 'b1']
                },
                '100': {
                    't00': ['c1', 'd1', 'e1'],
                    't01': ['f1', 'g1', 'h1'],
                    't02': ['i1', 'j1', 'k1']
                }
            },
            'background-tokens': ['l1', 'm1', 'n1']
        },
        'regularizers': ['reg1_params_pformat', 'reg2_params_pformat'],
        'reg_defs': {'type-a': 'reg1', 'type-b': 'reg2'},
        'score_defs': {'perplexity': 'prl', 'top-tokens-10': 'top10'}
    })

    if not os.path.isdir(os.path.join(test_collection_dir, 'results')):
        os.mkdir(os.path.join(test_collection_dir, 'results'))
    exp.save_as_json(json_path)
    return exp
