from collections import defaultdict
import artm
from patm.utils import cfg2model_settings
from .regularizers import regularizers_factory
from patm.definitions import collections_dir, get_generic_topic_names, DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME

from .topic_model import TopicModel, TrainSpecs
from ..evaluation.scorer_factory import get_scorers_factory, EvaluationFactory


dicts2model_factory = {}

def get_model_factory(dictionary, ppmi_dicts):
    """
    :param dictionary:
    :param ppmi_dicts:
    :return:
    :rtype: ModelFactory
    """
    if dictionary not in dicts2model_factory:
        dicts2model_factory[dictionary] = ModelFactory(dictionary, ppmi_dicts)
    return dicts2model_factory[dictionary]


class ModelFactory(object):
    """
    This class can create a fresh TopicModel or restore a TopicModel's state from disk. In both cases the model is ready to be trained.
    """
    def __init__(self, dictionary, cooc_dict):

        self._tm = None
        self.dict = dictionary
        self._artm = None
        self.topic_model_evaluators = {}
        self._eval_factory = EvaluationFactory(self.dict, cooc_dict)
        # self.cooc_dict = cooc_dict

    def create_model1(self, label, nb_topics, document_passes, cfg_file, modality_weights=None):
        """
        Creates a new Topic Model ready for training. Regularizers are defined in the input .cfg file. Phi matrix is
        initialized with random numbers by default\n
        :param str label: the model's label
        :param int nb_topics: number of topics to infer
        :param str document_passes: the number of phi matrix updates per document
        :param str cfg_file: full path to a "train.cfg" file used to define the scores to track while training the model
        :param dict modality_weights: whether to define "custom" and/or extra modalities with their weights. If not set then only the "@default_class" modality will be utilized when training
        :return: the ready-to-train TopicModel object
        :rtype: patm.modeling.TopicModel
        """
        settings = cfg2model_settings(cfg_file)
        regularizers = regularizers_factory.construct_reg_pool_from_files(cfg_file)
        self._build_artm(nb_topics, document_passes, modalities_dict=modality_weights)
        self._add_scorers(settings['scores'])
        self._build_topic_model(label, regularizers)
        self._create_topic_model(label, nb_topics, document_passes, settings['scores'])
        return self._tm

    def create_model(self, label, cfg_file, modality_weights=None, reg_config=None):
        """
        Creates an Topic Model ready for training with settings and regularizers defined in the input .cfg file. Phi matrix
        is initialized with random numbers by default\n
        :param str label: the unique identifier for the model
        :param str cfg_file: a config file containing initial model parameters, regularizers and score metrics to use
        :param dict modality_weights: whether to define "custom" and/or extra modalities with their weights. If not set then only the "@default_class" modality will be utilized when training
        :param str reg_config: a config file containing values to initialize the regularizer parameters; it has as sections the type of regularizers supported. Each section's attributes in the file can be optionally set to values (if unset inits with defaults)
        :return: tuple of initialized model and training specifications
        :rtype: patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs
        """
        settings = cfg2model_settings(cfg_file)
        regularizers = regularizers_factory.construct_pool_from_file(settings['regularizers'].items(), reg_config=reg_config)
        self._build_artm(settings['learning']['nb_topics'], settings['learning']['document_passes'], modalities_dict=modality_weights)
        self._add_scorers(settings['scores'])
        self._build_topic_model(label, regularizers)
        return self._tm, TrainSpecs(settings['learning']['collection_passes'], [], [])

    # TODO load modalities from the results dictionary. Allow the results to contain modalities in the first place
    @deprecated
    def create_model_with_phi_from_disk(self, phi_file_path, results):
        """
        Given a phi file path, a unique label and a dictionary of experimental tracked_metrics_dict, initializes a TopicModel object with the restored state of a model stored in disk. Configures to track the same
        evaluation metrics/scores. Uses the self.dictionary for indexing.\n
        Sets the below parameters with the latest corresponding values found in the expermental tracked_metrics_dict:\n
        - number of phi-matrix-updates/passes-per-document\n
        - number of train iterations to perform on the whole documents collection\n
        - regularizers to use and their parameters\n
        :param str phi_file_path: strored phi matrix p_wt
        :param dict results: experimental results object; patm.modeling.experiment.Experiment.get_results() output
        :return: tuple of initialized model and training specifications
        :rtype: (patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs)
        """
        regularizers = regularizers_factory.get_reg_pool_from_latest_results(results)
        self._build_artm(int(results['model_parameters']['nb_topics'][-1][1]), results['model_parameters']['document_passes'][-1][1], phi_path=phi_file_path)
        self._add_scorers(results['_eval_name2eval_type'])
        self._build_topic_model(results['model_label'], regularizers)
        return self._tm

    def _build_artm(self, nb_topics, nb_document_passes, modalities_dict=None, phi_path=''):
        if modalities_dict:
            assert all(map(lambda x: x in (DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME), modalities_dict.keys()))
        self._artm = artm.ARTM(num_topics=nb_topics, topic_names=get_generic_topic_names(nb_topics), class_ids=modalities_dict, num_document_passes=nb_document_passes, dictionary=self.dict)
        if phi_path:
            self._artm.load(phi_path)

    def _add_scorers(self, score_type2score_name):
        for score_setting_name, eval_instance_name in score_type2score_name.items():
            self._artm.scores.add(self._eval_factory.create_artm_scorer(score_setting_name, eval_instance_name))
            self.topic_model_evaluators[score_setting_name] = get_scorers_factory(score_type2score_name).create_scorer(eval_instance_name)

    def _build_topic_model(self, label, regularizers):
        self._tm = TopicModel(label, self._artm, self.topic_model_evaluators)
        for reg_instance in regularizers:
            self._tm.add_regularizer(reg_instance, regularizers_factory.get_type(reg_instance))


class ModelLabelDisagreementException(Exception):
    def __init__(self, msg):
        super(ModelLabelDisagreementException, self).__init__(msg)
