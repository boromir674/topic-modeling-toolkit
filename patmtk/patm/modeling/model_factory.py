import artm

from .topic_model import TopicModel, TrainSpecs
from ..evaluation.scorer_factory import EvaluationFactory
from patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME
from patm.utils import cfg2model_settings
from patm.utils import generic_topic_names_builder as tn_builder
from patm.modeling.regularization.regularizers import RegularizersFactory, cfg2regularizer_settings


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
    abbreviation2_class_name = {'@dc': DEFAULT_CLASS_NAME, '@ic': IDEOLOGY_CLASS_NAME}
    def __init__(self, dictionary, cooc_dict):
        self.dict = dictionary
        self._eval_factory = EvaluationFactory(dictionary, cooc_dict, abbreviation2class_name=self.abbreviation2_class_name)
        self._regularizers_factory = RegularizersFactory()
        self._tm = None
        self._artm = None
        self._col_passes = 0
        self._nb_topics = 0
        self._nb_document_passes = 0
        # self.cooc_dict = cooc_dict
        self._modality_weights = {}
        self.topic_model_evaluators = {}
        self._eval_def2name = {}
        self._reg_types2names = {}

    def construct_model(self, label, nb_topics, nb_collection_passes, nb_document_passes, background_topics_pct, modality_weights, scores, regularizers, reg_settings=None):
        """
        :param str label:
        :param int nb_topics:
        :param int nb_collection_passes:
        :param int nb_document_passes:
        :param float background_topics_pct:
        :param dict modality_weights: IDEOLOGY_CLASS_NAME and/or DEFAULT_CLASS_NAME as keys
        :param dict scores:
        :param dict regularizers:
        :param dict reg_settings:
        :return:
        :rtype: patm.modeling.topic_model.TopicModel
        """
        self._col_passes, self._nb_topics, self._nb_document_passes = nb_collection_passes, nb_topics, nb_document_passes
        self.modalities, self._eval_def2name, self._reg_types2names = modality_weights, scores, regularizers
        return self._create_model(label,
                                  *tn_builder.define_nb_topics(self._nb_topics).define_background_pct(background_topics_pct).get_background_n_domain_topics(),
                                  reg_cfg=reg_settings)

    def create_train_specs(self, collection_passes=None):
        """Creates Train Specs according to the latest TopicModel instance created."""
        if not collection_passes:
            collection_passes = self._col_passes
        self._tm.initialize_regularizers(collection_passes, self._nb_document_passes)
        return TrainSpecs(collection_passes, list(map(lambda x: x[0], self._tm.tau_trajectories)), list(map(lambda x: x[1], self._tm.tau_trajectories)))

    def create_model(self, label, train_cfg, reg_cfg=None):
        _ = cfg2model_settings(train_cfg)
        return self.construct_model(label, _['learning']['nb_topics'], _['learning']['collection_passes'], _['learning']['document_passes'],
                                    float(_['information'].get('background-topics-percentage', 0)), self._parse_modalities(_['information']),
                                    _['scores'], _['regularizers'], reg_settings=reg_cfg)

    def create_model11(self, label, nb_topics, document_passes, train_cfg, modality_weights=None, background_topics_pct=0.0):
        _ = cfg2model_settings(train_cfg)
        return self.construct_model(label, nb_topics, _['learning']['collection_passes'], document_passes, background_topics_pct, modality_weights, _['scores'], _['regularizers'])

    def create_model00(self, label, train_cfg, reg_cfg=None, modality_weights=None, background_topics_pct=0.0):
        _ = cfg2model_settings(train_cfg)
        return self.construct_model(label, _['learning']['nb_topics'], _['learning']['collection_passes'], _['learning']['document_passes'], background_topics_pct, modality_weights, _['scores'], _['regularizers'], reg_settings=reg_cfg)

    # def create_model_with_phi_from_disk(self, phi_file_path, results):
    #     """
    #     Given a phi file path, a unique label and a dictionary of experimental tracked_metrics_dict, initializes a TopicModel object with the restored state of a model stored in disk. Configures to track the same
    #     evaluation metrics/scores. Uses the self.dictionary for indexing.\n
    #     It copies the tau trajectory definitions found in loaded model. This means that tau will foloow the same trajectory from scratch
    #     Sets the below parameters with the latest corresponding values found in the experimental tracked_metrics_dict:\n
    #     - number of phi-matrix-updates/passes-per-document\n
    #     - number of train iterations to perform on the whole documents collection\n
    #     - regularizers to use and their parameters\n
    #     :param str phi_file_path: strored phi matrix p_wt
    #     :param dict results: experimental results object; patm.modeling.experiment.Experiment.get_results() output
    #     :return: tuple of initialized model and training specifications
    #     :rtype: (patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs)
    #     """
    #     self._nb_topics, self._nb_document_passes = int(results['model_parameters']['nb_topics'][-1][1]), results['model_parameters']['document_passes'][-1][1]
    #     self._eval_def2name, self._reg_types2names = results['eval_definition2eval_name'], {k: v['name'] for k, v in results['reg_parameters'][-1][1].items()}
    #     return self._create_model(results['model_label'], results['modalities'], results['background_topics'], results['domain_topics'], reg_cfg=results['reg_parameters'][1][1], phi_path=phi_file_path)

    def _create_model(self, label, background_topics, domain_topics, reg_cfg=None, phi_path=''):
        assert self._nb_topics == len(background_topics + domain_topics)
        self._build_artm(background_topics, domain_topics, modalities_dict=self._modality_weights, phi_path=phi_path)
        self._add_scorers()
        self._tm = TopicModel(label, self._artm, self.topic_model_evaluators)
        self._tm.add_regularizer_wrappers(self._regularizers_factory.set_regularizers_definitions(self._reg_types2names, background_topics, domain_topics, reg_cfg=reg_cfg).create_reg_wrappers())
        return self._tm

    def _build_artm(self, background_topics, domain_topics, modalities_dict=None, phi_path=''):
        self._eval_factory.domain_topics = domain_topics
        self._artm = artm.ARTM(num_topics=self._nb_topics, dictionary=self.dict)
        if phi_path:
            self._artm.load(phi_path)
        self._artm.topic_names = background_topics + domain_topics
        self._artm.class_ids = modalities_dict
        self._artm.num_document_passes = self._nb_document_passes

    def _add_scorers(self):
        self._eval_factory.modalities = self._modality_weights
        for evaluator_definition, eval_instance_name in self._eval_def2name.items():
            if evaluator_definition.split('-')[-1][0] == '@' and self.abbreviation2_class_name[evaluator_definition.split('-')[-1]] not in self._modality_weights:
                continue
            self.topic_model_evaluators[evaluator_definition] = self._eval_factory.create_evaluator(evaluator_definition, eval_instance_name)
            # ob = self.topic_model_evaluators[evaluator_definition]
            self._artm.scores.add(self.topic_model_evaluators[evaluator_definition].artm_score)

    @property
    def modalities(self):
        return self._modality_weights

    @modalities.setter
    def modalities(self, modality_weights):
        """
        Sets '\@default_class' weight to 1 if key not found.\n
        Sets '\@ideology_class' weight to 0 if key not found.\n
        """
        assert all(map(lambda x: x in (DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME), modality_weights.keys()))
        weight = modality_weights.get(DEFAULT_CLASS_NAME, 1)
        if weight == 0:
            raise ZeroWeightedDefaultModalityException("Tried to set DEFAULT_CLASS_NAME weight to 0. Either the dictionary passed has a 0 value for the 'default-class-weight' key, or the key is missing.")
        else:
            self._modality_weights[DEFAULT_CLASS_NAME] = weight
        weight = modality_weights.get(IDEOLOGY_CLASS_NAME, 0)
        self._modality_weights[IDEOLOGY_CLASS_NAME] = weight

    def _parse_modalities(self, information_dict):
        """Default modality: default weight = 1, Ideology modality: default weight = 0"""
        return {DEFAULT_CLASS_NAME: float(information_dict.get('default-class-weight', 1)), IDEOLOGY_CLASS_NAME: float(information_dict.get('ideology-class-weight', 0))}


class ModelLabelDisagreementException(Exception):
    def __init__(self, msg):
        super(ModelLabelDisagreementException, self).__init__(msg)


class ZeroWeightedDefaultModalityException(Exception):
    def __init__(self, msg):
        super(ZeroWeightedDefaultModalityException, self).__init__(msg)
