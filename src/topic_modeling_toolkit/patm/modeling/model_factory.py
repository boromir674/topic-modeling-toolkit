import artm

from .topic_model import TopicModel, TrainSpecs
from ..evaluation.scorer_factory import EvaluationFactory
from topic_modeling_toolkit.patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME
from topic_modeling_toolkit.patm.utils import cfg2model_settings, generic_topic_names_builder as tn_builder
from .regularization.regularizers_factory import RegularizersFactory, cfg2regularizer_settings


import logging
logger = logging.getLogger(__name__)


class ModelFactory(object):
    """This class can create a fresh TopicModel or restore a TopicModel's state from disk. In both cases the model is ready to be trained"""
    abbreviation2_class_name = {'@dc': DEFAULT_CLASS_NAME, '@ic': IDEOLOGY_CLASS_NAME}
    _instances = {}

    @classmethod
    def _key(cls, *args):
        # return '_'.join(sorted([getattr(x, 'name', str(x)) for x in [args[0]] + [v['obj'] for v in args[1].values()]]))
        return getattr(args[0], 'name', str(args[0]))

    def __new__(cls, *args, **kwargs):
        if cls._key(*args) not in cls._instances:
            cls._instances[cls._key(*args)] = super(ModelFactory, cls).__new__(cls)
            cls._instances[cls._key(*args)].coherence_dict = args[0]
            cls._instances[cls._key(*args)]._eval_factory = EvaluationFactory(args[0], abbreviation2class_name=cls.abbreviation2_class_name.copy())
            cls._instances[cls._key(*args)]._regularizers_factory = RegularizersFactory(args[0])
            cls._instances[cls._key(*args)]._tm = None
            cls._instances[cls._key(*args)]._artm = None
            cls._instances[cls._key(*args)]._col_passes = 0
            cls._instances[cls._key(*args)]._nb_topics = 0
            cls._instances[cls._key(*args)]._nb_document_passes = 0
            # cls._instances[cls._key(*args)].cooc_dict = cooc_dict
            cls._instances[cls._key(*args)]._modality_weights = {}
            cls._instances[cls._key(*args)].topic_model_evaluators = {}
            cls._instances[cls._key(*args)]._eval_def2name = {}
            cls._instances[cls._key(*args)]._reg_types2names = {}
            cls._instances[cls._key(*args)]._progress_bars = False
        return cls._instances[cls._key(*args)]

    def __init__(self, dictionary):
        """
        :param dictionary: this dictionary is passed to artm.ARTM constructor. IT should be the one holding the pmi values computed for token pairs
        :param cooc_dict:
        """
    def create_model(self, label, train_cfg, reg_cfg=None, show_progress_bars=False):
        """
        Call this method to create a new TopicModel. Before training on it you mush invoke the create_train_specs of the ModelFactory
        in order to potentially initialize model's regularization dynamic parameters trajectories\n
        :param str label: a unique model label
        :param str train_cfg: path to config file used to define nb_topics, collection_passes, codument_passes, background_topics_percentage, modalities_weights, scoring components and active regulariers
        :param str reg_cfg: path to cfg file used to initialize (where applicable) the parameters (such as 'tau coefficinet') of the active regularizers
        :param bool show_progress_bars:
        :return: the topics model constructed
        :rtype: patm.modeling.topic_model.TopicModel
        """
        _ = cfg2model_settings(train_cfg)
        return self.construct_model(label, _['learning']['nb_topics'], _['learning']['collection_passes'], _['learning']['document_passes'],
                                    float(_['information'].get('background-topics-percentage', 0)), self._parse_modalities(_['information']),
                                    _['scores'], _['regularizers'], reg_settings=reg_cfg, show_progress_bars=show_progress_bars)

    def create_train_specs(self, collection_passes=None):
        """Creates Train Specs according to the latest TopicModel instance created."""
        if not collection_passes:
            collection_passes = self._col_passes
        self._tm.initialize_regularizers(collection_passes, self._nb_document_passes)
        return TrainSpecs(collection_passes, list(map(lambda x: x[0], self._tm.tau_trajectories)), list(map(lambda x: x[1], self._tm.tau_trajectories)))

    def construct_model(self, label, nb_topics, nb_collection_passes, nb_document_passes, background_topics_pct, modality_weights, scores, regularizers, reg_settings=None, show_progress_bars=False):
        """
        :param str label: a unique model label
        :param int nb_topics: number of assumed topics; Phi matrix number of columns
        :param int nb_collection_passes: number of train iterations through the entire dataset
        :param int nb_document_passes: number of iterations through each document
        :param float background_topics_pct: the portion of topics designated to hold background words distributions (occupy the first columns of the Phi matrix)
        :param dict modality_weights: key to float mapping of modalities and scaling coefficients contributing in the likelihood function
        :param dict scores: mapping of evaluator/scorer definition string to name, indicating metrics to compute/track while training
        :param dict regularizers: mapping of regularization component definition string to name, indicating the active regularizers while training
        :param dict reg_settings: path to cfg file used to initialize (where applicable) the parameters (such as 'tau coefficinet') of the active regularizers
        :param bool show_progress_bars:
        :return: the topics model constructed
        :rtype: patm.modeling.topic_model.TopicModel
        """
        self._col_passes, self._nb_topics, self._nb_document_passes = nb_collection_passes, nb_topics, nb_document_passes
        self.modalities, self._eval_def2name, self._reg_types2names = modality_weights, scores, regularizers
        self._progress_bars = show_progress_bars
        return self._create_model(label,
                                  *tn_builder.define_nb_topics(self._nb_topics).define_background_pct(background_topics_pct)
                                  .get_background_n_domain_topics(),
                                  reg_cfg=reg_settings)

    def create_model_with_phi_from_disk(self, phi_file_path, results, show_progress_bars=False):
        """
        Loaded model will overwrite ARTM.topic_names and class_ids fields.
        All class_ids weights will be set to 1.0, (if necessary, they have to be specified by hand. The method call will empty ARTM.score_tracker.
        All regularizers and scores will be forgotten. It is strongly recommend to reset all important parameters of the ARTM model, used earlier.\n

        Given a phi file path, a unique label and a dictionary of experimental tracked_metrics_dict, initializes a TopicModel object with the restored state of a model stored in disk. Configures to track the same
        evaluation metrics/scores. Uses the self.coherence_dict for indexing.\n
        It copies the tau trajectory definitions found in loaded model. This means that tau will foloow the same trajectory from scratch
        Sets the below parameters with the latest corresponding values found in the experimental tracked_metrics_dict:\n
        - number of phi-matrix-updates/passes-per-document\n
        - number of train iterations to perform on the whole documents collection\n
        - regularizers to use and their parameters\n
        :param str phi_file_path: strored phi matrix p_wt
        :param results.experimental_results.ExperimentalResults results:
        :return: tuple of initialized model and training specifications
        :rtype: (patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs)
        """
        self.modalities = results.scalars.modalities
        self._eval_def2name = results.score_defs

        # BUILD
        bgt, dmt = results.scalars.background_topics, results.scalars.domain_topics
        self._eval_factory.domain_topics = dmt
        self._artm = artm.ARTM(topic_names=bgt + dmt, dictionary=self.coherence_dict, show_progress_bars=show_progress_bars)
        self._artm.load(phi_file_path)
        if self._artm.get_phi().shape[1] != len(bgt + dmt):
            raise RuntimeError("Quite unfortunate! The loaded phi p(w|t) matrix, loaded from {}, has {} columns, but the total length of background and domain topics in the experimental results".format(self._artm.get_phi().shape[1], phi_file_path, len(bgt + dmt)))
        self._artm.topic_names = bgt + dmt
        self._artm.num_document_passes = results.scalars.document_passes
        self._artm.class_ids = self.modalities

        # SCORES
        self._add_scorers()
        # REGS
        self._tm = TopicModel(results.scalars.model_label, self._artm, self.topic_model_evaluators)

        self._tm.add_regularizer_wrappers(self._regularizers_factory.create_reg_wrappers(results.reg_defs, bgt, dmt))
        return self._tm

    def _create_model(self, label, background_topics, domain_topics, reg_cfg=None, phi_path=''):
        assert self._nb_topics == len(background_topics + domain_topics)
        self._build_artm(background_topics, domain_topics, modalities_dict=self._modality_weights, phi_path=phi_path)
        self._add_scorers()
        self._tm = TopicModel(label, self._artm, self.topic_model_evaluators)
        self._tm.add_regularizer_wrappers(self._regularizers_factory.create_reg_wrappers(self._reg_types2names,
                                                                                                  background_topics, domain_topics,
                                                                                                  reg_cfg=reg_cfg))
        return self._tm

    # TODO replace calls to _build_artm with _build_artm_new
    def _build_artm(self, background_topics, domain_topics, modalities_dict=None, phi_path=''):
        self._eval_factory.domain_topics = domain_topics
        self._artm = artm.ARTM(num_topics=self._nb_topics, dictionary=self.coherence_dict, show_progress_bars=self._progress_bars)
        if phi_path:
            self._artm.load(phi_path)
        self._artm.topic_names = background_topics + domain_topics
        self._artm.class_ids = modalities_dict
        self._artm.num_document_passes = self._nb_document_passes

    def _add_scorers(self):
        """Call this method to attach the appropriate scoring components to the topic model under construction. It assumes that
            modalities have already been set (modalities.setter can be used for that) with weights, and that evaluation definitions have already been fed and
        :return:
        """
        self.topic_model_evaluators = {}
        self._eval_factory.modalities = self._modality_weights
        for evaluator_definition, eval_instance_name in self._eval_def2name.items():
            if evaluator_definition.split('-')[-1][0] == '@' and self.abbreviation2_class_name[evaluator_definition.split('-')[-1]] not in self._modality_weights:
                # raise RuntimeError("Skipping score definition {}. weights: {}, struct: {}".format({evaluator_definition: eval_instance_name}, self._modality_weights, self.abbreviation2_class_name))
                logger.warning("Skipping score definition {} because its respective modality weight is 0".format({evaluator_definition: eval_instance_name}))
                continue
            self.topic_model_evaluators[evaluator_definition] = self._eval_factory.create_evaluator(evaluator_definition, eval_instance_name)
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
        self._modality_weights = {DEFAULT_CLASS_NAME: 1}
        for modality, weight in modality_weights.items():
            self._set_modality(modality, weight)
        logger.info("'Modalities/class-ids' and 'weights as coefficients' additive factors contributing to the likelihodd function: {}".format(self._modality_weights))

    def _set_modality(self, modality, weight):
        if float(weight) <= 0:
            raise SetInvalidDefaultModalityWeightError("Tried to set {}  modality weight to {:.2f}. It should be a positive number".format(modality, weight))
        if modality not in (DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME):
            logger.warning("Input modality '{}' is not in the supported list: ['{}', '{}']".format(modality, DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME))
        else:
            self._modality_weights[modality] = float(weight)

    def _parse_modalities(self, information_dict):
        """Default modality: default weight = 1, Ideology modality: default weight = 0"""
        return {k: v for k, v in {DEFAULT_CLASS_NAME: float(information_dict.get('default-class-weight', 1)), IDEOLOGY_CLASS_NAME: float(information_dict.get('ideology-class-weight', 0))}.items() if v}


class ModelLabelDisagreementException(Exception): pass
class SetInvalidDefaultModalityWeightError(Exception): pass
