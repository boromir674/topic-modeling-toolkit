import artm
from patm.utils import cfg2model_settings
from patm.definitions import collections_dir
from .topic_model import TopicModel, TrainSpecs
from .regularizers import init_from_file, init_from_latest_results, get_reg_pool_from_files
from ..evaluation.scorer_factory import get_scorers_factory, EvaluationFactory


dicts2model_factory = {}

def get_model_factory(dictionary):
    if dictionary not in dicts2model_factory:
        dicts2model_factory[dictionary] = ModelFactory(dictionary)
    return dicts2model_factory[dictionary]


class ModelFactory(object):
    """
    This class can create a fresh TopicModel or restore a TopicModel's state from disk. In both cases the model is ready to be trained.
    """
    def __init__(self, dictionary):
        self.dict = dictionary
        self._tm = None
        self._eval_factory = EvaluationFactory(self.dict)
        self._scores = {}

    # def _parse_train_cfg(self, cfg):
    #     _ = cfg2model_settings(cfg)
    #     self._nb_topics = _['learning']['nb_topicss']
    #     self._col_passes = _['learning']['collection_passes']
    #     self._doc_passes = _['learning']['document_passes']
    #     self._scores = _['scores']
    #     self._regs = _['regularizers']

    def create_model1(self, label, nb_topics, document_passes, cfg_file, reg_cfg):
        """

        :param int nb_topics:
        :param int document_passes:
        :param str cfg_file:
        :param str reg_cfg:
        :return:
        :rtype: patm.modeling.topic_model.TopicModel
        """
        settings = cfg2model_settings(cfg_file)
        regularizers = get_reg_pool_from_files(cfg_file, reg_cfg)
        model = artm.ARTM(num_topics=nb_topics,
                          dictionary=self.dict,
                          topic_names=get_generic_topic_names(nb_topics))
        model.num_document_passes = document_passes
        self._create_topic_model(label, model, settings['scores'])
        self._set_regularizers(regularizers)
        return self._tm

    def create_model(self, label, cfg_file, reg_cfg):
        """
        Creates an artm _topic_model instance based on the input config file. Phi matrix is initialized with random numbers by default\n
        :param str label: the unique identifier for the model
        :param str cfg_file: A config file containing initial model parameters, regularizers and score metrics to use
        :param str reg_cfg: A config file containing values to initialize the regularizer parameters
        :return: tuple of initialized model and training specifications
        :rtype: patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs
        """
        settings = cfg2model_settings(cfg_file)
        nb_topics = settings['learning']['nb_topics']
        regularizers = init_from_file(settings['regularizers'].items(), reg_cfg)
        model = artm.ARTM(num_topics=nb_topics,
                          dictionary=self.dict,
                          topic_names=get_generic_topic_names(nb_topics))
        model.num_document_passes = settings['learning']['document_passes']
        self._create_topic_model(label, model, settings['scores'])
        self._set_regularizers(regularizers)
        return self._tm, TrainSpecs(settings['learning']['collection_passes'], [], [])

    def create_model_with_phi_from_disk(self, phi_file_path, results):
        """
        Given a phi file path, a unique label and a dictionary of experimental res_dict, initializes a TopicModel object with the restored state of a model stored in disk. Configures to track the same
        evaluation metrics/scores. Uses the self.dictionary for indexing.\n
        Sets the below parameters with the latest corresponding values found in the expermental res_dict:\n
        - number of phi-matrix-updates/passes-per-document\n
        - number of train iterations to perform on the whole documents collection\n
        - regularizers to use and their parameters\n
        :param str phi_file_path: strored phi matrix p_wt
        :param dict results: experimental results object; patm.modeling.experiment.Experiment.get_results() output
        :return: tuple of initialized model and training specifications
        :rtype: (patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs)
        """
        regularizers = init_from_latest_results(results)
        nb_topics = int(results['model_parameters']['nb_topics'][-1][1])
        model = artm.ARTM(num_topics=nb_topics,
                          dictionary=self.dict,
                          topic_names=get_generic_topic_names(nb_topics)) # number of topics have to be set, but doesn't matter, because they are overridden by the phi matrix loaded below
        model.load(filename=phi_file_path)
        model.num_document_passes = results['model_parameters']['document_passes'][-1][1]
        self._create_topic_model(results['model_label'], model, dict(results['_eval_name2eval_type']))
        self._set_regularizers(regularizers)
        return self._tm

    def _create_topic_model(self, label, artm_model, score_type2score_name):
        scorers = {}
        for score_setting_name, eval_instance_name in score_type2score_name.items():
            artm_model.scores.add(self._eval_factory.score_type2constructor[score_setting_name](eval_instance_name))
            scorers[score_setting_name] = get_scorers_factory(score_type2score_name).create_scorer(eval_instance_name)
        self._tm = TopicModel(label, artm_model, scorers)

    def _set_regularizers(self, reg_list):
        for reg_instance in reg_list:
            self._tm.add_regularizer(reg_instance)

def get_generic_topic_names(nb_topics):
    return ['top_' + index for index in map(lambda x: str(x) if len(str(x)) > 1 else '0'+str(x), range(1, nb_topics+1))]

class ModelLabelDisagreementException(Exception):
    def __init__(self, msg):
        super(ModelLabelDisagreementException, self).__init__(msg)
