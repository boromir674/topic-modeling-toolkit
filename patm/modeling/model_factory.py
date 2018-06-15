import artm
from .topic_model import TopicModel, TrainSpecs
from ..evaluation.scorer_factory import get_scorers_factory
from patm.utils import cfg2model_settings
from .regularizers import init_from_file, init_from_latest_results
from patm.definitions import collections_dir


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
        self.score2constructor = {
            'backgroung-tokens-ratio': lambda x: artm.BackgroundTokensRatioScore(name=x),
            'items-processed': lambda x: artm.ItemsProcessedScore(name=x),
            'perplexity': lambda x: artm.PerplexityScore(name=x, dictionary=self.dict),
            'sparsity-phi': lambda x: artm.SparsityPhiScore(name=x),
            'sparsity-theta': lambda x: artm.SparsityThetaScore(name=x),
            'theta-snippet': lambda x: artm.ThetaSnippetScore(name=x),
            'topic-mass-phi': lambda x: artm.TopicMassPhiScore(name=x),
            'topic-kernel': lambda x: artm.TopicKernelScore(name=x, probability_mass_threshold=0.3),
            'top-tokens': lambda x: artm.TopTokensScore(name=x)
        }
        self._tm = None

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
        regularizers = init_from_file(settings['regularizers'].items(), reg_cfg)
        scorers = {}
        model = artm.ARTM(num_topics=settings['learning']['nb_topics'], dictionary=self.dict)
        model.num_document_passes = settings['learning']['document_passes']
        self._set_scorers(settings['scores'])
        self._tm = TopicModel(label, model, scorers)
        self._set_regularizers(regularizers)
        specs = TrainSpecs(collection_passes=settings['learning']['collection_passes'])
        return self._tm, specs

    def create_model_with_phi_from_disk(self, phi_file_path, results):
        """
        Given a phi file path and a dictionary of experimental results, initializes a TopicModel object with the restored state of a model stored in disk. Configures to track the same
        evaluation metrics/scores. Uses the self.dictionary for indexing.\n
        Sets the below parameters with the latest corresponding values found in the expermental results:\n
        - number of phi-matrix-updates/passes-per-document\n
        - number of train iterations to perform on the whole documents collection\n
        - regularizers to use and their parameters\n
        :param str phi_file_path: strored phi matrix p_wt
        :param dict results: experimental results
        :return: tuple of initialized model and training specifications
        :rtype: (patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs)
        """
        regularizers = init_from_latest_results(results)
        scorers = {}
        model = artm.ARTM(num_topics=5, dictionary=self.dict) # number of topics have to be set, but doesn't matter, because they are overridden by the phi matrix loaded below
        model.load(filename=phi_file_path)
        model.num_document_passes = results['model_parameters']['document_passes'][-1][1]
        self._set_scorers(dict(results['evaluators']))
        self._tm = TopicModel(label, model, scorers)
        self._set_regularizers(regularizers)
        specs = TrainSpecs(collection_passes=results['collection_passes'][-1])
        return self._tm, specs

    def _set_scorers(self, score_type_name2score_name):
        for score_setting_name, eval_instance_name in score_type_name2score_name.items():
            self._tm.scores.add(self.score2constructor[score_setting_name](eval_instance_name))
            scorers[score_setting_name] = get_scorers_factory(score_type_name2score_name).create_scorer(eval_instance_name)

    def _set_regularizers(self, reg_list):
        for reg_instance in reg_list:
            self._tm.add_regularizer(reg_instance)

if __name__ == '__main__':
    sett = cfg2model_settings('/data/thesis/code/train.cfg')
    print sett
