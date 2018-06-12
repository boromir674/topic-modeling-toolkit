from configparser import ConfigParser
from collections import OrderedDict

import artm
from .topic_model import TopicModel, TrainSpecs
from ..evaluation.scorer_factory import ArtmScorerFactory

dicts2model_factory = {}


def get_model_factory(dictionary):
    if dictionary not in dicts2model_factory:
        dicts2model_factory[dictionary] = ModelFactory(dictionary)
        # print 'using new oblect'
        # print 'using existing object'
    return dicts2model_factory[dictionary]


class ModelFactory(object):

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

    def create_model(self, label, cfg_file, output_dir):
        """
        Creates an artm _topic_model instance based on the input config file.\n
        :param str label: the unique identifier for the model
        :param str cfg_file: A cfg file containing initial model parameters, regularizers and score metrics to use
        :param str output_dir: the top level directory to use for dumping files related to training
        :return: tuple of initialized model and
        :rtype: patm.modeling.topic_model.TopicModel, patm.modeling.topic_model.TrainSpecs
        """
        """
        
        :param str label: a unique identifier
        :param str cfg_file:
        :param str output_dir:
        :rtype: patm.modeling.topic_model.TopicModel
        :return:
        """
        settings = cfg2model_settings(cfg_file)
        scorers = {}
        model = artm.ARTM(num_topics=settings['learning']['nb_topics'], dictionary=self.dict)
        model.num_document_passes = settings['learning']['document_passes']

        for score_setting_name, value in settings['scores'].iteritems():
            model.scores.add(self.score2constructor[score_setting_name](value))
            scorers[score_setting_name] = get_scorers_factory(settings['scores']).create_scorer(value)

        tm = TopicModel(label, model, scorers)

        for reg_instance in regularizers:
            tm.add_regularizer(reg_instance)

        print 'REGULARIZERS:', ', '.join(x for x in model.regularizers.data)

        specs = TrainSpecs(collection_passes=settings['learning']['collection_passes'], output_dir=output_dir)

        return tm, specs


def cfg2model_settings(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)
    return OrderedDict([(section.encode('utf-8'), OrderedDict([(setting_name.encode('utf-8'), section2encoder[section](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])


section2encoder = {
    'learning': int,
    'regularizers': str,
    'scores': str
}


if __name__ == '__main__':
    sett = cfg2model_settings('/data/thesis/code/train.cfg')
    print sett

