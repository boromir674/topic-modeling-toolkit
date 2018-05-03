from configparser import ConfigParser
from collections import OrderedDict

import artm

dicts2model_factory = {}


def get_model_factory(dictionary):
    if dictionary not in dicts2model_factory:
        dicts2model_factory[dictionary] = ModelFactory(dictionary)
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
            'topic-kernel': lambda x: artm.TopicKernelScore(name=x),
            'top-tokens': lambda x: artm.TopTokensScore(name=x)
        }
        self.regularizer2constructor = {
            'kl-function-info': lambda x: artm.KlFunctionInfo(),
            'smooth-sparse-phi': lambda x: artm.SmoothSparsePhiRegularizer(name=x) if x else None,
            'smooth-sparse-theta': lambda x: artm.SmoothSparseThetaRegularizer(name=x) if x else None,
            'decorrelator-phi': lambda x: artm.DecorrelatorPhiRegularizer(name=x) if x else None,
            'label-regularization-phi': lambda x: artm.LabelRegularizationPhiRegularizer(name=x),
            'specified-sparse-phi': lambda x: artm.SpecifiedSparsePhiRegularizer(name=x),
            'improve-coherence-phi': lambda x: artm.ImproveCoherencePhiRegularizer(name=x),
            'smooth-ptdw': lambda x: artm.SmoothPtdwRegularizer(name=x),
            'topic-selection': lambda x: artm.TopicSelectionThetaRegularizer(name=x)
        }

    def create_model(self, cfg_file):
        settings = cfg2model_settings(cfg_file)
        print settings
        model = artm.ARTM(num_topics=settings['learning']['nb_topics'], dictionary=self.dict)
        for reg_setting_name, value in settings['regularizers'].iteritems():
            model.regularizers.add(self.regularizer2constructor[reg_setting_name](value))

        for score_setting_name, value in settings['scores'].iteritems():
            model.scores.add(self.score2constructor[score_setting_name](value))

        model.num_document_passes = settings['learning']['document_passes']
        return model, settings['learning']['collection_passes']


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
