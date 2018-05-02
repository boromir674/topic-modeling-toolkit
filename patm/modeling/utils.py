from configparser import ConfigParser
from collections import OrderedDict

import artm


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
            'decorelator-phi': lambda x: artm.DecorrelatorPhiRegularizer(name=x) if x else None,
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
        return model


def cfg2model_settings(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)
    return OrderedDict([(section, OrderedDict([(setting_name, section2encoder[section](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])


section2encoder = {
    'learning': int,
    'regularizers': str,
    'scores': str
}

# def get_regularizer()
# model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer'))
# model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer'))
# model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'))
# model.regularizers['sparse_phi_regularizer'].tau = -1.0
# model.regularizers['sparse_theta_regularizer'].tau = -0.5
# model.regularizers['decorrelator_phi_regularizer'].tau = 1e+5

# model.scores.add(artm.PerplexityScore(name='my_first_perplexity_score', dictionary=dictionary))
# model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
# model.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
# model.scores.add(artm.TopTokensScore(name='top_tokens_score'))

if __name__ == '__main__':
    sett = cfg2model_settings('/data/thesis/code/train.cfg')
    print sett

