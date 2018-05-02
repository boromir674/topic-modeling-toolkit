import ConfigParser
from collections import OrderedDict

import artm


class ModelFactory(object):

    def __init__(self, dictionary):
        self.dict = dictionary

    def _get_score_constructor(self, score):
        score2constructor = {
            'perplexity': lambda x: artm.PerplexityScore(name=x, dictionary=self.dict),
            'sparsity-phi': lambda x: artm.SparsityPhiScore(name=x),
            'sparsity-theta': lambda x: artm.SparsityThetaScore(name=x),
            'top-tokens': lambda x: artm.TopTokensScore(name=x)
        }
        return score2constructor[score]

    def create_model(self, cfg_file, dictionary):
        settings = cfg2model_settings(cfg_file)
        model = artm.ARTM(num_topics=settings['learning']['nb_topics'], dictionary=dictionary)

        for reg_setting_name, value in settings['regularizers'].iteritems():
            model.regularizers.add(regularizer2constructor[reg_setting_name](value))

        for score_setting_name, value in settings['scores'].iteritems():
            model.scores.add(self._get_score_constructor(score_setting_name)(value))

        model.num_document_passes = settings['learning']['document_passes']


def cfg2model_settings(cfg_file):
    config = ConfigParser.ConfigParser()
    config.read(cfg_file)
    return OrderedDict([(section, OrderedDict([(settings_name, section2encoder[section](value)) for setting_name, value in config.items(section)])) for section in config.sections()])


section2encoder = {
    'learning': int,
    'regularizers': str,
    'scores': str
}

regularizer2constructor = {
    'smooth-sparse-phi': lambda x: artm.SmoothSparsePhiRegularizer(name=x) if x else None,
    'smooth-sparse-theta': lambda x: artm.SmoothSparseThetaRegularizer(name=x) if x else None,
    'decorelate-phi': lambda x: artm.DecorrelatorPhiRegularizer(name=x) if x else None,
}

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
