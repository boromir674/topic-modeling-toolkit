import os
import json
import artm
from .base_evaluator import AbstractEvaluator


class EvaluationFactory(object):
    def __init__(self, dictionary, cooc_dict):
        """

        :param artm.Dictionary dictionary:
        :param dict cooc_dict:
        """
        self._dict = dictionary
        self.cooc_df_dict = cooc_dict['df']['obj']

        self.score_type2constructor = {
            'background-tokens-ratio': lambda x: artm.BackgroundTokensRatioScore(name=x, delta_threshold=0.3),
            # Computes KL - divergence between p(t) and p(t | w) distributions \mathrm{KL}(p(t) | | p(t | w)) (or vice versa)
            # for each token and counts the part of tokens that have this value greater than given delta.
            # Such tokens are considered to be background ones. Also returns all these tokens, if it was requested.
            'items-processed': lambda x: artm.ItemsProcessedScore(name=x),
            'perplexity': lambda x: artm.PerplexityScore(name=x, dictionary=self._dict),
            'sparsity-phi': lambda x: artm.SparsityPhiScore(name=x),
            'sparsity-theta': lambda x: artm.SparsityThetaScore(name=x),
            'theta-snippet': lambda x: artm.ThetaSnippetScore(name=x),
            'topic-mass-phi': lambda x: artm.TopicMassPhiScore(name=x),
            'topic-kernel': lambda x: artm.TopicKernelScore(name=x, probability_mass_threshold=0.6, dictionary=self.cooc_df_dict),
            # p(t|w) > probability_mass_threshold
            'top-tokens-10': lambda x: artm.TopTokensScore(name=x, num_tokens=10, dictionary=self.cooc_df_dict),
            'top-tokens-100': lambda x: artm.TopTokensScore(name=x, num_tokens=100, dictionary=self.cooc_df_dict)
        }

    def create_artm_scorer(self, score_type, scorer_name):
        print score_type, scorer_name
        return self.score_type2constructor[score_type](scorer_name)


score_type2_reportables = {
    'background-tokens-ratio': ('tokens', # the actual tokens with value greater than delta
                                'value'), # the percentage of tokens (against all tokens: background + domain-specific) belonging to background topics
    'items-processed': 'value',
    'perplexity': ('class_id_info', 'normalizer', 'raw', 'value'),  # smaller the "value", better. bigger the "raw", better
    'sparsity-phi': ('total_tokens', 'value', 'zero_tokens'),  # bigger value, better.
    'sparsity-theta': ('total_topics', 'value', 'zero_topics'),  # bigger value, better.
    'theta-snippet': ('document_ids', 'snippet'),
    'topic-mass-phi': ('topic_mass', 'topic_ratio', 'value'),
    'topic-kernel': ('average_coherence', 'average_contrast', 'average_purity', 'average_size', 'coherence', 'contrast', 'purity', 'size', 'tokens'), # bigger value of contr, pur, coher, the better.
    'top-tokens': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights'),
    'top-tokens-10': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights'),
    'top-tokens-100': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights')
}


class ArtmScorer(AbstractEvaluator):

    def __init__(self, name, attributes):
        super(ArtmScorer, self).__init__(name)
        self._attrs = attributes

    def evaluate(self, model):
        return {attr: model.score_tracker[self.name].__getattribute__(attr) for attr in sorted(self._attrs)}
        # return {attr: model.score_tracker[self.name].__getattribute__('last_{}'.format(attr)) for attr in sorted(self._attrs)}

    @property
    def attributes(self):
        return self._attrs


class ArtmScorerFactory(object):
    def __init__(self, scorers):
        """
        :param scorers: a mapping of 'scorers' types to 'scorers' names. This structure can be parsed out of the 'scores' section of a 'train.cfg'
        :type scorers: dict
        """
        self.scorers = scorers

    def create_scorer(self, name):
        assert name in self.scorers.values()
        return ArtmScorer(name, tuple(score_type2_reportables[{v: k for k, v in self.scorers.items()}[name]]))


# TODO change this suspicious code
scorer_factories = {}
def get_scorers_factory(scorers):
    key = '.'.join(sorted(scorers.values()))
    if key not in scorer_factories:
        scorer_factories[key] = ArtmScorerFactory(scorers)
    return scorer_factories[key]
