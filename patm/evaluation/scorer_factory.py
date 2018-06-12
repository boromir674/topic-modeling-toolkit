import os
import json
from .base_evaluator import AbstractEvaluator


scorer_type2_reportables = {
    'background-tokens-ratio': ('tokens', 'value'),
    'items-processed': 'value',
    'perplexity': ('class_id_info', 'normalizer', 'raw', 'value'),  # bigger value, better. bigger raw, worse
    'sparsity-phi': ('total_tokens', 'value', 'zero_tokens'),  # bigger value, better.
    'sparsity-theta': ('total_topics', 'value', 'zero_topics'),  # bigger value, better.
    'theta-snippet': ('document_ids', 'snippet'),
    'topic-mass-phi': ('topic_mass', 'topic_ratio', 'value'),
    'topic-kernel': ('average_coherence', 'average_contrast', 'average_purity', 'average_size', 'coherence', 'contrast', 'purity', 'size', 'tokens'),  # the bigger the value (contr, pur, coher), the better.
    'top-tokens': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights')
}


class ArtmScorer(AbstractEvaluator):

    def __init__(self, name, attributes):
        super(ArtmScorer, self).__init__(name)
        self._attrs = attributes

    def evaluate(self, model):
        return {attr: model.score_tracker[self.name].__getattribute__(attr).value for attr in sorted(self._attrs)}

    @property
    def atttributes(self):
        return self._attrs


class ArtmScorerFactory(object):
    def __init__(self, model, scorers_dict):
        self._model = model
        self.scorers = scorers_dict

    def create_scorer(self, name):
        assert name in self.scorers.values()
        return ArtmScorer(mame, tuple(scorer_type2_reportables[{v: k for k, v in self.scorers.items()}[name]]))


# TODO change this suspicious code
def get_scorers_factory(scorers):
    key = '.'.join(sorted(scorers.values()))
    if key not in scorer_factories:
        scorer_factories[key] = ArtmScorerFactory(scorers)
    return scorer_factories[key]
