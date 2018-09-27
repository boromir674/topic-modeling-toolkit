import os
import json
from base_evaluator import *
from patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME


class EvaluationFactory(object):
    abbreviation2_class_name = {'@dc': DEFAULT_CLASS_NAME, '@ic': IDEOLOGY_CLASS_NAME}
    def __init__(self, dictionary, cooc_dict):
        """
        :param artm.Dictionary dictionary:
        :param dict cooc_dict:
        """
        self._dict = dictionary
        self.cooc_df_dict = cooc_dict['df']['obj']
        self._domain_topics = []
        self.eval_constructor_hash = {
            'perplexity': lambda x: PerplexityEvaluator(x, self._dict),
            'sparsity-theta': lambda x: SparsityThetaEvaluator(x, self._domain_topics),
            'sparsity-phi': lambda x: SparsityPhiEvaluator(x[0], x[1]),
            'topic-kernel': lambda x: KernelEvaluator(x[0], self._domain_topics, self.cooc_df_dict, x[1]),
            'top-tokens': lambda x: TopTokensEvaluator(x[0], self._domain_topics, self.cooc_df_dict, x[1]),
            'background-tokens-ratio': lambda x: BackgroundTokensRatioEvaluator(x[0], x[1])
        }

    @property
    def domain_topics(self):
        return self._domain_topics

    @domain_topics.setter
    def domain_topics(self, domain_topics):
        self._domain_topics = domain_topics

    def create_evaluator(self, score_definition, score_name):
        """
        :param str score_definition:
        :param str score_name:
        :return:
        :rtype: ArtmEvaluator
        """
        self._build_evaluator_definition(score_definition)
        return self.eval_constructor_hash['-'.join(self._tokens)](score_name, *self._params)

    def _build_evaluator_definition(self, score_definition):
        self._tokens, self._params = [], []
        for el in score_definition.split('-'):
            try:
                self._params.append(float(el))
            except ValueError:
                if el[0] == '@':
                    self._params.append(self.abbreviation2_class_name[el])
                else:
                    self._tokens.append(el)
#
#     def create_artm_scorer(self, score_type, scorer_name):
#         """
#         :param str score_type:
#         :param str scorer_name:
#         :return:
#         :rtype: artm.scores.BaseScore
#         """
#         print score_type, scorer_name
#         return self.score_type2constructor[score_type](scorer_name)
#
#
# score_type2_reportables = {
#     'background-tokens-ratio': ('tokens', # the actual tokens with value greater than delta
#                                 'value'), # the percentage of tokens (against all tokens: background + domain-specific) belonging to background topics
#     'items-processed': 'value',
#     'perplexity': ('class_id_info', 'normalizer', 'raw', 'value'),  # smaller the "value", better. bigger the "raw", better
#     'sparsity-phi': ('total_tokens', 'value', 'zero_tokens'),  # bigger value, better.
#     'sparsity-theta': ('total_topics', 'value', 'zero_topics'),  # bigger value, better.
#     'theta-snippet': ('document_ids', 'snippet'),
#     'topic-mass-phi': ('topic_mass', 'topic_ratio', 'value'),
#     'topic-kernel': ('average_coherence', 'average_contrast', 'average_purity', 'average_size', 'coherence', 'contrast', 'purity', 'size', 'tokens'), # bigger value of contr, pur, coher, the better.
#     'top-tokens': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights'),
#     'top-tokens-10': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights'),
#     'top-tokens-100': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights')
# }
#
#
# class ArtmScorerFactory(object):
#     def __init__(self, scorers):
#         """
#         :param dict scorers: a str => str mapping of 'scorers' types to 'scorers' names. This structure can be parsed out of the 'scores' section of a 'train.cfg'
#         """
#         self.scorers = scorers
#         self._reversed_scorers = {v: k for k, v in self.scorers.items()}
#
#     def create_scorer(self, name):
#         """
#         Given a name which should be present as a value in the self.scorers dictionary, constructs an ArtScorer wrapper object.\n
#         :param str name: the supported name of the artm score object
#         :return: the scorer object reference
#         :rtype: ArtmEvaluator
#         """
#         assert name in self.scorers.values()
#         return ArtmScorer(name, tuple(score_type2_reportables[self._reversed_scorers[name]]))
#
#
# # TODO change this suspicious code
# scorer_factories = {}
# def get_scorers_factory(scorers):
#     """
#     :param dict scorers:
#     :return:
#     :rtype: ArtmScorerFactory
#     """
#     key = '.'.join(sorted(scorers.values()))
#     if key not in scorer_factories:
#         scorer_factories[key] = ArtmScorerFactory(scorers)
#     return scorer_factories[key]
