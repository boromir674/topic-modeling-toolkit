import os
import json
from base_evaluator import *
from patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME


class EvaluationFactory(object):
    class_names = [DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME]

    def __init__(self, dictionary, cooc_dict, abbreviation2class_name=None):
        """
        :param artm.Dictionary dictionary:
        :param dict cooc_dict:
        """
        self._abbr2class = abbreviation2class_name
        if not abbreviation2class_name:
            self._abbr2class = {k: v for k, v in map(lambda x: tuple(['{}{}'.format(x[0], ''.join(map(lambda y: y[0], x[1:].split('_')))),
                                                                      x]), self.class_names)}
        self._dict = dictionary
        self.cooc_df_dict = cooc_dict['tf']['obj']
        self._domain_topics = []
        self._modality_weights = {}
        self.eval_constructor_hash = {
            'perplexity': lambda x: PerplexityEvaluator(x[0], self._dict, [DEFAULT_CLASS_NAME]),
            'sparsity-theta': lambda x: SparsityThetaEvaluator(x[0], self._domain_topics),
            'sparsity-phi': lambda x: SparsityPhiEvaluator(x[0], x[1]),
            'topic-kernel': lambda x: KernelEvaluator(x[0], self._domain_topics, self.cooc_df_dict, x[1]),
            'top-tokens': lambda x: TopTokensEvaluator(x[0], self._domain_topics, self.cooc_df_dict, int(x[1])),
            'background-tokens-ratio': lambda x: BackgroundTokensRatioEvaluator(x[0], x[1])}

    @property
    def domain_topics(self):
        return self._domain_topics

    @domain_topics.setter
    def domain_topics(self, domain_topics):
        self._domain_topics = domain_topics

    @property
    def modalities(self):
        return self._modality_weights

    @modalities.setter
    def modalities(self, modality_weights_dict):
        self._modality_weights = modality_weights_dict

    def create_evaluator(self, score_definition, score_name):
        """
        Call this method to construct a new artm scorer custom wrapper object parametrized by pasing the given definition.\n
        :param str score_definition: evaluators definitions that can be parsed on '-' ie 'perplexity', 'sparsity-phi-\@dc',
         'sparsity-phi-\@ic', 'topic-kernel-0.6', 'topic-kernel-0.8', 'top-tokens-10'
        :param str score_name: an arbitrary name
        :return: a reference to the constructed object
        :rtype: ArtmEvaluator
        """
        self._build_evaluator_definition(score_definition)
        return self.eval_constructor_hash['-'.join(self._tokens)]([score_name] + self._params)

    def _build_evaluator_definition(self, score_definition):
        self._tokens, self._params = [], []
        for el in score_definition.split('-'):
            try:
                self._params.append(float(el))
            except ValueError:
                if el[0] == '@':
                    modality_name = self._abbr2class[el]
                    if modality_name not in self._modality_weights:
                        raise RequiredModalityWeightNotFoundError("Modality named as '{}' has not been set with a positive weight".format(modality_name))
                    else:
                        self._params.append(modality_name)
                else:
                    self._tokens.append(el)


class RequiredModalityWeightNotFoundError(Exception):
    def __init__(self, msg):
        super(RequiredModalityWeightNotFoundError, self).__init__(msg)