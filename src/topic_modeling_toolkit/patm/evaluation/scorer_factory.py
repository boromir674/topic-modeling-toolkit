import os
import json
from .base_evaluator import *
from topic_modeling_toolkit.patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME

import logging
logger = logging.getLogger(__name__)


class EvaluationFactory(object):
    """Supports evaluating TopicModels by computing 'model perplexity', 'sparsity of matrix Phi', 'sparsity of matrix Theta',
    coherence, contrast and purity metrics computed and averaged for all topics' kernels, coherence on x number of most probable token averaged
    for all topics and 'background tokens ratio' metrics."""
    class_names = [DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME]

    def __init__(self, dictionary, abbreviation2class_name=None):
        """
        :param artm.Dictionary dictionary: this is the dictionary used to compute cooc tf/df and pmi values
        :param dict cooc_dict: a hash of dictionaries (ccoc tf/df and pmi tf/df) (probably not needed)
        """
        self._abbr2class = abbreviation2class_name
        if not abbreviation2class_name:
            self._abbr2class = {k: v for k, v in map(lambda x: tuple(['{}{}'.format(x[0], ''.join(map(lambda y: y[0], x[1:].split('_')))),
                                                                      x]), self.class_names)}
        # self._dict = dictionary
        self.cooc_tf_artm_dictionary = dictionary
        self._domain_topics = []
        self._modality_weights = {}
        self.eval_constructor_hash = {
            'perplexity': lambda x: PerplexityEvaluator(x[0], self.cooc_tf_artm_dictionary, [DEFAULT_CLASS_NAME]),
            'sparsity-theta': lambda x: SparsityThetaEvaluator(x[0], self._domain_topics),
            'sparsity-phi': lambda x: SparsityPhiEvaluator(x[0], x[1]),
            'topic-kernel': lambda x: KernelEvaluator(x[0], self._domain_topics, self.cooc_tf_artm_dictionary, x[1]),
            'top-tokens': lambda x: TopTokensEvaluator(x[0], self._domain_topics, self.cooc_tf_artm_dictionary, int(x[1])),
            'background-tokens-ratio': lambda x: BackgroundTokensRatioEvaluator(x[0], x[1])}
        logger.info("Initialized EvaluationFactory with dict to supply to artm scorers (Perplexity, TopTokens and Kernel coherence): {} ".format(self.cooc_tf_artm_dictionary))

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


class RequiredModalityWeightNotFoundError(Exception): pass
