from artm.scores import *
from abc import ABCMeta, abstractmethod
from patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME
import warnings


class MetaEvaluator:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__mro__ = [AbstractEvaluator]

    @abstractmethod
    def evaluate(self, data):
        raise NotImplementedError

    def __str__(self):
        return type(self).__name__

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractEvaluator:
            if any('evaluate' in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


class AbstractEvaluator(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def to_label(self):
        return str(self)

class ArtmEvaluator(AbstractEvaluator):
    """A wrapper class around each individual artm.BaseScore that provides a common 'evaluate' interface"""
    def __init__(self, name, artm_score, reportable_attributes):
        """
        :param str name: a custom name for this "evaluator"
        :param tuple attributes: the attributes that can be referenced from the artm.BaseScore object; supported reportable metrics/quantities
        """
        super(ArtmEvaluator, self).__init__(name)
        self._artm_score = artm_score
        self._attrs = reportable_attributes
        print name, self._attrs

    def evaluate(self, model):
        """
        Given an artm model object, this method provides an evaluation on all the supported reportable attributes and gives information for every training cycle performed so far.\n
        :param artm.ARTM model: the model to compute evaluation metrics on
        :return: the attribute => metrics-4all-cycles information
        :rtype: dict
        """
        return {attr: model.score_tracker[self.name].__getattribute__(attr) for attr in sorted(self._attrs)}
        # return {attr: model.score_tracker[self.name].__getattribute__('last_{}'.format(attr)) for attr in sorted(self._attrs)}

    @property
    def reportable_attributes(self):
        return self._attrs

    @property
    def artm_score(self):
        return self._artm_score


class PerplexityEvaluator(ArtmEvaluator):
    attributes = ('class_id_info', 'normalizer', 'raw', 'value')  # smaller the "value", better. bigger the "raw", better
    def __init__(self, name, dictionary):
        super(PerplexityEvaluator, self).__init__(name, PerplexityScore(name=name, class_ids=[DEFAULT_CLASS_NAME], dictionary=dictionary), self.attributes)


class SparsityPhiEvaluator(ArtmEvaluator):
    attributes = ('total_tokens', 'value', 'zero_tokens') # bigger value, better.
    def __init__(self, name, modality):
        assert modality in (DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME)
        super(SparsityPhiEvaluator, self).__init__(name, SparsityPhiScore(name=name, class_id=modality), self.attributes)
        self._modality = modality

    def to_label(self):
        return str(self) + self._modality[0] + ''.join(map(lambda x: x[0], self._modality[1:].split('_')))


class SparsityThetaEvaluator(ArtmEvaluator):
    attributes = ('total_topics', 'value', 'zero_topics')  # bigger value, better.
    def __init__(self, name, domain_topics):
        super(SparsityThetaEvaluator, self).__init__(name, SparsityThetaScore(name=name, topic_names=domain_topics), self.attributes)


class KernelEvaluator(ArtmEvaluator):
    """
    Threshold parameter is recommended to be set to 0.5 and higher, it should be set once and fixed.\n
    Only if p(t|w) > probability_mass_threshold then a token is considered to be part of the kernel.
    """
    attributes = ('average_coherence', 'average_contrast', 'average_purity', 'average_size', 'coherence', 'contrast', 'purity', 'size', 'tokens') # bigger value of contr, pur, coher, the better.
    def __init__(self, name, domain_topics, dictionary, threshold):
        assert 0 < threshold < 1
        if threshold < 0.5:
            warnings.warn("The value of 'probability_mass_threshold' parameter should be set to 0.5 or higher")
        super(KernelEvaluator, self).__init__(name, TopicKernelScore(name=name, class_id=DEFAULT_CLASS_NAME, topic_names=domain_topics, probability_mass_threshold=threshold, dictionary=dictionary), self.attributes)
        self._probability_mass_threshold = threshold

    def to_label(self):
        return '{}-{:.2f}'.format(self, self._probability_mass_threshold)


class TopTokensEvaluator(ArtmEvaluator):
    attributes = ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights')
    def __init__(self, name, domain_topics, dictionary, nb_top_tokens):
        assert 0 < nb_top_tokens
        super(TopTokensEvaluator, self).__init__(name, TopTokensScore(name=name, class_id=DEFAULT_CLASS_NAME, topic_names=domain_topics, num_tokens=nb_top_tokens, dictionary=dictionary), self.attributes)
        self._top_tokens = nb_top_tokens

    def to_label(self):
        return '{}{}'.format(self, self._top_tokens)


class BackgroundTokensRatioEvaluator(ArtmEvaluator):
    """
    Computes KL - divergence between p(t) and p(t | w) distributions \mathrm{KL}(p(t) | | p(t | w)) (or vice versa)
    for each token and counts the part of tokens that have this value greater than the given (non-negative) delta_threshold.
    Such tokens are considered to be background ones. Also returns all these tokens, if it was requested.
    """
    attributes = ('tokens', # the actual tokens with value greater than delta
                  'value')  # the percentage of tokens (against all tokens: background + domain-specific) belonging to background topics
    def __init__(self, name, delta_threshold):
        """
        :param str name: the name of this evaluator object
        :param float delta_threshold: the threshold for KL-div between p(t|w) and p(t) to get token into background. Should be non-negative
        """
        assert 0 <= delta_threshold
        super(BackgroundTokensRatioEvaluator, self).__init__(name, BackgroundTokensRatioScore(name=name, class_id=DEFAULT_CLASS_NAME, delta_threshold=delta_threshold), self.attributes)
        self._delta_threshold = delta_threshold

    def to_label(self):
        return '{}-{:.2f}'.format(self, self._delta_threshold)


MetaEvaluator.register(AbstractEvaluator)
