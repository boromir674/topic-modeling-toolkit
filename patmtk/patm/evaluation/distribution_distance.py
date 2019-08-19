import os
import re
from math import log
import pandas as pd
from patm.definitions import IDEOLOGY_CLASS_NAME


import logging
logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()

c_handler.setLevel(logging.INFO)


# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

c_handler.setFormatter(c_format)

logger.addHandler(c_handler)


class DivergenceComputer(object):  # In python 2 you MUST inherit from object to use @foo.setter feature!

    topic_extractors = {'all': lambda x: x.topic_names,
                        'domain': lambda x: x.domain_topics,
                        'background': lambda x: x.background_topics}

    def __init__(self, allowed_classes):
        self._classes = allowed_classes
        self._psi_prob_distros = {}
        self._tm = None

    @property
    def topic_model(self):
        return self._tm

    @topic_model.setter
    def topic_model(self, model):
        self._tm = model
        if model.label not in self._psi_prob_distros:
            self._psi_prob_distros[model.label] = self._get_psi_matrix(model)

    @property
    def psi_matrix(self):
        """[nb_classes X nb_topics] matrix holding p(c|t) probabilities\n
        :rtype: PsiMatrix
        """
        return self._psi_prob_distros[self._tm.label]

    def symmetric_KL(self, topic_model, class1, class2, topics):
        self.topic_model = topic_model
        s = 0
        for topic in self._topic_names(topic_model, topics):
            s += self._point_sKL(class1, class2, topic)
        return s

    def _point_sKL(self, c1, c2, topic):
        if self.p_ct(c1, topic) == 0 or self.p_ct(c2, topic) == 0:
            logger.warning("One of p(c|t) is zero: [{:.3f}, {:.3f}]. Skipping topic '{}' from the summation (over topics) of the symmetric KL formula, because none of limits [x->0], [y->0], [x,y->0] exist.".format(self.p_ct(c1, topic), self.p_ct(c2, topic), topic))
            return 0
        return self._point_KL(c1, c2, topic) + self._point_KL(c2, c1, topic)

    def _point_KL(self, c1, c2, topic):
        # if self.p_ct(c1, topic) == 0 or
        # try:
        return self.p_ct(c1, topic) * log(float(self.p_ct(c1, topic) / self.p_ct(c2, topic)))
        # except ValueError as e:
        #     print e
        #     print "Topic: {}, c1: {}, c2: {}\np(c1|t) = {:.3f}, p(c2|t) = {:.3f}, p(c1|t) / p(c2|t) = {:.5f}".format(topic, c1, c2, self.p_ct(c1, topic), self.p_ct(c2, topic), self.p_ct(c1, topic) / self.p_ct(c2, topic))
        #     import sys
        #     sys.exit(1)
    # def _index(self, c):
    #     if type(c) == str:
    #         try:
    #             c = self._classes.index(c)
    #         except ValueError:
    #             raise ValueError("Requested class '{}' but the defined allowed classes are [{}].".format(c, ', '.join(self._classes)))
    #     return int(c) + len(self._p_ct) - len(self._classes)

    def _topic_names(self, topic_model, topics):
        if type(topics) == str:
            topics = self.topic_extractors.get(topics, lambda x: topics)(topic_model)
        if not all(x in topic_model.topic_names for x in topics):
            raise RuntimeError("Not all the topic names given [{}] are in the defined topics [{}] of the input model '{}'".format(', '.join(topics), ', '.join(topic_model.topic_names), topic_model.label))
        return topics

    def p_ct(self, c, t):
        """
        Probability of class=c given topic=t: p(class=c|topic=t)\n
        :param str c:
        :param str or int t:
        :return:
        :rtype: float
        """
        return self.psi_matrix.p_ct(c, t)

    def _get_psi_matrix(self, topic_model):
        _ = topic_model.artm_model.get_phi()
        psi_matrix = _.set_index(pd.MultiIndex.from_tuples(_.index)).loc[IDEOLOGY_CLASS_NAME]
        if psi_matrix.shape != (len(self._classes), topic_model.nb_topics):
            raise RuntimeError("Shape of p_ct (Psi distribution; p(c|t)) {} does not match the expected one {}".format(psi_matrix.shape, (len(self._classes), topic_model.nb_topics)))
        return PsiMatrix(psi_matrix)

    @classmethod
    def from_dataset(cls, dataset_path):
        """Create a DivergenceComputer able to compute psi, p(c|t) probabilities, from the document labels found in the input dataset's vocabulary file.\n
        :param str dataset_path: path to a directory correspoonding to a computed dataset
        :return: the initialized divergence computer
        :rtype: DivergenceComputer
        """

        if not os.path.isdir(dataset_path):
            raise RuntimeError("Parameter dataset should be a valid dataset name or a full path to a dataset dir in collections. Instead we computed '{}' from the input '{}'".format(dataset_path, dataset))
        vocab_file = os.path.join(dataset_path, 'vocab.{}.txt'.format(dataset))
        # if not os.path.isfile(vocab_file):
        #     raise RuntimeError("Vocabulary file path inferred '{}' does not match an actual file".format(vocab_file))
        with open(vocab_file, 'r') as f:
            document_classes = re.findall(r'^(\w+) {}'.format(IDEOLOGY_CLASS_NAME), f.read(), re.M)
        return DivergenceComputer(document_classes)


class PsiMatrix:
    """Class x Topics matrix holdig p(c|t) probabilities"""
    def __init__(self, psi_dataframe):
        self._check_debug(psi_dataframe)
        self._psi = psi_dataframe

    @staticmethod
    def _check_debug(psi_dataframe):
        for i, t in enumerate(psi_dataframe):
            l = [psi_dataframe[t][x] for x in range(len(psi_dataframe[t]))]
            try:
                assert abs(sum(l) - 1) < 0.001
            except AssertionError:
                raise RuntimeError("{}: [{}] sum: {} abs-diff-with-zero: {}".format(t, ', '.join('{:.2f}'.format(x) for x in l), sum(l), abs(sum(l)-1)))

    def p_ct(self, c, t):
        """
        Probability of class=c given topic=t: p(class=c|topic=t)\n
        :param str c:
        :param str or int t:
        :return:
        :rtype: float
        """
        return self._psi.loc[c][t]

    def classes_distribution(self, topic):
        """Probabilities of classes conditioned on topic; p(c|topic=topic)\n
        :param str topic:
        :return: the p(c|topic) probabilities as an integer-indexable object
        :rtype: pandas.core.series.Series
        """
        return self._psi[topic]
