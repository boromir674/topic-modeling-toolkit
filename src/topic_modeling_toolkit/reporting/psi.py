import os
from os import path
import re
from glob import glob
from collections import defaultdict
import attr
from math import log
from topic_modeling_toolkit.results.experimental_results import ExperimentalResults
import artm
import warnings
import pandas as pd


import logging
logger = logging.getLogger(__name__)

############### COMPUTER ############################
@attr.s
class DivergenceComputer(object):  # In python 2 you MUST inherit from object to use @foo.setter feature!
    pct_models = attr.ib(init=False, default={})
    __model = attr.ib(init=False)

    @property
    def psi(self):
        return self.pct_models[self.__model]

    @psi.setter
    def psi(self, psi_matrix):
        if psi_matrix.label not in self.pct_models:
            self.pct_models[psi_matrix.label] = {'obj': psi_matrix, 'distances': {}}
        self.__model = psi_matrix.label

    def __call__(self, *args, **kwargs):
        self._first_class = args[0]
        self._rest_classes = args[1:]
        return [self.symmetric_KL(self._first_class, c, kwargs['topics']) for c in self._rest_classes]

    def get_symmetric_KL(self, class1, class2, topics):
        return self.psi['distances'].get('{}-{}'.format(class1, class2),
                                         self.psi['distances'].get('{}-{}'.format(class2, class1),
                                                                   self.symmetric_KL(class1, class2, topics)))

    def symmetric_KL(self, class1, class2, topics):
        s = 0
        for topic in topics:
            s += self._point_sKL(class1, class2, topic)
        self.psi['distances']['{}-{}'.format(class1, class2)] = s
        return s

    def _point_sKL(self, c1, c2, topic):
        if self.p_ct(c1, topic) == 0 or self.p_ct(c2, topic) == 0:
            logger.warning(
                "One of p(c|t) is zero: [{:.3f}, {:.3f}]. Skipping topic '{}' from the summation (over topics) of the symmetric KL formula, because none of limits [x->0], [y->0], [x,y->0] exist.".format(
                    self.p_ct(c1, topic), self.p_ct(c2, topic), topic))
            return 0
        return self._point_KL(c1, c2, topic) + self._point_KL(c2, c1, topic)

    def _point_KL(self, c1, c2, topic):
        return self.p_ct(c1, topic) * log(float(self.p_ct(c1, topic) / self.p_ct(c2, topic)))

    def p_ct(self, c, t):
        """
        Probability of class=c given topic=t: p(class=c|topic=t)\n
        :param str c:
        :param str or int t:
        :return:
        :rtype: float
        """
        return self.psi['obj'].p_ct(c, t)


################# REPORTER #############################3

@attr.s
class PsiReporter(object):
    datasets = attr.ib(init=True, default={})
    _dataset_path = attr.ib(init=True, default='')
    _topics_extractors = attr.ib(init=False, default={'all': lambda x: x.domain_topics + x.background_topics,
                                          'domain': lambda x: x.domain_topics,
                                          'background': lambda x: x.background_topics})

    discoverable_class_modality_names = attr.ib(init=True, default=['@labels_class', '@ideology_class'])
    computer = attr.ib(init=False, default=DivergenceComputer())
    # dataset_name = attr.ib(init=False, default=attr.Factory(lambda self: path.basename(self._dataset_path), takes_self=True))
    has_registered_class_names = {}
    # models = attr.ib(init=False, default=) self._selected_topics
    _precision = attr.ib(init=False, default=2)
    _psi = attr.ib(init=False, default='')
    _selected_topics = attr.ib(init=False, default=[])

    @property
    def dataset(self):
        return self.datasets[self._dataset_path]

    @dataset.setter
    def dataset(self, dataset_path):
        if dataset_path not in self.datasets:
            self.datasets[dataset_path] = DatasetCollection(dataset_path, self.discoverable_class_modality_names)
        self._dataset_path = dataset_path
        if not hasattr(self.datasets[dataset_path], 'class_names'):
            raise RuntimeError(
                "A dataset '{}' object found without 'class_names' attribute".format(self.datasets[dataset_path].name))
        logger.info("Dataset '{}' at {}".format(self.datasets[dataset_path].name, self.datasets[dataset_path].dir_path))
        # logger.info("{}".format(str(self.datasets[dataset_path])))
        if not self.datasets[dataset_path].doc_labeling_modality_name:
            logger.warning("Dataset's '{}' vocabulary file has no registered tokens representing document class label names".format(self.datasets[dataset_path].name))

    @property
    def psi_matrix(self):
        return self._psi

    @psi_matrix.setter
    def psi_matrix(self, psi_matrix):
        if len(self.dataset.class_names) != psi_matrix.shape[0]:
            raise RuntimeError(
                "Number of classes do not correspond to the number of rows of Psi matrix. Found {} registered 'class names' tokens: [{}]. Psi matrix number of rows (classes) = {}.".
                format(len(self.dataset.class_names), self.dataset.class_names, psi_matrix.shape[0]))
        if len(self._topic_names) != psi_matrix.shape[1]:
            raise RuntimeError(
                "Number of topics in experimental results do not correspond to the number of columns rows of the Psi matrix. Found {} topics, while number of columns = {}".
                    format(len(self._topic_names), psi_matrix.shape[1]))
        self._psi = psi_matrix

    @property
    def topics(self):
        """The selected topics to sum over when computing the symmetric KL divergence"""
        return self._selected_topics

    @topics.setter
    def topics(self, topics):
        """The selected topics to sum over when computing the symmetric KL divergence"""
        if type(topics) == str:
            topics = self._topics_extractors[topics](self.exp_res.scalars)
        if not all(x in self._topic_names for x in topics):
            raise RuntimeError("Not all the topic names given [{}] are in the defined topics [{}] of the input model '{}'".format(', '.join(topics), ', '.join(self._topic_names), self.exp_res.scalars.model_label))
        self._selected_topics = topics

    def pformat(self, model_paths, topics_set='domain', show_model_name=True, show_class_names=True, precision=2):
        self._precision = precision
        b = []
        if self.dataset.doc_labeling_modality_name:
            for phi_path, json_path in self._all_paths(model_paths):
                # model_label = path.basename(json_path)
                # logger.info("Phi model '{}', experimentsl results '{}".format(phi_path, json_path))
                print("Phi model '{}', experimentsl results '{}".format(phi_path, json_path))
                model = self.artifacts(phi_path, json_path)
                is_WTDC_model = any(x in self.exp_res.scalars.modalities for x in self.discoverable_class_modality_names)
                if is_WTDC_model:
                    self.topics = topics_set
                    # if not self.dataset.doc_labeling_modality_name:
# warnings.warn("The document class modality (one of [{}]) was found in experimental results '{}', but dataset's vocabulary file '{}' does not contain registered tokens representing the unique document classes and thus phi matrix ( p(c|t) probabilities ) where probably not computed during training.".format(', '.join(sorted(self.discoverable_class_modality_names)), path.basename(json_path), path.basename(self.dataset.vocab_file)))
                    # else:

                    self.psi_matrix = PsiMatrix.from_artm(model, self.dataset.doc_labeling_modality_name)
                    # if len(self.dataset.class_names) != self.psi.shape[0]:
                    #     raise RuntimeError("Number of classes do not correspond to the number of rows of Psi matrix. Found {} registered 'class names' tokens: [{}]. Psi matrix number of rows (classes) = {}.".
                    #                        format(len(self.dataset.class_names), self.dataset.class_names, self.psi.shape[0]))
                    #
                    # if len(self._topic_names) != self.psi.shape[1]:
                    #     raise RuntimeError(
                    #         "Number of topics in experimental results do not correspond to the number of columns rows of the Psi matrix. Found {} topics, while number of columns = {}".
                    #         format(len(self._topic_names), self.psi.shape[1]))
                    b.append(self.divergence_str(topics_set=topics_set, show_model_name=show_model_name, show_class_names=show_class_names))
                else:
                    print("Skipping model '{}' since it does not utilize any document metadata, such as document labels".format(path.basename(phi_path.replace('.phi', ''))))
        return '\n\n'.join(b)

    def values(self, model_paths, topics_set='domain'):
        """
        :param model_paths:
        :param topics_set:
        :return: list of lists of lists
        """
        list_of_lists = []
        if self.dataset.doc_labeling_modality_name:
            for phi_path, json_path in self._all_paths(model_paths):
                logger.info("Phi model '{}', experimentsl results '{}".format(phi_path, json_path))
                model = self.artifacts(phi_path, json_path)
                is_WTDC_model = any(
                    x in self.exp_res.scalars.modalities for x in self.discoverable_class_modality_names)
                if is_WTDC_model:
                    self.topics = topics_set
                    self.psi_matrix = PsiMatrix.from_artm(model, self.dataset.doc_labeling_modality_name)
                    self.psi_matrix.label = self.exp_res.scalars.model_label
                    self.computer.psi = self.psi_matrix
                    self.computer.class_names = self.dataset.class_names
                    list_of_lists.append([self._values(i, c) for i, c in enumerate(self.dataset.class_names)])
                else:
                    logger.info(
                        "Skipping model '{}' since it does not utilize any document metadata, such as document labels".format(
                            path.basename(phi_path.replace('.phi', ''))))
        return list_of_lists

    def artifacts(self, *args):
        self.exp_res = ExperimentalResults.create_from_json_file(args[1])
        self._topic_names = self.exp_res.scalars.domain_topics + self.exp_res.scalars.background_topics
        _artm = artm.ARTM(topic_names=self.exp_res.scalars.domain_topics + self.exp_res.scalars.background_topics, dictionary=self.dataset.lexicon, show_progress_bars=False)
        _artm.load(args[0])
        return _artm

    def _all_paths(self, model_paths):
        for m in model_paths:
            yield self.paths(m)

    def paths(self, *args):
        if os.path.isfile(args[0]):  # is a full path to .phi file
            return args[0], path.join(path.dirname(args[0]), '../results', path.basename(args[0]).replace('.phi', '.json'))
        return os.path.join(self._dataset_path, 'models', args[0]), path.join(self._dataset_path, 'results', args[0]).replace('.phi', '.json')  # input is model label

    ###### STRING BUILDING
    def divergence_str(self, topics_set='domain', show_model_name=True, show_class_names=True):
        self._show_class_names = show_class_names

        self._reportable_class_strings = list(map(lambda x: x, self.dataset.class_names))
        self.__max_class_len = max(len(x) for x in self._reportable_class_strings)
        self._psi.label = self.exp_res.scalars.model_label
        self.computer.psi = self._psi
        self.computer.class_names = self.dataset.class_names


        string_values = [[self._str(x) for x in self._values(i, c)] for i, c in enumerate(self.dataset.class_names)]
        self.__max_len = max(max(len(x) for x in y) for y in string_values)
        _ = ''.join('{}\n'.format(self._pct_row(i, strings)) for i, strings in enumerate(string_values))
        if show_model_name:
            return "{}\n{}".format(self.exp_res.scalars.model_label, _)
        return _

    def _values(self, index, class_name):
        distances = list(self.computer(*list([class_name] + self.dataset.class_names[:index] + self.dataset.class_names[index + 1:]), topics=self._selected_topics))
        distances.insert(index, 0)
        assert len(distances) == len(self.dataset.class_names)
        return distances

    def _pct_row(self, row_index, strings):
        if self._show_class_names:
            return '{}{} {}'.format(self._reportable_class_strings[row_index],
                                    ' ' * (self.__max_class_len - len(self._reportable_class_strings[row_index])),
                                    ' '.join('{}{}'.format(x, ' '*(self.__max_len - len(x))) for x in strings))
        return ' '.join('{}{}'.format(x, ' '*(self.__max_len - len(x))) for x in strings)

    def _str(self, value):
        if value == 0:
            return ''
        return '{:.1f}'.format(value)
# def _cooc_tf(self, *args):
    #     if path.isfile(args[0]):  # is a full path to .phi file e.match(r'^ppmi_(\d+)_([td]f)\.txt$', name)
    #         c = glob('{}/ppmi_*\.txt'.format(path.join(os.path.dirname(args[0]), '../')))
    #     else:
    #         c = glob('{}/ppmi_*\.txt'.format(self._dataset_path))
    #     if not c:
    #         raise RuntimeError("Did not find any 'ppmi' files in dataset directory '{}'".format(path.dirname(args[0]), '../'))
    #     return c[0]
    #
    # def _class_names(self, allowed_modality_names):
    #     """Call this method to extract possible set of document class names out of the dataset's vocabulary file and the discoverred modality name serving to the p(c|t) \psi model.
    #         Returns None if the dataset's vocabulary does not contain registered terms as the unique document class names"""
    #     vocab_file = path.join(self._dataset_path, 'vocab.{}.txt'.format(self.dataset_name))
    #     with open(vocab_file, 'r') as f:
    #         classname_n_modality_tuples = re.findall(r'^(\w+) ({})'.format('|'.join(allowed_modality_names)), f.read(), re.M)
    #         if not classname_n_modality_tuples:
    #             return [], ''
    #         modalities = set([modality_name for _, modality_name in classname_n_modality_tuples])
    #         if len(modalities) > 1:
    #             raise ValueError("More than one candidate modalities found to serve as the document classification scheme: [{}]".format(sorted()))
    #         document_classes = [class_name for class_name, _ in classname_n_modality_tuples]
    #         if len(document_classes) > 6:
    #             warnings.warn("Detected {} classes for dataset '{}'. Perhaps too many classes for a collection of {} documents. You can define a different discretization scheme (binning of the political spectrum)".format(len(document_classes), self.dataset_name, self.nb_docs))
    #         return document_classes, modalities.pop()
    #
    # @property
    # def nb_docs(self):
    #     return self.file_len(pth.join(self._dataset_path, 'vowpal.{}.txt'.format(self.dataset_name)))
    #
    # def file_len(self, file_path):
    #     with open(file_path) as f:
    #         return len([None for i, _ in enumerate(f)])

##################### PSI MATRIX #############################

def _valid_probs(instance, attribute, value):
    for i, topic in enumerate(value):
        topic_specific_class_probabilities = [value[topic][x] for x in range(len(value[topic]))]
        try:
            assert abs(sum(topic_specific_class_probabilities) - 1) < 0.001
        except AssertionError:
            raise RuntimeError("{}: [{}] sum: {} abs-diff-with-zero: {}".format(topic, ', '.join('{:.2f}'.format(x) for x in topic_specific_class_probabilities), sum(topic_specific_class_probabilities), abs(sum(topic_specific_class_probabilities) - 1)))

@attr.s
class PsiMatrix(object):
    """Class x Topics matrix holdig p(c|t) probabilities \forall c \in C and t \in T"""
    dataframe = attr.ib(init=True, validator=_valid_probs)
    shape = attr.ib(init=False, default=attr.Factory(lambda self: self.dataframe.shape, takes_self=True))

    def __str__(self):
        return str(self.dataframe)

    def iter_topics(self):
        return (topic_name for topic_name in self.dataframe)

    def iterrows(self):
        return self.dataframe.iterrows()

    def itercolumns(self):
        return self.dataframe.iteritems()

    def p_ct(self, c, t):
        """
        Probability of class=c given topic=t: p(class=c|topic=t)\n
        :param str c:
        :param str or int t:
        :return:
        :rtype: float
        """
        return self.dataframe.loc[c][t]

    def classes_distribution(self, topic):
        """Probabilities of classes conditioned on topic; p(c|topic=topic)\n
        :param str topic:
        :return: the p(c|topic) probabilities as an integer-indexable object
        :rtype: pandas.core.series.Series
        """
        return self.dataframe[topic]

    @classmethod
    def from_artm(cls, artm_model, modality_name):
        phi = artm_model.get_phi()
        psi_matrix = phi.set_index(pd.MultiIndex.from_tuples(phi.index)).loc[modality_name]
        return PsiMatrix(psi_matrix)


########################## DATASET ###########################

def _id_dir(instance, attribute, value):
    if not path.isdir(value):
        raise IOError("'{}' is not a valid directory path".format(value))

def _class_names(self, attribute, value):
    """Call this method to extract possible set of document class names out of the dataset's vocabulary file and the discoverred modality name serving to the p(c|t) \psi model.
        Returns None if the dataset's vocabulary does not contain registered terms as the unique document class names"""
    vocab_file = path.join(self.dir_path, 'vocab.{}.txt'.format(self.name))
    with open(vocab_file, 'r') as f:
        classname_n_modality_tuples = re.findall(r'(\w+)[\t\ ]({})'.format('|'.join(x for x in self.allowed_modality_names)), f.read())

        if not classname_n_modality_tuples:
            self.class_names = []
            self.doc_labeling_modality_name = ''
        else:
            modalities = set([modality_name for _, modality_name in classname_n_modality_tuples])
            if len(modalities) > 1:
                raise ValueError("More than one candidate modalities found to serve as the document classification scheme: [{}]".format(sorted(x for x in modalities)))
            document_classes = [class_name for class_name, _ in classname_n_modality_tuples]
            warn_threshold = 8
            if len(document_classes) > warn_threshold:
                warnings.warn("Detected {} classes for dataset '{}'. Perhaps too many classes for a collection of {} documents. You can define a different discretization scheme (binning of the political spectrum)".format(len(document_classes), self.name, self.nb_docs))
            self.class_names = document_classes
            self.doc_labeling_modality_name = modalities.pop()


def _file_len(file_path):
    with open(file_path) as f:
        return len([None for i, _ in enumerate(f)])


@attr.s
class DatasetCollection(object):
    dir_path = attr.ib(init=True, converter=str, validator=_id_dir, repr=True)

    allowed_modality_names = attr.ib(init=True, default=['@labels_class', '@ideology_class'])
    name = attr.ib(init=False, default=attr.Factory(lambda self: path.basename(self.dir_path), takes_self=True))
    vocab_file = attr.ib(init=False, default=attr.Factory(lambda self: path.join(self.dir_path, 'vocab.{}.txt'.format(self.name)), takes_self=True))
    lexicon = attr.ib(init=False, default=attr.Factory(lambda self: artm.Dictionary(name=self.name), takes_self=True))
    doc_labeling_modality_name = attr.ib(init=False, default='')
    class_names = attr.ib(init=False, default=[], validator=_class_names)
    # nb_docs = attr.ib(init=False, default=attr.Factory(lambda self: _file_len(path.join(self.dir_path, 'vowpal.{}.txt'.format(self.name))), takes_self=True))
    ppmi_file = attr.ib(init=False, default=attr.Factory(lambda self: self._cooc_tf(), takes_self=True))

    def __attrs_post_init__(self):
        self.lexicon.gather(data_path=self.dir_path,
                                       cooc_file_path=self.ppmi_file,
                                       vocab_file_path=self.vocab_file,
                                       symmetric_cooc_values=True)

    def _cooc_tf(self):
        c = glob('{}/ppmi_*tf.txt'.format(self.dir_path))
        if not c:
            raise RuntimeError("Did not find any 'ppmi' (computed with simple 'tf' scheme) files in dataset directory '{}'".format(self.dir_path))
        return c[0]
