import os
from os import path
import re
from glob import glob
from collections import defaultdict
import attr
from results.experimental_results import ExperimentalResults
import artm
import warnings
import pandas as pd


@attr.s
class PsiReporter(object):

    datasets = attr.ib(init=True, default={})
    _dataset_path = attr.ib(init=True, default='')
    topics = attr.ib(init=False, default={'all': lambda x: x.topic_names,
                                          'domain': lambda x: x.domain_topics,
                                          'background': lambda x: x.background_topics})
    # lexicons = attr.ib(init=False, default={})
    # class_names = attr.ib(init=False, default={})
    # modality_names = attr.ib(init=False, default={})
    discoverable_class_modality_names = attr.ib(init=True, default=['@labels_class', '@ideology_class'])
    # dataset_name = attr.ib(init=False, default=attr.Factory(lambda self: path.basename(self._dataset_path), takes_self=True))
    has_registered_class_names = {}
    # models = attr.ib(init=False, default=)

    @property
    def dataset(self):
        return self.datasets[self._dataset_path]

    @dataset.setter
    def dataset(self, dataset_path):
        if dataset_path not in self.datasets:
            self.datasets[dataset_path] = DatasetCollection(dataset_path, self.discoverable_class_modality_names)
        self._dataset_path = dataset_path

    def pformat(self, model_paths, topics_set='domain', show_class_names=True, show_topic_names=True, precision=2):
        for phi_path, json_path in self._all_paths(model_paths):
            # model_label = path.basename(json_path)
            print(phi_path, json_path)
            model, exp_res = self.artifacts(phi_path, json_path)
            is_WTDC_model = any(x in exp_res.scalars.modalities for x in self.discoverable_class_modality_names)
            if is_WTDC_model:

                if not self.dataset.doc_labeling_modality_name:
                    warnings.warn("The document class modality (one of [{}]) was found in experimental results '{}', but dataset's vocabulary file '{}' does not contain registered tokens representing the unique document classes and thus phi matrix ( p(c|t) probabilities ) where probably not computed during training.".format(', '.join(sorted(self.discoverable_class_modality_names)), path.basename(json_path), path.basename(self.dataset.vocab_file)))
                else:
                    psi = PsiMatrix.from_artm(_artm)
                    if len(self.dataset.class_names) != psi.shape[0]:
                        raise RuntimeError("Found {} registered 'class names' tokens: [{}]. Psi matrix number of rows (classes) = {}".format(len(self.dataset.class_names), self.dataset.class_names, psi.shape[0]))
                    print(psi)

            # TODO remove argument from Experiment constructor. make a self.exp and set cur_dir of its model and resuls writer/loaders

    def _str(self, psi_matrix, topics_set='domain', show_class_names=True, show_topic_names=True, precision=2):
        pass

    def artifacts(self, *args):
        # phi_path, result_path = self.paths(*args)

        exp_res = ExperimentalResults.create_from_json_file(args[1])
        _artm = artm.ARTM(topic_names=exp_res.scalars.domain_topics + exp_res.scalars.background_topics, dictionary=self.dataset.lexicon, show_progress_bars=False)
        _artm.load(args[0])
        return _artm, exp_res

    def topic_names(self, topics_set):
        return {'all': lambda x: x.topic_names}.get(topics_set, )

    def _all_paths(self, model_paths):
        for m in model_paths:
            yield self.paths(m)

    def paths(self, *args):
        if os.path.isfile(args[0]):  # is a full path to .phi file
            return args[0], path.join(path.dirname(args[0]), '../results', path.basename(args[0]).replace('.phi', '.json'))
        return os.path.join(self._dataset_path, 'models', args[0]), path.join(self._dataset_path, 'results', args[0]).replace('.phi', '.json')  # input is model label
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
    def from_artm(cls, artm_model):
        phi = artm_model.get_phi()
        psi_matrix = phi.set_index(pd.MultiIndex.from_tuples(phi.index)).loc[IDEOLOGY_CLASS_NAME]
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
        _ = f.read()
        print(type(_))
        classname_n_modality_tuples = re.findall(r"([\w_]+)['\"]*[\t\ ]({})".format('|'.join(self.allowed_modality_names)),
                                                 _, re.M)
        if not classname_n_modality_tuples:
            return [], ''
        modalities = set([modality_name for _, modality_name in classname_n_modality_tuples])
        if len(modalities) > 1:
            raise ValueError(
                "More than one candidate modalities found to serve as the document classification scheme: [{}]".format(
                    sorted(x for x in modalities)))
        document_classes = [class_name for class_name, _ in classname_n_modality_tuples]
        if len(document_classes) > 6:
            warnings.warn(
                "Detected {} classes for dataset '{}'. Perhaps too many classes for a collection of {} documents. You can define a different discretization scheme (binning of the political spectrum)".format(
                    len(document_classes), self.name, self.nb_docs))
        return document_classes, modalities.pop()


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
    class_names = attr.ib(init=False, validator=_class_names)
    # nb_docs = attr.ib(init=False, default=attr.Factory(lambda self: _file_len(path.join(self.dir_path, 'vowpal.{}.txt'.format(self.name))), takes_self=True))
    ppmi_file = attr.ib(init=False, default=attr.Factory(lambda self: self._cooc_tf(), takes_self=True))

    def __attrs_post_init__(self):
        self.lexicon.gather(data_path=self.dir_path,
                                       cooc_file_path=path.join(self.dir_path, self._cooc_tf()),
                                       vocab_file_path=path.join(self.vocab_file),
                                       symmetric_cooc_values=True)

    def _cooc_tf(self):
        # if path.isfile(args[0]):  # is a full path to .phi file e.match(r'^ppmi_(\d+)_([td]f)\.txt$', name)
        #     c = glob('{}/ppmi_*\.txt'.format(path.join(os.path.dirname(args[0]), '../')))
        # else:
        c = glob('{}/ppmi_*tf.txt'.format(self.dir_path))
        if not c:
            raise RuntimeError("Did not find any 'ppmi' (computed with simple 'tf' scheme) files in dataset directory '{}'".format(self.dir_path))
        return c[0]