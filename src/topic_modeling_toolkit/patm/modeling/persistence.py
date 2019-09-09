import os
import abc
from glob import glob
from topic_modeling_toolkit.patm.definitions import MODELS_DIR_NAME, RESULTS_DIR_NAME
import attr
from topic_modeling_toolkit.results import ExperimentalResults

import logging
logger = logging.getLogger(__name__)


class WriterLoader(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        raise NotImplemented
#
# @attr.s(repr=True, slots=False)
# class PWL(object):
#     location = attr.ib(init=True, converter=str, repr=True)
#     extension = attr.ib(init=True, converter=str, repr=True)
#     _exp = attr.ib(init=True, repr=True)
#     split = attr.ib(init=True, default='', converter=str, repr=True)
#     _post = attr.ib(init=False, default=attr.Factory(lambda self: '-{}'.format(self.split) if self.split else '', takes_self=True), repr=False)
#     saved = attr.ib(init=False, default=[], repr=False)
#     list = attr.ib(init=False, default=[], repr=True)
#
#     def __attrs_post_init__(self):
#         if not os.path.isdir(self.location):
#             os.mkdir(self.location)
#             logger.info("Created '{}' dir to persist experimental results".format(self.location))
#
#     @property
#     def list(self):
#         """
#         This property dynamically holds the files' names, which are found in the saving location
#         :return:
#         :rtype: list
#         """
#         return [os.path.basename(file_path[:file_path.index(self.extension)]) for file_path in glob('{}/*{}'.format(self.location, ''.join(filter(None, [self._post, self.extension]))))]
#
#     def get_full_path(self, name):
#         return os.path.join(self.location, ''.join(str(x) for x in [name, self._post, self.extension]))
#
#     def save(self, *args, **kwargs):
#         self.saved.append(args[0])
#
#
# @attr.s(repr=True, slots=False)
# class ResultsWL(PWL, WriterLoader):
#     """Class responsible for saving and loading experimental results (eg model training measurements/metrics)"""
#
#     @classmethod
#     def from_experiment(cls, exp, split=''):
#         return ResultsWL(os.path.join(exp.current_root_dir, RESULTS_DIR_NAME), '.json', exp, split)
#
#     def save(self, name):
#         results_file_path = self.get_full_path(name)
#         if not str(results_file_path).endswith(self.extension):
#             raise RuntimeError("Transformed '{}' to '{}' but still experimental results file name does not end with extension '{}'. split = '{}'".format(name, results_file_path, self.extension, self._post))
#         results = ExperimentalResults.create_from_experiment(self._exp)
#         results.save_as_json(results_file_path, human_redable=True)
#         super(ResultsWL, self).save(results_file_path)
#         # self.saved.append(results_file_path)
#
#     def load(self, name):
#         return ExperimentalResults.create_from_json_file(self.get_full_path(name))
#
#
# @attr.s(repr=True, slots=True)
# class ModelWL(PWL, WriterLoader):
#     """Class responsible for saving and loading topic models"""
#     matrix_label = attr.ib(init=False, default='p_wt', repr=True)
#
#     @classmethod
#     def from_experiment(cls, exp, split=''):
#         return ModelWL(os.path.join(exp.current_root_dir, MODELS_DIR_NAME), '.phi', exp, split)
#
#     def save(self, name):
#         file_path = self.get_full_path(name)
#         self._exp.topic_model.artm_model.save(file_path, model_name=self.matrix_label)  # saves one Phi-like matrix to disk
#         super(ModelWL, self).save(file_path)
#
#     def load(self, name, exp_results_obj):
#         """
#         Loaded model will overwrite ARTM.topic_names and class_ids fields.
#         All class_ids weights will be set to 1.0, (if necessary, they have to be specified by hand. The method call will empty ARTM.score_tracker.
#         All regularizers and scores will be forgotten. It is strongly recommend to reset all important parameters of the ARTM model, used earlier.\n
#         :param name:
#         :return:
#         """
#         return self._exp.model_factory.create_model_with_phi_from_disk(self.get_full_path(name), exp_results_obj)



class BaseWriterLoader(WriterLoader):

    def __init__(self, location, extension, post_fix=''):
        """
        :param str location: path to directory where all objects shall be saved
        :param str extension: the file extension to use or the saved files
        :param str post_fix: an optional string to append to each resulting file name
        """
        self._loc = location
        self._extension = extension
        self._post = ''
        if post_fix:
            self._post = '-' + post_fix
        if not os.path.isdir(self._loc):
            os.mkdir(self._loc)
            print('Created \'{}\' dir to persist experimental results'.format(self._loc))
        self._saved = []

    @property
    def location(self):
        return self._loc

    @property
    def list(self):
        """
        This property dynamically holds the files' names, which are found in the saving location
        :return:
        :rtype: list
        """
        return [os.path.basename(file_path[:file_path.index(self._extension)]) for file_path in glob('{}/*{}'.format(self._loc, ''.join(filter(None, [self._post, self._extension]))))]

    @property
    def saved(self):
        return self._saved

    def save(self, name):
        raise NotImplementedError

    def load(self, name):
        raise NotImplementedError


class ExperimentWL(BaseWriterLoader):
    def __init__(self, experiment, location, extension, post_fix=''):
        super(ExperimentWL, self).__init__(location, extension, post_fix=post_fix)
        self._exp = experiment

    @property
    def split(self):
        return self._post.replace('-', '')

    def get_full_path(self, name):
        return os.path.join(self._loc, ''.join(str(x) for x in [name, self._post, self._extension]))

    def save(self, name):
        raise NotImplementedError
    def load(self, name, *args):
        raise NotImplementedError


#### CONCRETE CLASSES ######

class ResultsWL(ExperimentWL):
    """Class responsible for saving and loading experimental results (eg model training measurements/metrics)"""
    def __init__(self, experiment, split_label=None):
        """
        :param patm.modeling.experiment.Experiment experiment:
        :param str split_label: if not given then the assumptions is that there is no need to 'label' saved artifacts (ie to distinguish 'train' split from 'test' split)
        """
        super(ResultsWL, self).__init__(experiment, os.path.join(experiment.current_root_dir, RESULTS_DIR_NAME), '.json',post_fix=split_label)

    def save(self, name):
        results_file_path = self.get_full_path(name)
        if not str(results_file_path).endswith('json'):
            raise RuntimeError("Transformed '{}' to '{}' but still experimental results file name does not end with extension 'json'. self._post = '{}', self._extension = '{}'".format(name, results_file_path, self._post, self._extension))
        results = ExperimentalResults.create_from_experiment(self._exp)
        results.save_as_json(results_file_path, human_redable=True)
        self._saved.append(results_file_path)

    def load(self, name, *args):
        return ExperimentalResults.create_from_json_file(self.get_full_path(name))


class ModelWL(ExperimentWL):
    """Class responsible for saving and loading topic models"""
    def __init__(self, experiment, split_label=None):
        """
        :param patm.modeling.experiment.Experiment experiment:
        :param str split_label: if not given then the assumptions is that there is no need to 'label' saved artifacts (ie to distinguish 'train' split from 'test' split)
        """
        self._phi_matrix_label = 'p_wt'
        super(ModelWL, self).__init__(experiment, os.path.join(experiment.current_root_dir, MODELS_DIR_NAME), '.phi', post_fix=split_label)

    def save(self, name):
        self._exp.topic_model.artm_model.save(self.get_full_path(name), model_name=self._phi_matrix_label)  # saves one Phi-like matrix to disk
        self._saved.append(self.get_full_path(name))

    def load(self, name, *args):
        """
        Loaded model will overwrite ARTM.topic_names and class_ids fields.
        All class_ids weights will be set to 1.0, (if necessary, they have to be specified by hand. The method call will empty ARTM.score_tracker.
        All regularizers and scores will be forgotten. It is strongly recommend to reset all important parameters of the ARTM model, used earlier.\n
        :param name:
        :return:
        """
        return self._exp.model_factory.create_model_with_phi_from_disk(self.get_full_path(name), args[0])
