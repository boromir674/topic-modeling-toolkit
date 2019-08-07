import os
import abc
from glob import glob
from patm.definitions import MODELS_DIR_NAME, RESULTS_DIR_NAME

from results import ExperimentalResults


class WriterLoader(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def save(self, name):
        raise NotImplemented

    @abc.abstractmethod
    def load(self, name):
        raise NotImplemented


class BaseWriterLoader(WriterLoader):

    def __init__(self, location, extension, post_fix=''):
        """
        :param str location: path to directory where all objects shall be saved
        :param str extension: the file extension to use or the saved files
        :param str post_fix: an optional string to append to each resulting file name
        """
        self._loc = location
        self._post = post_fix
        self._extension = extension
        if not os.path.isdir(self._loc):
            os.mkdir(self._loc)
            print 'Created \'{}\' dir to persist experimental results'.format(self._loc)
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
        return self._post

    def get_full_path(self, name):
        return os.path.join(self._loc, '{}{}{}'.format(name, (lambda x: '-'+x if x else '')(self._post), self._extension))
        # if self._post:
        #     return os.path.join(self._loc, '{}-{}{}'.format(name, self._post, self._extension))
        # return os.path.join(self._loc, '{}{}'.format(name, self._extension))

    def save(self, name):
        raise NotImplementedError
    def load(self, name, *args):
        raise NotImplementedError


##### CONCRETE CLASSES ######

class ResultsWL(ExperimentWL):
    """Class responsible for saving and loading experimental results (eg model training measurements/metrics)"""
    def __init__(self, experiment, split_label=None):
        """
        :param patm.modeling.experiment.Experiment experiment:
        :param str split_label: if not given then the assumptions is that there is no need to 'label' saved artifacts (ie to distinguish 'train' split from 'test' split)
        """
        self._ext = '.json'
        super(ResultsWL, self).__init__(experiment, os.path.join(experiment.current_root_dir, RESULTS_DIR_NAME), split_label, self._ext)

    def save(self, name):
        results_file_path = self.get_full_path(name)
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
        self._extension = '.phi'
        self._phi_matrix_label = 'p_wt'
        super(ModelWL, self).__init__(experiment, os.path.join(experiment.current_root_dir, MODELS_DIR_NAME), split_label, self._extension)

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
