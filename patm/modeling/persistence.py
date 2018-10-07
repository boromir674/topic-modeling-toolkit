import os
import abc
import glob
import json

from patm.utils import load_results


class WriterLoader(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def location(self):
        raise NotImplemented

    @abc.abstractproperty
    def list(self):
        raise NotImplemented

    @abc.abstractproperty
    def saved(self):
        raise NotImplemented

    @abc.abstractmethod
    def save(self, name):
        raise NotImplemented

    @abc.abstractmethod
    def load(self, name):
        raise NotImplemented


class BaseWriterLoader(WriterLoader):

    def __init__(self, location, post_fix, extension):
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
        return [os.path.basename(file_path) for file_path in glob.glob('{}/*{}{}'.format(self._loc, self._post, self._extension))]

    @property
    def saved(self):
        return self._saved

    def save(self, name):
        raise NotImplementedError

    def load(self, name):
        raise NotImplementedError


class ExperimentWL(BaseWriterLoader):
    def __init__(self, experiment, location, post_fix, extension):
        super(ExperimentWL, self).__init__(location, post_fix, extension)
        self._exp = experiment

    @property
    def split(self):
        return self._post

    def get_full_path(self, name):
        if self._post:
            return os.path.join(self._loc, '{}-{}{}'.format(name, self._post, self._extension))
        return os.path.join(self._loc, '{}{}'.format(name, self._extension))

    def save(self, name):
        raise NotImplementedError
    def load(self, name, *args):
        raise NotImplementedError


class ResultsWL(ExperimentWL):
    def __init__(self, experiment, split_label):
        self._my_root = 'results'
        self._ext = '.json'
        super(ResultsWL, self).__init__(experiment, os.path.join(experiment.current_root_dir, self._my_root), split_label, self._ext)

    def save(self, name):
        results = _stringify_trackable_dicts(self._exp.get_results())
        with open(self.get_full_path(name), 'w') as f:
            json.dump(results, f)
        self._saved.append(self.get_full_path(name))

    def load(self, name, *args):
        return load_results(self.get_full_path(name))


class ModelWL(ExperimentWL):
    def __init__(self, experiment, split_label):
        self._my_root = 'models'
        self._extension = '.phi'
        self._phi_matrix_label = 'p_wt'
        super(ModelWL, self).__init__(experiment, os.path.join(experiment.current_root_dir, self._my_root), split_label, self._extension)

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


def _stringify_trackable_dicts(results):
    tr = results['trackables']
    for k, v in tr.items():
        if type(v) == list:
            tr[k] = map(lambda x: str(x), tr[k])
        elif type(v) == dict:
            for in_k, in_v in v.items():
                if type(in_v) == list:
                    v[in_k] = map(lambda x: str(x), v[in_k])
    return results
