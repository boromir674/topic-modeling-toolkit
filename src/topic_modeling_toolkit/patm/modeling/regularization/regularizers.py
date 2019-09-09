import abc
from warnings import warn

import artm

from .trajectory import TrajectoryBuilder


import logging
logger = logging.getLogger(__name__)

class ArtmRegularizerWrapper(object):
    __metaclass__ = abc.ABCMeta
    subclasses = {}
    labeling_ordering = ('tau', 'alpha_iter')
    _traj_type2traj_def_creator = {'alpha_iter': lambda x: '0_' + x[1],
                              'tau': lambda x: '{}_{}'.format(x[0], x[1])}

    def __init__(self, parameters_dict, verbose=False):
        self.trajectory_builder = TrajectoryBuilder()
        self._regularizer = None
        self._alpha_iter_scalar = None
        self._trajectory_lambdas = {}
        self._traj_def = {}
        self._reg_constr_params = {}
        self._params_for_labeling = {}
        self._start = None
        self._name = parameters_dict.pop('name', 'no-name')
        self._long_type = parameters_dict.pop('long-type', 'type-not-found')

        if 'start' in parameters_dict:
            self._start = parameters_dict['start']
        elif type(parameters_dict['tau']) == str:
            try:
                _ = float(parameters_dict['tau'])
            except ValueError:
                self._start = parameters_dict['tau'].split('_')[0]
                parameters_dict['tau'] = '_'.join(parameters_dict['tau'].split('_')[1:])

        for k, v in parameters_dict.items():
            # in case it is none then parameter it is handled by the default behaviour of artm
            if v is None or type(v) == list or type(v) == artm.dictionary.Dictionary or k == 'class_ids':  # one of topic_names or class_ids or dictionary
                self._reg_constr_params[k] = v
            else:
                try:  # by this point v should be a string, if exception occurs is shoud be only if a trajectory is defined (eg 'linear_-2_-10')
                    vf = float(v)
                    if k == 'alpha_iter':
                        self._alpha_iter_scalar = vf  # case: alpha_iter = a constant scalar which will be used for each of the 'nb_doument_passes' iterations
                    else:
                        self._reg_constr_params[k] = vf # case: parameter_name == 'tau'
                    self._params_for_labeling[k] = vf
                except ValueError:
                    if self._start is None:
                        print("INFO Defaulting to activating the regularizer from the 1st iteration")
                        self._start = 0
                    self._traj_def[k] = self._traj_type2traj_def_creator[k]([self._start, v])  # case: parameter_value is a trajectory definition without the 'start' setting (nb of initial iterations that regularizer stays inactive)
                    self._params_for_labeling[k] = self._traj_def[k]
        self._create_artm_regularizer(dict(self._reg_constr_params, **{'name': self._name}))

    def _create_artm_regularizer(self, parameters):
        self._regularizer = self._artm_constructor(**parameters)
        logger.info("Built '{}'/'{}' reg, named '{}', with settings: {}".format(self.type, self._long_type, self._name, '{'+', '.join(map(lambda x: '{}={}'.format(x[0], x[1]), parameters.items()))+'}'))

    @classmethod
    def register_subclass(cls, regularizer_type):
        def decorator(subclass):
            cls.subclasses[regularizer_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, regularizer_type, *args, **kwargs):
        if regularizer_type not in cls.subclasses:
            raise ValueError("Bad regularizer type '{}'".format(regularizer_type))
        return cls.subclasses[regularizer_type](*args, **kwargs)

    @property
    def label(self):
        return '{}|{}'.format(self._name, '|'.join(map(lambda x: '{}:{}'.format(x[0][0], x[1]), self._get_labeling_data())))

    def _get_labeling_data(self):
        return sorted(self._params_for_labeling.items(), key=lambda x: x[0])

    def get_tau_trajectory(self, collection_passes):
        if 'tau' in self._traj_def:
            return self._create_trajectory('tau', collection_passes)
        return None

    def set_alpha_iters_trajectory(self, nb_document_passes):
        if 'alpha_iter' in self._traj_def:
            self._regularizer.alpha_iter = list(self._create_trajectory('alpha_iter', nb_document_passes))
        elif self._alpha_iter_scalar:
            self._regularizer.alpha_iter = [self._alpha_iter_scalar] * nb_document_passes

    def _create_trajectory(self, name, length):
        _ = self._traj_def[name].split('_')
        return self.trajectory_builder.begin_trajectory('tau')\
            .deactivate(int(_[0]))\
            .interpolate_to(length - int(_[0]), float(_[3]), interpolation=_[1], start=float(_[2]))\
            .create()

    @property
    def static_parameters(self):
        return self._reg_constr_params

    @property
    def artm_regularizer(self):
        return self._regularizer
    def __str__(self):
        return self.name
    @property
    def name(self):
        return self._name

    @property
    def type(self):
        for k, v in ArtmRegularizerWrapper.subclasses.items():
            if type(self) == v:
                return k
    @property
    def long_type(self):
        return self._long_type

class SmoothSparseRegularizerWrapper(ArtmRegularizerWrapper):
    __metaclass__ = abc.ABCMeta

    def __init__(self, params_dict, targeted_topics):
        """
        :param params_dict:
        :param targeted_topics:
        """
        if len(targeted_topics) == 0:
            logger.warning("Did not specify topics to target with the '{}' {} regularizer. By default the Smooth regularizer will target all topics. "
                           "This is recommended only if all your regularizers target all topics (no notion of background-domain separation). "
                           "If you are modeling an LDA (plsa_formula + smoothing regularization over all topics), ignore this warning.".format(params_dict['name'], type(self).__name__))
            targeted_topics = None
        super(SmoothSparseRegularizerWrapper, self).__init__(dict(params_dict, **{'topic_names': targeted_topics}))


class SmoothSparsePhiRegularizerWrapper(SmoothSparseRegularizerWrapper):
    _artm_constructor = artm.SmoothSparsePhiRegularizer
    def __init__(self, params_dict, topic_names, class_ids):
        super(SmoothSparsePhiRegularizerWrapper, self).__init__(dict(params_dict, **{'class_ids': class_ids}), topic_names)

@ArtmRegularizerWrapper.register_subclass('sparse-phi')
class SparsePhiRegularizerWrapper(SmoothSparsePhiRegularizerWrapper):
    def __init__(self, params_dict, topic_names, class_ids):
        super(SparsePhiRegularizerWrapper, self).__init__(params_dict, topic_names, class_ids)

@ArtmRegularizerWrapper.register_subclass('smooth-phi')
class SmoothPhiRegularizerWrapper(SmoothSparsePhiRegularizerWrapper):
    def __init__(self, params_dict, topic_names, class_ids):
        super(SmoothPhiRegularizerWrapper, self).__init__(params_dict, topic_names, class_ids)


class SmoothSparseThetaRegularizerWrapper(SmoothSparseRegularizerWrapper):
    _artm_constructor = artm.SmoothSparseThetaRegularizer
    def __init__(self, params_dict, topic_names):
        super(SmoothSparseThetaRegularizerWrapper, self).__init__(params_dict, topic_names)


@ArtmRegularizerWrapper.register_subclass('sparse-theta')
class SparseThetaRegularizerWrapper(SmoothSparseThetaRegularizerWrapper):
    def __init__(self, params_dict, topic_names):
        super(SparseThetaRegularizerWrapper, self).__init__(params_dict, topic_names)

@ArtmRegularizerWrapper.register_subclass('smooth-theta')
class SmoothThetaRegularizerWrapper(SmoothSparseThetaRegularizerWrapper):
    def __init__(self, params_dict, topic_names):
        super(SmoothThetaRegularizerWrapper, self).__init__(params_dict, topic_names)

@ArtmRegularizerWrapper.register_subclass('label-regularization-phi')  # can be used to expand the probability space to DxWxTxC eg author-topic model
class DocumentClassificationRegularizerWrapper(ArtmRegularizerWrapper):
    _artm_constructor = artm.LabelRegularizationPhiRegularizer
    def __init__(self, params_dict, topic_names, dictionary=None, class_ids=None):
        """
        :param str name:
        :param dict params_dict: Can contain keys: 'tau', 'gamma', 'dictionary'
        :param list of str topic_names: list of names of topics to regularize, will regularize all topics if not specified.
            Should correspond to the domain topics
        :param list of str class_ids: class_ids to regularize, will regularize all classes if not specified
        :param dictionary:
        :param class_ids:
        """
        if len(topic_names) == 0: # T.O.D.O below: the warning should fire if smooth is active because then there must be defined
            # non overlapping sets of 'domain' and 'background' topics
            warn("Set DocumentClassificationRegularizer to target all topics. This is valid only if you do use 'background topics'.")
            topic_names = None
        super(DocumentClassificationRegularizerWrapper, self).__init__(dict(params_dict, **{'topic_names': topic_names,
                                                                                            'dictionary': dictionary,
                                                                                            'class_ids': class_ids}))
@ArtmRegularizerWrapper.register_subclass('decorrelate-phi')
class PhiDecorrelator(ArtmRegularizerWrapper):
    _artm_constructor = artm.DecorrelatorPhiRegularizer
    def __init__(self, params_dict, topic_names, class_ids=None):
        super(PhiDecorrelator, self).__init__(dict(params_dict, **{'topic_names': topic_names,
                                                                   'class_ids': class_ids}))

@ArtmRegularizerWrapper.register_subclass('improve-coherence')
class ImproveCoherence(ArtmRegularizerWrapper):
    _artm_constructor = artm.ImproveCoherencePhiRegularizer # name=None, tau=1.0, class_ids=None, topic_names=None, dictionary=None, config=None)
    def __init__(self, params_dict, topic_names, dictionary, class_ids=None):
        super(ImproveCoherence, self).__init__(dict(params_dict, **{'dictionary': dictionary,
                                                                    'topic_names': topic_names,
                                                                    'class_ids': class_ids}))
