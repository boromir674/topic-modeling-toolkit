import abc

from patm.modeling.parameters import trajectory_builder

import artm


class ArtmRegularizerWrapper(object):
    __metaclass__ = abc.ABCMeta
    subclasses = {}
    labeling_ordering = ('tau', 'alpha_iter')
    _traj_type2traj_def_creator = {'alpha_iter': lambda x: '0_' + x[1],
                              'tau': lambda x: '{}_{}'.format(x[0], x[1])}

    def __new__(cls, *args, **kwargs):
        x = super(ArtmRegularizerWrapper, cls).__new__(cls)
        x._type = args[0]
        return x

    def __init__(self, reg_type, parameters_dict, verbose=False):
        self._regularizer = None
        self._alpha_iter_scalar = None
        self._trajectory_lambdas = {}
        self._traj_def = {}
        self._reg_constr_params = {}
        self._params_for_labeling = {}

        self._name = parameters_dict.pop('name', 'no-name')

        self._start = int(parameters_dict.pop('start', 0))
        for k, v in parameters_dict.items():
            if type(v) == list:  # one of topic_names or class_ids
                self._reg_constr_params[k] = v
            else:
                try:
                    vf = float(v)
                    if k == 'alpha_iter':
                        self._alpha_iter_scalar = vf  # case: alpha_iter = a constant scalar which will be used for each of the 'nb_doument_passes' iterations
                    else:
                        self._reg_constr_params[k] = vf # case: parameter_name == 'tau'
                    self._params_for_labeling[k] = vf
                except ValueError:
                    self._traj_def[k] = self._traj_type2traj_def_creator[k]([self._start, v])  # case: parameter_value is a trajectory definition without the 'start' setting (nb of initial iterations that regularizer stays inactive)
                    self._params_for_labeling[k] = self._traj_def[k]
        self._create_artm_regularizer(dict(self._reg_constr_params, **{'name': self._name}), verbose=verbose)

    def _create_artm_regularizer(self, parameters, verbose=False):
        self._regularizer = self._artm_constructor(**parameters)
        if verbose:
            print "Constructed '{}' reg, named '{}', with settings: {}".format(self._type, self._name, '{'+', '.join(map(lambda x: '{}={}'.format(x[0], x[1]), parameters.items()))+'}')

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
        print("DD", regularizer_type, args, kwargs)
        return cls.subclasses[regularizer_type](regularizer_type, *args, **kwargs)

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
        return trajectory_builder.begin_trajectory('tau')\
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
        return self._type


class SmoothSparseRegularizerWrapper(ArtmRegularizerWrapper):
    __metaclass__ = abc.ABCMeta

    def __init__(self, reg_type, name, params_dict, targeted_topics):
        super(SmoothSparseRegularizerWrapper, self).__init__(reg_type, dict(params_dict, **{'name': name, 'topic_names': targeted_topics}))


class SmoothSparsePhiRegularizerWrapper(SmoothSparseRegularizerWrapper):
    _artm_constructor = artm.SmoothSparsePhiRegularizer
    def __init__(self, reg_type, name, params_dict, topic_names, class_ids):
        super(SmoothSparsePhiRegularizerWrapper, self).__init__(reg_type, name, dict(params_dict, **{'class_ids': class_ids}), topic_names)

@ArtmRegularizerWrapper.register_subclass('sparse-phi')
class SparsePhiRegularizerWrapper(SmoothSparsePhiRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names, class_ids):
        super(SparsePhiRegularizerWrapper, self).__init__('sparse-phi', name, params_dict, topic_names, class_ids)

@ArtmRegularizerWrapper.register_subclass('smooth-phi')
class SmoothPhiRegularizerWrapper(SmoothSparsePhiRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names, class_ids):
        super(SmoothPhiRegularizerWrapper, self).__init__('smooth-phi', name, params_dict, topic_names, class_ids)


class SmoothSparseThetaRegularizerWrapper(SmoothSparseRegularizerWrapper):
    _artm_constructor = artm.SmoothSparseThetaRegularizer
    def __init__(self, reg_type, name, params_dict, topic_names):
        super(SmoothSparseThetaRegularizerWrapper, self).__init__(reg_type, name, params_dict, topic_names)


@ArtmRegularizerWrapper.register_subclass('sparse-theta')
class SparseThetaRegularizerWrapper(SmoothSparseThetaRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names):
        super(SparseThetaRegularizerWrapper, self).__init__('sparse-theta', name, params_dict, topic_names)

@ArtmRegularizerWrapper.register_subclass('smooth-theta')
class SmoothThetaRegularizerWrapper(SmoothSparseThetaRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names):
        super(SmoothThetaRegularizerWrapper, self).__init__('smooth-theta', name, params_dict, topic_names)


class DocumentClassificationRegularizerWrapper(ArtmRegularizerWrapper):
    _artm_constructor = artm.LabelRegularizationPhiRegularizer
    def __init__(self, name, params_dict, topic_names, class_ids):
        """

        :param str name:
        :param dict params_dict: Can contain keys: 'tau', 'gamma', 'dictionary'
        :param list of str topic_names: list of names of topics to regularize, will regularize all topics if not specified.
            Should correspond to the domain topics
        :param list of str class_ids: class_ids to regularize, will regularize all classes if not specified
        """
        super(DocumentClassificationRegularizerWrapper, self).__init__(
            'document-classification', dict(params_dict, **{'name': name, 'topic_names': topic_names, 'class_ids': class_ids}))
