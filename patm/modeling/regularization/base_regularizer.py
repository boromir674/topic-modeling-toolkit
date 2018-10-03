from patm.modeling.parameters.trajectory import trajectory_builder

import artm


class ArtmRegularizerWrapper(object):
    labeling_ordering = ('tau', 'alpha_iter')
    _traj_type2traj_def_creator = {'alpha_iter': lambda x: '0_' + x[1],
                              'tau': lambda x: '{}_{}'.format(x[0], x[1])}

    def _create_artm_regularizer(self, parameters, verbose=False):
        self._regularizer = self._artm_constructor_callback(**parameters)
        if verbose:
            print "Constructed '{}' reg, named '{}', with settings: {}".format(self._type, self._name, '{'+', '.join(map(lambda x: '{}={}'.format(x[0], x[1]), self._computed_params.items()))+'}')

    def __init__(self, reg_type, parameters_dict, artm_constructor):
        self._name = parameters_dict.pop('name', 'no-name')
        self._type = reg_type
        self._label = ''
        self._regularizer = None
        self._computed_params = {}
        self._trajectory_lambdas = {}
        self._alpha_iter_scalar = None
        self._traj_def = {}
        self._reg_constr_params = {}
        self._artm_constructor_callback = artm_constructor
        self._params_for_labeling = {}

        self._start = int(parameters_dict.pop('start', 0))
        for k, v in parameters_dict.items():
            self._parse(k, v)
        self._create_artm_regularizer(dict(self._reg_constr_params, **{'name': self._name}), verbose=True)

    def _parse(self, parameter_name, parameter_value):
        if type(parameter_value) == list: # one of topic_names or class_ids
            self._reg_constr_params[parameter_name] = parameter_value
        else:
            try:
                vf = float(parameter_value)
                if parameter_name == 'alpha_iter':
                    self._alpha_iter_scalar = vf  # case: alpha_iter = a constant scalar which will be used for each of the 'nb_doument_passes' iterations
                else:
                    self._reg_constr_params[parameter_name], self._params_for_labeling[parameter_name] = vf, vf  # case: parameter_name == 'tau'
            except ValueError:
                self._traj_def[parameter_name] = self._traj_type2traj_def_creator[parameter_name]([self._start, parameter_value])  # case: parameter_value is a trajectory definition without the 'start' setting (nb of initial iterations that regularizer stays inactive)

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
        return trajectory_builder.begin_trajectory('tau').deactivate(int(_[0])).interpolate_to(length - int(_[0]), float(_[3]), interpolation=_[1], start=float(_[2])).create()

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

    @property
    def label(self):
        # return '{}_{}'.format(self._name, '-'.join(filter(None, map(lambda x: self._stringify_tau_or_alpha_iter(x), self.labeling_ordering))))
        return '{}_{}'.format(self._name, '-'.join(map(lambda x: '{}'.format(x[1]), sorted(self._params_for_labeling.items()) + sorted(self._traj_def.items()))))


class SmoothSparsePhiRegularizerWrapper(ArtmRegularizerWrapper):
    _artm_constructor_callback = artm.SmoothSparsePhiRegularizer
    def __init__(self, reg_type, name, params_dict, topic_names, class_ids):
        super(SmoothSparsePhiRegularizerWrapper, self).__init__(reg_type, dict(params_dict, **{'name': name, 'topic_names': topic_names, 'class_ids': class_ids}), self._artm_constructor_callback)

class SparsePhiRegularizerWrapper(SmoothSparsePhiRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names, class_ids):
        super(SparsePhiRegularizerWrapper, self).__init__('sparse-phi', name, params_dict, topic_names, class_ids)

class SmoothPhiRegularizerWrapper(SmoothSparsePhiRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names, class_ids):
        super(SmoothPhiRegularizerWrapper, self).__init__('smooth-phi', name, params_dict, topic_names, class_ids)


class SmoothSparseThetaRegularizerWrapper(ArtmRegularizerWrapper):
    _artm_constructor_callback = artm.SmoothSparseThetaRegularizer
    def __init__(self, reg_type, name, params_dict, topic_names):
        super(SmoothSparseThetaRegularizerWrapper, self).__init__(reg_type, dict(params_dict, **{'name': name, 'topic_names': topic_names}), self._artm_constructor_callback )

class SparseThetaRegularizerWrapper(SmoothSparseThetaRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names):
        super(SparseThetaRegularizerWrapper, self).__init__('sparse-theta', name, params_dict, topic_names)

class SmoothThetaRegularizerWrapper(SmoothSparseThetaRegularizerWrapper):
    def __init__(self, name, params_dict, topic_names):
        super(SmoothThetaRegularizerWrapper, self).__init__('smooth-theta', name, params_dict, topic_names)
