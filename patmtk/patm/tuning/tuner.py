import os
import re
import sys
import warnings
import pprint
from tqdm import tqdm
from collections import OrderedDict, Counter

from building import RegularizersActivationDefinitionBuilder

from patm import trainer_factory, Experiment
from patm.modeling.parameters import ParameterGrid
from patm.utils import get_standard_evaluation_definitions
from patm.definitions import COLLECTIONS_DIR, DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME


class Tuner(object):
    """
    This class is able to perform a search over the parameter space (grid-search supported).

    Supports Static:\n
    - nb_topics                          eg: 80\n
    - collection_passes                  eg: 100\n
    - document_passes                    eg: 5\n
    - sparse_phi_reg_coef_trajectory     eg: {'deactivation_period_pct': 0.1, 'start': -0.2, 'end' = -8}\n
    - sparse_theta_reg_coef_trajectory   eg: {'deactivation_period_pct': 0.2, 'start': -0.1, 'end' = -4}\n\n
    - decorrelate_topics_reg_coef        eg: 1e+5\n

    and Explorable\n

    - nb_topics                          eg: [20, 40, 60, 80, 100]\n
    - collection_passes                  eg: [100]\n
    - document_passes                    eg: [1, 5, 10, 15]\n
    - sparse_phi_reg_coef_trajectory     eg: {'deactivation_period_pct': [0.1, 0.2], 'start': [-0.1, -0.2, -0.3, -0.4], 'end' = [-1, -2, -3, -4]}\n
    - sparse_theta_reg_coef_trajectory   eg: {'deactivation_period_pct': [0.2], 'start': [-0.2, -0.3, -0.4], 'end' = [-1, -3, -5]}\n
    - decorrelate_topics_reg_coef        eg: 1e+5\n
    """

    def __init__(self, collection, evaluation_definitions=None):
        """

        :param str collection: the name of the 'dataset'/collection to target the tuning process on
        :param str train_config: full path to 'the' train.cfg file used here only for initializing tracking evaluation scoring capabilities. Namely the necessary BaseScore objects of ARTM lib and the custom ArtmEvaluator objects are initialized
        :param list_of_tuples static_parameters: definition of parameters that will remain constant while exploring the parameter space. The order of this affects the order in which vectors of the parameter space are generated
            ie: [('collection_passes', 100), ('nb_topics', 20), ('document_passes', 5))].
        :param list of tuples explorable_parameters_names: definition of parameter "ranges" that will be explored; the order of list affects the order in which vectors of the parameter space are generated; ie
        :param bool enable_ideology_labels: Enable using the 'vowpal_wabbit' format to exploit extra modalities modalities. A modalitity is a disctreet space of values (tokens, class_labels, ..)

        """
        self._required_parameters = ('nb_topics', 'document_passes', 'collection_passes')
        self._dir = os.path.join(COLLECTIONS_DIR, collection)
        if evaluation_definitions:
            self._score_defs = evaluation_definitions
        else:
            self._score_defs = get_standard_evaluation_definitions()
        self._label_groups = None
        self._cur_label = None
        self._required_labels = []
        self._active_regs = {}
        self._reg_specs = {}
        self._max_digits_version = 3
        self._allowed_labeling_params = ['collection_passes', 'nb_topics', 'document_passes', 'background_topics_pct', 'ideology_class_weight'] + \
                                        reduce(lambda i,j: i+j, map(lambda y: map(lambda x: y+'.'+x, ('deactivate', 'kind', 'start', 'end')), ('sparse_phi', 'sparse_theta')))
        self._active_reg_def_builder = RegularizersActivationDefinitionBuilder(tuner=self)
        self.trainer = trainer_factory.create_trainer(collection, exploit_ideology_labels=True)  # forces to use (and create if not found) batches holding modality information in case it is needed
        self.experiment = Experiment(self._dir, self.trainer.cooc_dicts)
        self.trainer.register(self.experiment)  # when the model_trainer trains, the experiment object listens to changes
        self._parameter_grid_searcher = None
        self._default_regularizer_parameters = {'smooth-phi': {'tau': 1.0},
                                                'smooth-theta': {'tau': 1.0, 'alpha_iter': 1.0},
                                                'sparse-phi': {'tau': 'linear_-5_-15', 'start': 4},
                                                'sparse-theta': {'alpha_iter': 1, 'tau': 'linear_-3_-13', 'start': 4}}
    @staticmethod
    def _create_active_regs_with_default_names(reg_types_list):
        """
        :param list reg_types_list:
        :return: keys are the regularizer-types (ie 'smooth-theta', 'sparse-phi') and values automatically genereted abbreviations of the regularizer-types to serve as unique names
        :rtype: OrderedDict
        """
        return OrderedDict(map(lambda y: ('-'.join(y), ''.join(map(lambda z: z[0] if len(y) > 2 else z[:2], y))), map(lambda x: x.split('-'), sorted(reg_types_list))))

    @property
    def constants(self):
        return sorted(self._static_params_hash.keys())
    @property
    def explorables(self):
        return sorted(self._expl_params_hash.keys())

    @property
    def allowed_labeling_params(self):
        return self._allowed_labeling_params

    @property
    def activate_regularizers(self):
        return self._active_reg_def_builder.activate

    @property
    def active_regularizers(self):
        """
        :rtype: OrderedDict
        """
        return self._active_regs

    @active_regularizers.setter
    def active_regularizers(self, active_regularizers):
        """
        :param dist or dict active_regularizers: in case of list it should be a list of regularizer-types ie ['smooth-phi', 'sparse-phi', 'smooth-theta']\n
            in case of dict it should be a dict with keys regularizer-types ie ('smooth-phi', 'sparse-phi', 'smooth-theta') and values unique corresponding names
        """
        if type(active_regularizers) == list:
            self._active_regs = Tuner._create_active_regs_with_default_names(active_regularizers)
        elif isinstance(active_regularizers, dict):
            self._active_regs = active_regularizers
        else:
            raise TypeError

    @property
    def static_regularization_specs(self):
        return self._reg_specs

    @static_regularization_specs.setter
    def static_regularization_specs(self, regularizers_specs):
        self._reg_specs = regularizers_specs

    def _set_verbosity_level(self, input_verbose):
        try:
            self._vb = int(input_verbose)
            if self._vb < 0:
                self._vb = 0
            elif 4 < self._vb:
                self._vb = 4
        except ValueError:
            self._vb = 3

    def tune(self, parameters_mixture, prefix_label='', append_explorables='all', append_static=True, static_regularizers_specs=None, force_overwrite=False, verbose=3):
        """
        :param patm.tuning.building.TunerDefinition parameters_mixture:
        :param str prefix_label: an optional alphanumeric that serves as a coonstant prefix used for the naming files (models, results) saved on disk
        :param list or str append_explorables: if list is given will use these exact parameters' names as part of unique names used for files saved on disk. If 'all' is given then all possible parameter names will be used
        :param bool append_static: indicates whether to use the values of the parameters remaining constant during space exploration, as part of the unique names used for files saved on disk
        :param static_regularizers_specs:
        :param force_overwrite:
        :param verbose:
        :return:
        """
        self._set_verbosity_level(verbose)
        self._initialize0(parameters_mixture)
        self._initialize1(prefix_label=prefix_label, append_explorables=append_explorables, append_static=append_static,
                          static_regularizers_specs=static_regularizers_specs, overwrite=force_overwrite)
        if self._vb:
            print 'Tuning..'
            generator = tqdm(self._parameter_grid_searcher, total=len(self._parameter_grid_searcher), unit='model')
        else:
            generator = iter(self._parameter_grid_searcher)
        if 1 < self._vb:
            print 'Taking {} samples for grid-search'.format(len(self._parameter_grid_searcher))

        self._label_groups = Counter()
        for i, self.parameter_vector in enumerate(generator):
            self._cur_label = self._build_label(self.parameter_vector)
            assert self._cur_label == self._required_labels[i]
            if 4 == vb:
                tqdm.write(pprint.pformat(self.static_regularization_specs))
            tm, specs = self._create_model_n_specs()
            self.experiment.init_empty_trackables(tm)
            self.trainer.train(tm, specs)
            self.experiment.save_experiment(save_phi=True)
            if 2 < self._vb:
                tqdm.write(self._cur_label)
            del tm

    def _initialize0(self, tuner_definition):
        """
        :param patm.tuning.building.TunerDefinition tuner_definition:
        """
        if self._vb:
            print 'Initializing Tuner..'
        self._static_params_hash = tuner_definition.static_parameters
        self._expl_params_hash = tuner_definition.explorable_parameters
        self._check_parameters()
        self._tau_traj_to_build = tuner_definition.valid_trajectory_defs()
        self._parameter_grid_searcher = ParameterGrid(tuner_definition.parameter_spans)
        if 1 < self._vb :
            print 'Constants: [{}]\nExplorables: [{}]'.format(', '.join(self.constants), ', '.join(self.explorables))
            # print 'Tau trajectories for:', self._tau_traj_to_build.keys()
            print 'Search space', tuner_definition.parameter_spans
            print 'Potentially producing {} parameter vectors'.format(len(self._parameter_grid_searcher))

    def _define_labeling_scheme(self, explorables, constants):
        """Call this method to define the values to use for labeling the artifacts of tuning from the mixture of explorable and constant parameters.
            This method also determines if versioning is needed; whether to append strings like v001, v002 to the labels because
            of naming collisions
        """
        assert all(map(lambda x: type(x) == list or x == 'all', [constants, explorables]))
        explorables_labels = self._get_labels_extractor('explorables')(explorables)
        constants_labels = self._get_labels_extractor('constants')(constants)
        self._versioning_needed = not explorables_labels == self.explorables
        self._labeling_params = explorables_labels + constants_labels

    def _get_labels_extractor(self, parameters_type):
        extractors_hash = {list: lambda x: [_ for _ in x if _ in self._allowed_labeling_params],
                           str: lambda x: [_ for _ in getattr(self, parameters_type) if _ in self._allowed_labeling_params]}
        return lambda y: sorted(extractors_hash[type(y)](y))


    def _initialize1(self, prefix_label='', append_explorables='all', append_static=None, static_regularizers_specs=None, overwrite=False):
        """
        :param str prefix_label: an optional alphanumeric that serves as a coonstant prefix used for the naming files (models, results) saved on disk
        :param bool enable_ideology_labels: Enable using the 'vowpal_wabbit' format to exploit extra modalities modalities. A modalitity is a disctreet space of values (tokens, class_labels, ..)
        :param list or str append_explorables: if list is given will use these exact parameters' names as part of unique names used for files saved on disk. If 'all' is given then all possible parameter names will be used
        :param bool append_static: indicates whether to use the values of the parameters remaining constant during space exploration, as part of the unique names used for files saved on disk
        :param static_regularizers_specs:
        :return:
        """
        self._prefix = prefix_label
        self._experiments_saved = []
        self._label_groups = Counter()
        self._overwrite = overwrite
        print append_explorables, (lambda y: [] if y is None else y)(append_explorables)
        print append_static, (lambda y: [] if y is None else y)(append_static)
        self._define_labeling_scheme((lambda y: [] if y is None else y)(append_explorables), (lambda y: [] if y is None else y)(append_static))

        if 1 < self._vb:
            print 'Automatically labeling files using parameter values: {[]}'.format(', '.join(self._labeling_params))

        if static_regularizers_specs:
            # for which ever of the activated regularizers there is a missing setting, then use a default value
            self.static_regularization_specs = self._create_reg_specs(static_regularizers_specs, self.active_regularizers)
        else:
            self.static_regularization_specs = self._create_reg_specs(self._default_regularizer_parameters, self.active_regularizers)

        self._build_required_labels()  # depends on self.parameter_grid_searcher and self._labeling_params

        if not self._overwrite:
            _ = self._get_overlapping_indices()
            result_inds, model_inds = IndicesList(_[0], 'train results'), IndicesList(_[1], 'phi matrix')
            common = result_inds + model_inds  # finds intersection of indices
            only_res = result_inds - common
            only_mods = model_inds - common
            self._parameter_grid_searcher.ommited_indices = common.indices
            self._required_labels = [x for i, x in enumerate(self._required_labels) if i not in common.indices]
            if 2 < self._vb:
                for obj in (_ for _ in (common, only_mods, only_res) if _):
                    print obj.msg(self._required_labels)
            if 3 < self._vb:
                print 'Required labels:', '[{}]'.format(', '.join(self._required_labels))
            if 1 < self._vb:
                print 'Ommiting creating a total of {} files'.format(len(common)+len(only_res)+len(only_mods))

    def _create_reg_specs(self, reg_settings, active_regularizers):
        """Call this method to use default values when missing, according to given activated regularizers"""
        return {k: dict(map(lambda x: (x[0], reg_settings[k].get(x[0],
            self._default_regularizer_parameters[k][x[0]])), self._default_regularizer_parameters[k].items())) for k in active_regularizers}

    def _get_overlapping_indices(self):
        _ = map(lambda x: self._expr((x[1] in self.experiment.train_results_handler.list, x[1] in self.experiment.phi_matrix_handler.list), x[0]), enumerate(self._required_labels))
        return map(lambda x: filter(lambda y: y is not None, x), map(list, zip(*_)))

    def _expr(self, tt, el):
        return map(lambda x: el if x else None, tt)

    def _build_required_labels(self):
        self._label_groups = Counter()
        self._required_labels = map(lambda x: self._build_label(x), self._parameter_grid_searcher)
        
    def _build_label(self, parameter_vector):
        cached = map(lambda x: str(self._extract(parameter_vector, x)), self._labeling_params)
        label = '_'.join(filter(None, [self._prefix] + cached))
        self._label_groups[label] += 1
        if self._versioning_needed:
            label = '_'.join(filter(None, [self._prefix] + ['v{}'.format(self._iter_prepend(self._label_groups[label]))] + cached))
        return label

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits == self._max_digits_version:
            return str(int_num)
        return '{}{}'.format((self._max_digits_version - nb_digits) * '0', int_num)

    def _create_model_n_specs(self):
        self.static_regularization_specs = self._replace_settings_with_supported_explorable(self._reg_specs)
        # for reg_type in (_ for _ in self.active_regularizers if _.startswith('sparse_')):
        #     self._reg_specs[reg_type] = self._build_definition(self._val('sparse_' + reg_type.split('-')[1]))
        tm = self.trainer.model_factory.construct_model(self._cur_label, self._val('nb_topics'),
                                                              self._val('collection_passes'),
                                                              self._val('document_passes'),
                                                              self._val('background_topics_pct'),
                                                              {DEFAULT_CLASS_NAME: self._val('default_class_weight'), IDEOLOGY_CLASS_NAME: self._val('ideology_class_weight')},
                                                              self._score_defs,
                                                              self._active_regs,
                                                              reg_settings=self.static_regularization_specs)
        tr_specs = self.trainer.model_factory.create_train_specs(self._val('collection_passes'))
        return tm, tr_specs

    def _replace_settings_with_supported_explorable(self, settings):
            # _ = map(lambda y: y(1), filter(None, map(lambda x: getattr(re.match('^(sparse_\w+)$', x), 'group', None), self._tau_traj_to_build)))
        # for reg_type in map(lambda x: x.replace('_', '-'), self._tau_traj_to_build):
        #     settings[reg_type] = dict(settings[reg_type], **self._get_tau_trajectory_definition_dict(reg_type.replace('-', '_')))
        # return dict(map(lambda x: (x[0], x[1]), map(lambda x: x.replace('_', '-'), self._tau_traj_to_build)))
        return dict(map(lambda x: (x[0], self._get_settings_dict(x[0], x[1])), settings.items()))

    def _get_settings_dict(self, reg_type, settings_dict):
        if reg_type.replace('-', '_') in self._tau_traj_to_build:
            return dict(settings_dict, **self._get_tau_trajectory_definition_dict(reg_type.replace('-', '_')))
        return settings_dict

    def _get_tau_trajectory_definition_dict(self, tau_traj_type):
        """
        Call this method to query the current parameter vector for a specific tau trajectory\n
        :param str tau_traj_type: underscore splittable
        :rtype: dict
        """
        _ = self._val(tau_traj_type)
        return {'tau': '_'.join(map(lambda x: str(x) ,(_['kind'], _['start'], _['end']))), 'start': _['deactivate']}

    def _val(self, parameter_name):
        return self._extract(self.parameter_vector, parameter_name)

    def _extract(self, parameter_vector, parameter_name):
        """Parameter value extractor from currently generated parameter vector and from static parameters"""
        if parameter_name in self._static_params_hash:
            return self._static_params_hash[parameter_name]
        if parameter_name in self._expl_params_hash:  # and not parameter_name.startswith('sparse_'): #in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory'):
            return parameter_vector[self._expl_params_hash.keys().index(parameter_name)]
        if len(parameter_name.split('_')) == 2 and parameter_name.startswith('sparse_') and len(parameter_name.split('.')) == 1:  # ie if parameter_name == 'sparse_phi' or 'sparse_theta'
            return {'deactivate': self._extract(parameter_vector, 'sparse_' + parameter_name.split('_')[1] + '.deactivate'),
                    'kind': self._extract(parameter_vector, 'sparse_' + parameter_name.split('_')[1] + '.kind'),
                    'start': self._extract(parameter_vector, 'sparse_' + parameter_name.split('_')[1] + '.start'),
                    'end': self._extract(parameter_vector, 'sparse_' + parameter_name.split('_')[1] + '.end')}
        else:
            if parameter_name == 'default_class_weight':
                return 1.0
            if parameter_name == 'ideology_class_weight':
                return 0.0

    def _check_parameters(self):
        for i in self._static_params_hash.keys():
            if i in self._expl_params_hash:
                raise ParameterFoundInStaticAndExplorablesException("Parameter '{}' defined both as static and explorable".format(i))
        missing_required_parameters = filter(None, map(lambda x: None if x in self.constants + self.explorables else x, self._required_parameters))
        if missing_required_parameters:
            raise MissingRequiredParametersException("[{}] were not found in [{}]".format(', '.join(missing_required_parameters), ', '.join(self.constants + self.explorables)))


class IndicesList(object):
    def __init__(self, indices_list, string_label):
        self._inds = indices_list
        self._label = string_label
        assert len(indices_list) == len(frozenset(indices_list))
    def __str__(self):
        return self._label
    def __len__(self):
        return len(self._inds)
    def __contains__(self, item):
        return item in self._inds
    def __iter__(self):
        for _ in self._inds:
            yield _
    def __add__(self, other):
        return IndicesList([_ for _ in self if _ in other], '{} and {}'.format(self, other))
    def __sub__(self, other):
        return IndicesList([_ for _ in self if _ not in other], str(self))
    @property
    def indices(self):
        return self._inds
    @property
    def labels(self):
        return lambda x: [x[_] for _ in self._inds]
    @property
    def msg(self):
        return lambda x: "Existing labels {} for '{}' files will overlap with the newly required ones by the tuner.".format(self.labels(x), self._label)


class MissingRequiredParametersException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

class ParameterFoundInStaticAndExplorablesException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


if __name__ == '__main__':
    tuner = Tuner('gav')
    from patm.tuning.building import tuner_definition_builder as tdb

    d2 = tdb.initialize().nb_topics([20]).collection_passes(100).document_passes(1).background_topics_pct(
        0.1).ideology_class_weight([5]). \
        sparse_phi().deactivate(10).kind(['quadratic']).start(-1).end([-10, -20]). \
        sparse_theta().deactivate(10).kind('linear').start([-3]).end(-10).build()

    tuner.activate_regularizers.smoothing.phi.theta.sparsing.phi.theta.done()
    # tuner.static_regularization_specs = {'smooth-phi': {'tau': 1.0},
    #                                      'smooth-theta': {'tau': 1.0},
    #                                      'sparse-theta': {'alpha_iter': 1}}
                                         # 'sparse-theta': {'alpha_iter': 'linear_1_4'}}
    tuner.tune(d2, prefix_label='tag1', append_explorables='all', append_static=True)
