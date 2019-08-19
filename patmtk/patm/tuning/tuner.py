import os
import re
import sys
import warnings
import pprint
from tqdm import tqdm
from collections import OrderedDict, Counter
from functools import reduce

from .building import RegularizersActivationDefinitionBuilder

from patm.modeling import Experiment
from patm.modeling.parameters import ParameterGrid
from patm.modeling import TrainerFactory
from patm.utils import get_standard_evaluation_definitions
from patm.definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME


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

    def __init__(self, collection_dir, evaluation_definitions=None, verbose=3):
        """
        :param str collection_dir: the path to a 'dataset'/collection to target the tuning process on
        :param str train_config: full path to 'the' test-train.cfg file used here only for initializing tracking evaluation scoring capabilities. Namely the necessary BaseScore objects of ARTM lib and the custom ArtmEvaluator objects are initialized
        :param list_of_tuples static_parameters: definition of parameters that will remain constant while exploring the parameter space. The order of this affects the order in which vectors of the parameter space are generated
            ie: [('collection_passes', 100), ('nb_topics', 20), ('document_passes', 5))].
        :param list of tuples explorable_parameters_names: definition of parameter "ranges" that will be explored; the order of list affects the order in which vectors of the parameter space are generated; ie
        :param bool enable_ideology_labels: Enable using the 'vowpal_wabbit' format to exploit extra modalities modalities. A modalitity is a disctreet space of values (tokens, class_labels, ..)

        """
        self._set_verbosity_level(verbose)
        self._required_parameters = ('nb_topics', 'document_passes', 'collection_passes')
        self._dir = collection_dir
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
                                        reduce(lambda i,j: i+j, [[y+'.'+x for x in ('deactivate', 'kind', 'start', 'end')] for y in ('sparse_phi', 'sparse_theta')])
        self._active_reg_def_builder = RegularizersActivationDefinitionBuilder(tuner=self)
        self.trainer = TrainerFactory().create_trainer(self._dir, exploit_ideology_labels=True)  # forces to use (and create if not found) batches holding modality information in case it is needed
        self.experiment = Experiment(self._dir)
        self.trainer.register(self.experiment)  # when the model_trainer trains, the experiment object listens to changes
        self._parameter_grid_searcher = None
        self._default_regularizer_parameters = {'smooth-phi': {'tau': 1.0},
                                                'smooth-theta': {'tau': 1.0, 'alpha_iter': 1.0},
                                                'sparse-phi': {'tau': 'linear_-5_-15', 'start': 4},
                                                'sparse-theta': {'alpha_iter': 1, 'tau': 'linear_-3_-13', 'start': 4},
                                                'label-regularization-phi': {'tau': 1.0},
                                                'label-regularization-phi-dom-def': {'tau': 1e5},
                                                'label-regularization-phi-dom-cls': {'tau': 1e5},
                                                'decorrelate-phi-def': {'tau': 10000},
                                                'decorrelate-phi-dom-def': {'tau': 10000},
                                                'decorrelate-phi-class': {'tau': 10000},
                                                'decorrelate-phi-domain': {'tau': 10000},
                                                'decorrelate-phi-background': {'tau': 10000},
                                                'improve-coherence': {'tau': 1.0}}
    @staticmethod
    def _create_active_regs_with_default_names(reg_types_list):
        """
        :param list reg_types_list:
        :return: keys are the regularizer-types (ie 'smooth-theta', 'sparse-phi') and values automatically genereted abbreviations of the regularizer-types to serve as unique names
        :rtype: OrderedDict
        """
        return OrderedDict([('-'.join(y), ''.join([z[0] if len(y) > 2 else z[:2] for z in y])) for y in [x.split('-') for x in sorted(reg_types_list)]])

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
        :param list or dict active_regularizers: in case of list it should be a list of regularizer-types ie ['smooth-phi', 'sparse-phi', 'smooth-theta']\n
            in case of dict it should be a dict having as keys regularizer-types ie ('smooth-phi', 'sparse-phi', 'smooth-theta') and as values the corresponding unique names
        """
        if type(active_regularizers) == list:
            self._active_regs = Tuner._create_active_regs_with_default_names(active_regularizers)
        elif isinstance(active_regularizers, dict):
            self._active_regs = active_regularizers
        else:
            raise TypeError
        if 2 < self._vb:
            print("Tuner: activated regs: [{}]".format(', '.join(sorted([_ for _ in self._active_regs.keys()]))))

    @property
    def static_regularization_specs(self):
        return self._reg_specs

    def _set_verbosity_level(self, input_verbose):
        try:
            self._vb = int(input_verbose)
            if self._vb < 0:
                self._vb = 0
            elif 5 < self._vb:
                self._vb = 5
        except ValueError:
            self._vb = 3

    # def _all(self, tm):
    #     """
    #     :param patm.modeling.topic_model.TopicModel tm:
    #     :param reg_name:
    #     :return:
    #     """
    #     return {k: dict(v, **{k:v for k,v in {'target topics': (lambda x: 'all' if len(x) == 0 else x)(tm.get_reg_obj(tm.get_reg_name(k)).topic_names), 'mods': getattr(tm.get_reg_obj(tm.get_reg_name(k)), 'class_ids', None)}.items()}) for k, v in self.static_regularization_specs.items()}

    # def _reg_settings(self, tm, reg_type):
        #
        # target_topics = tm.get_reg_obj(tm.get_reg_name(reg_type)).topic_names
        # target_modalities = getattr(tm.get_reg_obj(tm.get_reg_name(reg_type)), 'class_ids', None)
        # return {k:v for k,v in {'target topics': (lambda x: 'all' if len(x) == 0 else x)(target_topics), 'mods': target_modalities}.items()}

    def tune(self, parameters_mixture, prefix_label='', append_explorables=True, append_static=False, static_regularizers_specs=None, force_overwrite=False, cache_theta=True, verbose=None):
        """
        :param patm.tuning.building.TunerDefinition parameters_mixture: an object encapsulating a unique tuning process; constant parameters and parameters to tune on
        :param str prefix_label: an optional alphanumeric that serves as a coonstant prefix used for the naming files (models, results) saved on disk
        :param list or bool append_explorables: if list is given, then the naming scheme, for the path of the products of a training process
            (model evaluation results and weight matrices saved on disk), will include the values used to do grid-search on,
            corresponding to the specified explorable parameter names. If True, then all the values will be used. If False
            the naming scheme will not use information from the (explorable) parameter value space
        :param list or bool append_static: if list is given, then the naming scheme, for the path of the products of a training process
            (model evaluation results and weight matrices saved on disk), will include the values corresponding to the
            input subset of the parameters selected to remain constant during the grid-search. If True, then all the values
            will be used. If False the naming scheme will not use any information from the parameters selected to remain
            constant during the grid-search
        :param dict static_regularizers_specs: example input:\n
            {'smooth-phi': {'tau': 1.0},\n
            'smooth-theta': {'tau': 1.0, 'alpha_iter': 1.0},\n
            'sparse-phi': {'tau': 'linear_-5_-15', 'start': 4},\n
            'sparse-theta': {'alpha_iter': 1, 'tau': 'linear_-3_-13', 'start': 4}}
        :param bool force_overwrite: whether to overwrite any existing files (json results and phi matrices) on disk, which names collide
            with the newly created files
        :param int verbose: the verbosity level on the stdout
        """
        if verbose:
            self._set_verbosity_level(verbose)
        self._initialize_parameters(parameters_mixture, static_regularizers_specifications=static_regularizers_specs)
        self._initialize_labeling_functionality(prefix_label=prefix_label,
                                                append_explorables=append_explorables,
                                                append_static=append_static,
                                                overwrite=force_overwrite)
        if 1 < self._vb:
            print('Taking {} samples for grid-search'.format(len(self._parameter_grid_searcher)))
        if force_overwrite:
            print('Overwritting any existing results and phi matrices found')
        if self._vb:
            print('Tuning..')
            generator = tqdm(self._parameter_grid_searcher, total=len(self._parameter_grid_searcher), unit='model')
        else:
            generator = iter(self._parameter_grid_searcher)

        self._label_groups = Counter()
        for i, self.parameter_vector in enumerate(generator):
            self._cur_label = self._labeler(i)
            tm, specs = self._create_model_n_specs()
            if 4 < self._vb:
                tqdm.write(pprint.pformat({k: dict(v, **{k:v for k,v in {'target topics': (lambda x: 'all' if len(x) == 0 else '[{}]'.format(', '.join(x)))(tm.get_reg_obj(tm.get_reg_name(k)).topic_names), 'mods': getattr(tm.get_reg_obj(tm.get_reg_name(k)), 'class_ids', None)}.items()}) for k, v in self.static_regularization_specs.items()}))
            if 3 < self._vb:
                tqdm.write(pprint.pformat(tm.modalities_dictionary))
            self.experiment.init_empty_trackables(tm)
            self.trainer.train(tm, specs, cache_theta=cache_theta)
            self.experiment.save_experiment(save_phi=True)
            if 2 < self._vb:
                tqdm.write(self._cur_label)
            # del tm

    def _initialize_parameters(self, tuner_definition, static_regularizers_specifications=None):
        """Call this method to initialize/define:
            - the parameters that will remain steady during the tuning phase\n
            - the parameters that comprise the search space\n
            - the regularizers' tau coefficients that will dynamically change during each training cycle; tau trajectories, if applicable\n
            - the parameters' "grid searcher" object\n
            - the static untunable regularizers' parameters with the provided values or with default ones where necessary
        :param patm.tuning.building.TunerDefinition tuner_definition:
        :param dict static_regularizers_specifications: example input:\n
            {'smooth-phi': {'tau': 1.0},\n
            'smooth-theta': {'tau': 1.0, 'alpha_iter': 1.0},\n
            'sparse-phi': {'tau': 'linear_-5_-15', 'start': 4},\n
            'sparse-theta': {'alpha_iter': 1, 'tau': 'linear_-3_-13', 'start': 4}}
        """
        if self._vb:
            print('Initializing Tuner..')
        self._static_params_hash = tuner_definition.static_parameters
        self._expl_params_hash = tuner_definition.explorable_parameters
        self._check_parameters()
        self._tau_traj_to_build = tuner_definition.valid_trajectory_defs()
        self._parameter_grid_searcher = ParameterGrid(tuner_definition.parameter_spans)
        if 1 < self._vb:
            print('Constants: [{}]\nExplorables: [{}]'.format(', '.join(self.constants), ', '.join(self.explorables)))
            print('Search space', tuner_definition.parameter_spans)
        if static_regularizers_specifications:
            # for which ever of the activated regularizers there is a missing setting, then use a default value
            self._reg_specs = self._create_reg_specs(static_regularizers_specifications, self.active_regularizers)
        else:
            self._reg_specs = self._create_reg_specs(self._default_regularizer_parameters, self.active_regularizers)

    def _initialize_labeling_functionality(self, prefix_label='', append_explorables='all', append_static=None, overwrite=False):
        """Call this method to:
            - define the labeling scheme to use for naming the files created on disk\n
            - determine any potential naming collisions with existing files on disk and resolve by either overwritting or skipping them
        :param str prefix_label: an optional alphanumeric that serves as a constant prefix used for naming the files (models phi matrices dumped on disk, and results saved on disk as jsons)
        :param list or bool append_explorables: if list is given, then the naming scheme, for the path of the products of a training process
            (model evaluation results and weight matrices saved on disk), will include the values used to do grid-search on,
            corresponding to the specified explorable parameter names. If True, then all the values will be used. If False
            the naming scheme will not use information from the (explorable) parameter value space
        :param list or bool append_static: if list is given, then the naming scheme, for the path of the products of a training process
            (model evaluation results and weight matrices saved on disk), will include the values corresponding to the
            input subset of the parameters selected to remain constant during the grid-search. If True, then all the values
            will be used. If False the naming scheme will not use any information from the parameters selected to remain
            constant during the grid-search
        """
        self._prefix = prefix_label
        self._experiments_saved = []
        self._label_groups = Counter()
        # transform potential False boolean input to empty list
        self._define_labeling_scheme((lambda y: [] if not y else y)(append_explorables), (lambda y: [] if not y else y)(append_static))
        if 1 < self._vb:
            print('Automatically labeling files using parameter values: [{}]'.format(', '.join(self._labeling_params)))

        if not overwrite:
            maximal_model_labels = list(map(self._build_label, self._parameter_grid_searcher))  # depends on self.parameter_grid_searcher and self._labeling_params
            results_indices_list, phi_indices_list = self._get_overlapping_indices(maximal_model_labels)
            result_inds, model_inds = IndicesList(results_indices_list, 'train results'), IndicesList(phi_indices_list, 'phi matrix')
            common = result_inds + model_inds  # finds intersection of indices
            only_res = result_inds - common
            only_mods = model_inds - common
            # print 'DEBUG:\ncommon: {}\nonly_res: {}\nonly_mods: {}'.format(list(common), list(only_res), list(only_mods))
            self._parameter_grid_searcher.ommited_indices = common.indices
            self._required_labels = [x for i, x in enumerate(maximal_model_labels) if i not in common.indices]
            if 2 < self._vb:
                for obj in (_ for _ in (common, only_mods, only_res) if _):
                    print(obj.msg(maximal_model_labels))
            if 2 < self._vb:
                print('Models to create:', '[{}]'.format(', '.join(self._required_labels)))
            if 1 < self._vb:
                print('Ommiting creating a total of {} files'.format(len(common)+len(only_res)+len(only_mods)))
            self._labeler = lambda index: self._required_labels[index]
        else:
            self._labeler = lambda index: self._build_label(self.parameter_vector)

    def _create_model_n_specs(self):
        self._reg_specs = self._replace_settings_with_supported_explorable(self._reg_specs)
        tm = self.trainer.model_factory.construct_model(self._cur_label, self._val('nb_topics'),
                                                        self._val('collection_passes'),
                                                        self._val('document_passes'),
                                                        self._val('background_topics_pct'),
                                                        {k:v for k, v in {DEFAULT_CLASS_NAME: self._val('default_class_weight'), IDEOLOGY_CLASS_NAME: self._val('ideology_class_weight')}.items() if v},
                                                        self._score_defs,
                                                        self._active_regs,
                                                        reg_settings=self.static_regularization_specs)
        tr_specs = self.trainer.model_factory.create_train_specs(self._val('collection_passes'))
        return tm, tr_specs

    def _define_labeling_scheme(self, explorables, constants):
        """Call this method to define the values to use for labeling the artifacts of tuning from the mixture of explorable and constant parameters.
            This method also determines if versioning is needed; whether to append strings like v001, v002 to the labels because
            of naming collisions
        """
        assert all([type(x) == list or (type(x) == bool and bool(x)) for x in [constants, explorables]])
        explorables_labels = self._get_labels_extractor('explorables')(explorables)
        constants_labels = self._get_labels_extractor('constants')(constants)
        self._versioning_needed = not explorables_labels == self.explorables
        self._labeling_params = explorables_labels + constants_labels

    def _get_labels_extractor(self, parameters_type):
        extractors_hash = {list: lambda x: [_ for _ in x if _ in self._allowed_labeling_params],
                           bool: lambda x: [_ for _ in getattr(self, parameters_type) if _ in self._allowed_labeling_params]}
        return lambda y: sorted(extractors_hash[type(y)](y))

    def _create_reg_specs(self, reg_settings, active_regularizers):
        """Call this method to use default values where missing, according to given activated regularizers"""
        try:
            return {k: dict([(x[0], reg_settings[k].get(x[0],
                self._default_regularizer_parameters[k][x[0]])) for x in self._default_regularizer_parameters[k].items()]) for k in active_regularizers}
        except KeyError as e:
            raise KeyError("Error: {}. Probably you need to manually update the self._default_regularizer_parameters attribute so that it has the same kyes as the train.cfg".format(str(e)))

    def _get_overlapping_indices(self, maximal_model_labels):
        _ = [self._trans((x[1] in self.experiment.train_results_handler.list, x[1] in self.experiment.phi_matrix_handler.list), x[0]) for x in enumerate(maximal_model_labels)]
        return [filter(lambda y: y is not None, x) for x in list(map(list, zip(*_)))]

    def _trans(self, two_length_tupe_of_bool, index):
        return [index if x else None for x in two_length_tupe_of_bool]

    def _build_label(self, parameter_vector):
        cached = [str(self._extract(parameter_vector, x)) for x in self._labeling_params]
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

    def _replace_settings_with_supported_explorable(self, settings):
        """Gets all regularizers' settings and for each, if it supports a trajectory for 'tau' coefficient value, adds the trajectory definition string to its settings"""
            # _ = map(lambda y: y(1), filter(None, map(lambda x: getattr(re.match('^(sparse_\w+)$', x), 'group', None), self._tau_traj_to_build)))
        # for reg_type in map(lambda x: x.replace('_', '-'), self._tau_traj_to_build):
        #     settings[reg_type] = dict(settings[reg_type], **self._get_tau_trajectory_definition_dict(reg_type.replace('-', '_')))
        # return dict(map(lambda x: (x[0], x[1]), map(lambda x: x.replace('_', '-'), self._tau_traj_to_build)))
        return dict([(x[0], self._get_settings_dict(x[0], x[1])) for x in settings.items()])

    def _get_settings_dict(self, reg_type, settings_dict):
        """Gets a regularizer's settings and if it supports a trajectory for 'tau' coefficient value, add trajectory definition string in settings"""
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
        return {'tau': '_'.join([str(x) for x in (_['kind'], _['start'], _['end'])]),
                'start': _['deactivate']}

    def _val(self, parameter_name):
        return self._extract(self.parameter_vector, parameter_name)

    def _extract(self, parameter_vector, parameter_name):
        """Parameter value extractor from currently generated parameter vector and from static parameters"""
        if parameter_name in self._static_params_hash:
            return self._static_params_hash[parameter_name]
        if parameter_name in self._expl_params_hash:  # and not parameter_name.startswith('sparse_'): #in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory'):
            return parameter_vector[list(self._expl_params_hash.keys()).index(parameter_name)]
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
        missing_required_parameters = [x for x in self._required_parameters if x not in self.constants + self.explorables]
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
        return iter(self._inds)
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
    tuner.tune(d2, prefix_label='tag1', append_explorables=True, append_static=False)
