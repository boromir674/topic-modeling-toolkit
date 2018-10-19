import os
import re
import sys
import warnings
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
        self._dir = os.path.join(COLLECTIONS_DIR, collection)
        if evaluation_definitions:
            self._score_defs = evaluation_definitions
        else:
            self._score_defs = get_standard_evaluation_definitions()
        self._required_labels = []
        self._reg_specs = {}
        self._max_digits_version = 3
        self._allowed_labeling_params = ('collection_passes', 'nb_topics', 'document_passes', 'background_topics_pct', 'ideology_class_weight')
        self._active_reg_def_builder = RegularizersActivationDefinitionBuilder(tuner=self)
        self.trainer = trainer_factory.create_trainer(collection, exploit_ideology_labels=True)  # forces to use (and create if not found) batches holding modality information in case it is needed
        self.experiment = Experiment(self._dir, self.trainer.cooc_dicts)
        self.trainer.register(self.experiment)  # when the model_trainer trains, the experiment object listens to changes
        self._parameter_grid_searcher = None

    @property
    def explorable_parameters_names(self):
        return map(lambda x: x[0], self._expl_params_hash)

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
        if type(active_regularizers) == list:
            self._create_active_regs_with_default_names(active_regularizers)
        elif isinstance(active_regularizers, dict):
            self._active_regs = active_regularizers
        else:
            raise TypeError

    def _create_active_regs_with_default_names(self, reg_types_list):
        self._active_regs = OrderedDict(map(lambda y: ('-'.join(y), ''.join(map(lambda z: z[0] if len(y) > 2 else z[:2], y))), map(lambda x: x.split('-'), reg_types_list)))

    @property
    def static_regularization_specs(self):
        return self._reg_specs

    @static_regularization_specs.setter
    def static_regularization_specs(self, regularizers_specs):
        self._reg_specs = regularizers_specs

    def tune(self, parameters_mixture, prefix_label='', append_explorables='all', append_static=True, static_regularizers_specs=None, force_overwrite=False, verbose=False):
        """
        :param patm.tuning.building.TunerDefinition parameters_mixture:
        :param str prefix_label: an optional alphanumeric that serves as a coonstant prefix used for the naming files (models, results) saved on disk
        :param list or str append_explorables: if list is given will use these exact parameters' names as part of unique names used for files saved on disk. If 'all' is given then all possible parameter names will be used
        :param bool append_static: indicates whether to use the values of the parameters remaining constant during space exploration, as part of the unique names used for files saved on disk
        :return:
        """
        self._initialize0(parameters_mixture, verbose=verbose)
        self._initialize1(prefix_label=prefix_label, append_explorables=append_explorables, append_static=append_static, static_regularizers_specs=static_regularizers_specs, overwrite=force_overwrite)
        expected_iters = len(self._parameter_grid_searcher)
        print 'Doing grid-search, taking {} samples'.format(expected_iters)
        self._label_groups = Counter()
        for i, self.parameter_vector in enumerate(tqdm(self._parameter_grid_searcher, total=expected_iters, unit='model')):
            # print 'TUNE on vector:', self.parameter_vector
            self._cur_label = self._build_label(self.parameter_vector)
            assert self._cur_label == self._required_labels[i]
            tm, specs = self._create_model_n_specs()
            self.experiment.init_empty_trackables(tm)
            self.trainer.train(tm, specs)
            self.experiment.save_experiment(save_phi=True)
            del tm

    def _initialize0(self, tuner_definition, verbose=False):
        """
        :param patm.tuning.building.TunerDefinition tuner_definition:
        """
        print 'Initializing Tuner..'
        self._static_params_hash = tuner_definition.static_parameters
        self._expl_params_hash = tuner_definition.explorable_parameters
        self._check_parameters()
        self._tau_traj_to_build = tuner_definition.valid_trajectory_defs()
        self._parameter_grid_searcher = ParameterGrid(tuner_definition.parameter_spans)
        if verbose:
            print 'STATIC:', self._static_params_hash.keys(), '\nEXPL:', self._expl_params_hash.keys()
            print 'Tau trajectories for:', self._tau_traj_to_build.keys()
            print 'Search space', tuner_definition.parameter_spans
            print 'Potential GRID LENGTH:', len(self._parameter_grid_searcher)

    def _initialize1(self, prefix_label='', append_explorables='all', append_static=True, static_regularizers_specs=None, overwrite=False):
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
        self._labeling_params = []
        self._overwrite = overwrite

        if append_static:
            self._labeling_params = [param_name for param_name in self._static_params_hash.keys() if param_name in self._allowed_labeling_params]
        self._versioning_needed = False

        expl_labeling_params = []
        if append_explorables == 'all':
            expl_labeling_params = [i for i in self._expl_params_hash.keys() if i in self._allowed_labeling_params]
        elif append_explorables:
            expl_labeling_params = [i for i in append_explorables if i in self._allowed_labeling_params]
        if len(expl_labeling_params) != len(self._expl_params_hash):
            self._versioning_needed = True

        self._labeling_params.extend(expl_labeling_params)
        print 'Automatic labeling files using:', self._labeling_params, 'values'
        if static_regularizers_specs is not None:
            self.static_regularization_specs = static_regularizers_specs
        else:
            self.static_regularization_specs = self._get_default_parameters()

        self._build_required_labels()  # depends on self.parameter_grid_searcher and self._labeling_params
        print 'REQ', self._required_labels
        _ = self._get_overlapping_indices()
        result_inds, model_inds = IndicesList(_[0], 'train results'), IndicesList(_[1], 'phi matrix')
        common = result_inds + model_inds  # finds intersection of indices
        only_res = result_inds - common
        only_mods = model_inds - common

        for obj in (_ for _ in (common, only_mods, only_res) if _):
            print obj.msg(self._required_labels)
        if not self._overwrite:
            self._parameter_grid_searcher.ommited_indices = common.indices
            self._required_labels = [x for i, x in enumerate(self._required_labels) if i not in common.indices]
            print 'REQ', self._required_labels
        print 'Actual GRID LENGTH:', len(self._parameter_grid_searcher)
        # '\nvectors', [_ for _ in self._parameter_grid_searcher._get_filtered_generator()]
        # import sys
        # sys.exit()

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

    def _get_default_parameters(self):
        reg_specs = {}
        for reg_type in self._active_regs.keys():
            if reg_type.startswith('smooth-'):
                reg_specs[reg_type] = {'tau': 1.0}
            elif reg_type.startswith('sparse-'):
                reg_specs[reg_type] = {}
                if 'sparse_' + reg_type.split('-')[1] not in self._tau_traj_to_build:
                    warnings.warn("Activated a '{}' regularizer, but did not define a tau coefficient value trajectory".format(reg_type))
            if reg_type.endswith('-theta'):
                reg_specs[reg_type]['alpha_iter'] = 1.0
        return reg_specs

    def _create_model_n_specs(self):
        for reg_type in (_ for _ in self._active_regs.keys() if _.startswith('sparse_')):
            self._reg_specs[reg_type] = self._build_definition(self._val('sparse_' + reg_type.split('-')[1]))
        tm = self.trainer.model_factory.construct_model(self._cur_label, self._val('nb_topics'),
                                                              self._val('collection_passes'),
                                                              self._val('document_passes'),
                                                              self._val('background_topics_pct'),
                                                              {DEFAULT_CLASS_NAME: self._val('default_class_weight'), IDEOLOGY_CLASS_NAME: self._val('ideology_class_weight')},
                                                              self._score_defs,
                                                              self._active_regs,
                                                              reg_settings=self._reg_specs)
        tr_specs = self.trainer.model_factory.create_train_specs(self._val('collection_passes'))
        return tm, tr_specs

    def _build_definition(self, traj_dict):
        print 'BUILD DEFINITION', traj_dict
        if traj_dict is None:
            return None
        return {'tau': '_'.join(map(lambda x: str(x) ,(traj_dict['kind'], traj_dict['start'], traj_dict['end']))), 'start': traj_dict.get('deactivate', 0)}

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
            if i in self.explorable_parameters_names:
                raise ParameterFoundInStaticAndExplorablesException("Parameter '{}' defined both as static and explorable".format(i))
        missing = []
        for i in ('nb_topics', 'document_passes', 'collection_passes'):
            l = self._static_params_hash.keys() + self._expl_params_hash.keys()
            if i not in l:
                missing.append(i)
        if missing:
            print self._static_params_hash.keys()
            print self._expl_params_hash.keys()
            raise MissingRequiredParametersException("Missing <{}> required model parameters".format(', '.join(missing)))


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
