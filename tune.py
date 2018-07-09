import os
import re
import sys
import argparse

from tqdm import tqdm
from collections import OrderedDict, Counter
from patm.utils import cfg2model_settings
from patm.modeling import trajectory_builder, regularizers_factory
from patm.modeling.parameters import ParameterGrid
from patm.definitions import collections_dir, regularizers_param_cfg, get_generic_topic_names
from patm import get_model_factory, trainer_factory, Experiment, TrainSpecs
from patm.modeling.regularizers import construct_regularizer


class Tuner(object):
    """
    This class is able to perform a search over the parameter space (grid-search supported).
    Supports Static:
    - nb_topics              eg: 80
    - collection_passes      eg: 100
    - document_passes        eg:5

    # - sparse_phi_reg_coef_trajectory         eg: []
    # - sparse_theta_reg_coef_trajectory           []

    and Explorable

    - nb_topics                       eg: [20, 40, 60, 80, 100]\n
    - collection_passes               eg: [100]\n
    - document_passes                 eg: [1, 5, 10, 15]\n
    - sparse_phi_reg_coef_trajectory    eg: {'deactivation_period_pct': [0.1, 0.15, 0.2], 'start': [-0.1, -0.2, -0.3, -0.4], 'end' = [-1, -2, -3, -4]}
    - sparse_theta_reg_coef_trajectory  eg: {'deactivation_period_pct': [0.8, 0.14, 0.2], 'start': [-0.2, -0.3, -0.4], 'end' = [-1, -3, -5]}

    """
    def __init__(self, collection, train_config, regularizers_cfg, static_parameters, explorable_parameters, prefix_label=None, append_explorables='all', append_static=True):
        """

        :param str collection:
        :param str train_config:
        :param str regularizers_cfg:
        :param list_of_tuples static_parameters:
        :param list of tuples explorable_parameters: the order of this affects the order in which vectors of the parameter space are generated
        :param str prefix_label: an optional alphanumeric that serves as a coonstant prefix when building the names/paths of the trained models
        :param list or str append_explorables: if parameter names are defined, it appends the parameter values (static or explorable) when building the names/paths of the trained models
        :param bool append_static: indicates whether to append, after any possible prefix, the values of the static parameters used in the tuning process
        """
        self._dir = os.path.join(collections_dir, collection)
        self._train_cfg = train_config
        self._reg_cfg = regularizers_cfg
        self._static_params_hash = OrderedDict(static_parameters)
        self._expl_params = explorable_parameters
        self._expl_params_start_indices = {} # OrderedDict([(k, 0) for k in self.explorable_parameters])
        self._regex = re.compile('^(\w+)_coef_trajectory$')
        t = 0
        for k in self.explorable_parameters:
            self._expl_params_start_indices[k] = t
            if self._regex.match(k):
                t += 3
            else:
                t += 1
        print self._expl_params_start_indices
        self._prefix = prefix_label

        self._max_digits_version = 3
        self._labeling_params = []

        self.trainer = trainer_factory.create_trainer(collection)
        self.experiment = Experiment(self._dir, self.trainer.cooc_dicts)
        self.trainer.register(self.experiment)  # when the model_trainer trains, the experiment object listens to changes


        if append_static:
            self._labeling_params = [i for i,j  in static_parameters if not self._regex.match(i)]
        self._versioning_needed = False
        if append_explorables == 'all':
            expl_labeling_params = [i for i,j  in self._expl_params if not self._regex.match(i)]
        else:
            expl_labeling_params = [i for i in append_explorables if not self._regex.match(i)]
        if len(expl_labeling_params) != len(self._expl_params):
            self._versioning_needed = True
        self._labeling_params.extend(expl_labeling_params)

        self._check_parameters()
        self._tm = None
        self._specs = None
        self._label_groups = Counter()
        self.parameter_vector = []
        self._experiments_saved = []
        self._cur_label = ''

        self._tau_traj_to_build = map(lambda x: x.group(1), filter(None, map(lambda x: self._regex.match(x[0]), static_parameters + explorable_parameters)))

        self._traj_attr2index = OrderedDict([('deactivation_period_pct', 0), ('start', 1), ('end', 2)])
        ses = self._extract_simple_space()+self._extract_trajectory_space()
        if type(ses[0]) == list and type(ses[0][0]) == list:
            ses = [item for subl in ses for item in subl]
        print 'Search space', ses
        self._parameter_grid_searcher = ParameterGrid(ses)

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits == self._max_digits_version:
            return str(int_num)
        return '{}{}'.format((self._max_digits_version - nb_digits) * '0', int_num)

    @property
    def explorable_parameters(self):
        return map(lambda x: x[0], self._expl_params)

    def _build_label(self):
        self._cached = map(lambda x: str(self._val(x)), self._labeling_params)
        self._cur_label = '_'.join(filter(None, [self._prefix] + self._cached))
        self._label_groups[self._cur_label] += 1
        if self._versioning_needed:
            self._cur_label = '_'.join(filter(None, [self._prefix] + ['v{}'.format(self._iter_prepend(self._label_groups[self._cur_label]))] + self._cached))

    def _create_model(self):
        self._tm = self.trainer.model_factory.create_model1(self._cur_label, self._val('nb_topics'), self._val('document_passes'), self._train_cfg)

    def _set_parameters(self, reg_specs):
        """

        :param list reg_specs:
        """
        ns = []
        trs = []
        pool = []
        if reg_specs:
            _ = int(reg_specs[0] * self._val('nb_topics'))
            background_topics = get_generic_topic_names(self._val('nb_topics'))[:_]
            domain_topics = get_generic_topic_names(self._val('nb_topics'))[_:]
            for l in ('sparse-phi', 'sparse-theta', 'decorrelate'):
                if l in reg_specs[1]:
                    reg_specs[1][l]['topic_names'] = domain_topics
            for l in ('smooth-phi', 'smooth-theta'):
                if l in reg_specs[1]:
                    reg_specs[1][l]['topic_names'] = background_topics
            pool = map(lambda x: regularizers_factory.construct_reg_obj(x[0], x[1]['name'], x[1]), reg_specs[1].items())
        for reg_obj in pool:
            self._tm.add_regularizer(reg_obj, regularizers_factory.get_type(reg_obj))

        for tau_traj_parameter_name in map(lambda x: x+'_coef_trajectory', self._tau_traj_to_build):
            assert type(self._val('collection_passes')) == int
            print 'Building', tau_traj_parameter_name, 'trajectory', self._val(tau_traj_parameter_name)
            # print self._expl_params
            trs.append(self._build_traj(self._val(tau_traj_parameter_name), self._val('collection_passes')))

            reg_type = re.match('^(sparse_\w+)_reg_coef_trajectory$', tau_traj_parameter_name).group(1).replace('_', '-')
            ns.append(self._tm.regularizer_names[self._tm.regularizer_types.index(reg_type)])

        self._specs = TrainSpecs(self._val('collection_passes'), ns, trs)

    def _val(self, parameter_name):
        """Parameter value extractor from currently generated parameter vector and from static parameters"""
        if parameter_name in self._static_params_hash:
            return self._static_params_hash[parameter_name]
        if parameter_name in self.explorable_parameters and not self._regex.match(parameter_name): #in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory'):
            return self.parameter_vector[self._expl_params_start_indices[parameter_name]]
        if parameter_name in self.explorable_parameters:
            # print 'VAL', self.parameter_vector, parameter_name, self.explorable_parameters.index(parameter_name),

            return {'deactivation_period_pct': self.parameter_vector[self._expl_params_start_indices[parameter_name]+self._traj_attr2index['deactivation_period_pct']],
                    'start': self.parameter_vector[self._expl_params_start_indices[parameter_name]+self._traj_attr2index['start']],
                    'end': self.parameter_vector[self._expl_params_start_indices[parameter_name]+self._traj_attr2index['end']]}

    def _build_traj(self, traj_def, total_iters):
        deactivation_span = int(traj_def['deactivation_period_pct'] * total_iters)
        return trajectory_builder.begin_trajectory('tau').deactivate(deactivation_span).interpolate_to(total_iters - deactivation_span, traj_def['end'], start=traj_def['start']).create()

    def _extract_simple_space(self):
        return [_[1] for _ in self._expl_params if _[0] not in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory')]

    def _extract_trajectory_space(self):
        return map(lambda x: [x['deactivation_period_pct'], x['start'], x['end']], [v for k, v in self._expl_params if k in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory')])

    def tune(self, reg_specs):
        expected_iters = len(self._parameter_grid_searcher)
        print 'Doing grid-search, taking {} samples'.format(expected_iters)
        for self.parameter_vector in tqdm(self._parameter_grid_searcher, total=expected_iters, unit='model'):
        # for self.parameter_vector in self._parameter_grid_searcher:
            self._build_label()
            self._create_model()
            self._set_parameters(reg_specs)
            self.experiment.init_empty_trackables(self._tm)
            self.trainer.train(self._tm, self._specs)
            self.experiment.save_experiment(save_phi=True)
            del self._tm
        self._experiments_saved.extend([os.path.basename(_) for _ in self.experiment.train_results_handler.saved[-expected_iters:]])
        print 'Produced {} experiments'.format(len(self._experiments_saved[-expected_iters:]))
        for _ in self._experiments_saved[-expected_iters:]:
            print _

    def _check_parameters(self):
        for i in self._static_params_hash.keys():
            if i in self.explorable_parameters:
                raise ParameterFoundInStaticAndExplorablesException("Parameter '{}' defined both as static and explorable".format(i))
        missing = []
        for i in ('nb_topics', 'document_passes', 'collection_passes'):
            if i not in self._static_params_hash.keys() + self.explorable_parameters:
                missing.append(i)
        if missing:
            raise MissingRequiredParametersException("Missing <{}> required model parameters".format(', '.join(missing)))

class MissingRequiredParametersException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)
class ParameterFoundInStaticAndExplorablesException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Performs grid-search over the parameter space by creating and training topic models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('train_cfg', help='the fixed model parameters and regularizers to use. Regularizers can be turned on/off after consecutive train iterations through dynamic tau trajectory, but no new regularizer can be added after initialization')
    return parser.parse_args()

if __name__ == '__main__':
    from patm.modeling.regularizers import regularizer_pool_builder
    ## CONSTANTS ##
    args = get_cli_arguments()
    reg_cfg = '/data/thesis/code/regularizers.cfg'

    nb_topics = [20, 40, 60, 80, 100]
    train_iters = [100]
    doc_iterations = [1, 5, 10, 15]

    sparse_tau_start = [-0.1, -0.2, -0.3, -0.4]
    sparse_tau_end = [-1, -2, -3, -4]
    deactivation_period = [10, 20, 30]
    deactivation_period_pct = [10, 20, 30]
    splines = ['linear']

    nb_topics = [20, 40]
    train_iters = [100]
    doc_iterations = [10]

    p_sparse_tau_start = [-0.1, -0.2, -0.3, -0.4]
    p_sparse_tau_end = [-1, -2, -3, -4]
    p_deactivation_period = [10, 20, 30]
    p_deactivation_period_pct = [10, 20, 30]

    t_sparse_tau_start = [-0.1, -0.2, -0.3, -0.4]
    t_sparse_tau_end = [-1, -2, -3, -4]
    t_deactivation_period = [10, 20, 30]
    t_deactivation_period_pct = [10, 20, 30]

    #####################
    # REGULARIZERS POOL

    reg_pool_specs = [1, {'sparse-theta': {'name': 'spd_t', 'tau': -0.1}, 'smooth-theta': {'name': 'smb_t', 'tau': 1},
                            'sparse-phi': {'name': 'spd_p', 'tau': -0.1}, 'smooth-phi': {'name': 'smb_p', 'tau': 1},
                            'decorrelator-phi': {'name': 'ddt', 'tau': 1e+5}}]
    reg_pool_specs = [1, {'smooth-theta': {'name': 'smb_t', 'tau': 1},
                          'smooth-phi': {'name': 'smb_p', 'tau': 1}}
                          ]
    ####### EXPLORABLE TRAJECTORIES
    tr0 = {'deactivation_period_pct': 0.2, 'start': -0.4, 'end': -8}
    tr1 = {'deactivation_period_pct': [0.2], 'start': [-0.2, -0.5, -0.8], 'end': [-1, -4, -7]}
    tr2 = {'deactivation_period_pct': [0.2], 'start': [-0.1, -0.3, -0.5], 'end': [-1, -3, -5]}
    tr1 = {'deactivation_period_pct': [0.2], 'start': [-0.2, -0.5], 'end': [-3]}
    tr2 = {'deactivation_period_pct': [0.2], 'start': [-0.1, -0.7], 'end': [-5]}
    ##############################

    # static = [('collection_passes', 100), ('document_passes', 5), ('sparse_theta_reg_coef_trajectory', tr2), ('sparse_phi_reg_coef_trajectory', traj1)]
    # wlabel =
    # static = [('collection_passes', 100)]
    # static = [('collection_passes', 100), ('document_passes', 5)]

    tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
                  [('collection_passes', 10), ('document_passes', 5), ('nb_topics', 20), ('sparse_theta_reg_coef_trajectory', tr0)],
                  [('sparse_phi_reg_coef_trajectory', tr2)],
                  prefix_label='plsa+sss+test1',
                  append_explorables=[],
                  append_static=True)

    tuner.tune([0.2, {'sparse-theta': {'name': 'spd_t', 'tau': -0.1}, 'smooth-theta': {'name': 'smb_t', 'tau': 1},
                            'sparse-phi': {'name': 'spd_p', 'tau': -0.1}, 'smooth-phi': {'name': 'smb_p', 'tau': 1}}])
