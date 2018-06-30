import os
import re
import sys
import argparse
from tqdm import tqdm
from collections import OrderedDict
from patm.utils import cfg2model_settings
from patm.modeling import trajectory_builder
from patm.modeling.parameters import ParameterGrid
from patm.definitions import collections_dir, regularizers_param_cfg
from patm import get_model_factory, trainer_factory, Experiment, TrainSpecs


class Tuner(object):
    """
    This class is able to perform a search over the parameter space (grid-search supported).
    Supports Static:
    - nb_topics              eg: 80
    - collection_passes      eg: 100
    - document_passes        eg:5

    # - phi_trajectory         eg: []
    # - theta_trajectory []

    and Explorable

    - nb_topics                       eg: [20, 40, 60, 80, 100]\n
    - collection_passes               eg: [100]\n
    - document_passes                 eg: [1, 5, 10, 15]\n
    - sparse_phi_reg_coef_trajectory    eg: {'deactivation_period_pct': [10, 15, 20], 'start': [-0.1, -0.2, -0.3, -0.4], 'end' = [-1, -2, -3, -4]}
    - sparse_theta_reg_coef_trajectory  eg: {'deactivation_period_pct': [8, 14, 20], 'start': [-0.2, -0.3, -0.4], 'end' = [-1, -3, -5]}

    # deactivation_period = [10, 20, 30]

    """
    def __init__(self, collection, train_config, reg_cfg, static_parameters, explorable_parameters, model_type=None, append=None):
        """

        :param str collection:
        :param str train_config:
        :param reg_cfg:
        :param dict static_parameters:
        :param list of tuples explorable_parameters:
        :param str model_type:
        """
        self._dir = os.path.join(collections_dir, collection)
        self.experiment = Experiment(self._dir)
        self.trainer = trainer_factory.create_trainer(collection)
        self.trainer.register(self.experiment) # when the model_trainer trains, the experiment object listens to changes
        self.settings = cfg2model_settings(train_config)
        self._train_cfg = train_config
        self._reg_cfg = reg_cfg
        self._static_parameters = static_parameters
        self._expl_params = explorable_parameters
        self._explore_dict = dict(self._expl_params)
        self._tm = None
        self._specs = None

        self.parameter_vector = []
        if model_type:
            self._label = model_type
        else:
            self._label = '-'.join(sorted(self.settings['regularizers'].values()))
        self._labeling = []
        if append:
            self._labeling = append

        self._max_digits_version = 3
        regex = re.compile('^(\w+)_coef_trajectory$')

        self._tau_traj_to_build = map(lambda x: x.group(1), filter(None, map(lambda x: regex.match(x[0]), self._expl_params)))
        if self._tau_traj_to_build:
            self._tau_traj_to_build = sorted(self._tau_traj_to_build.keys())
        self._tau_traj_to_build = set(self._tau_traj_to_build)  # maximal set: ['sparse_phi_reg', 'sparse_theta_reg']

        self._traj_attr2index = OrderedDict([('deactivation_period_pct', 0), ('start', 1), ('end', 2)])
        self._traj_param_name2reg_type = {
            'sparse_phi_reg': 'smooth-phi',
            'sparse_theta_reg': 'smooth-theta'}
        self._vector_gen = {'grid-search': ParameterGrid([_ for _ in self._extract_simple_space()] + [__ for __ in self._extract_trajectory_space()])}

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits == self._max_digits_version:
            return str(int_num)
        return '{}{}'.format((self._max_digits_version - nb_digits) * '0', int_num)

    @property
    def explorable_parameters(self):
        return map(lambda x: x[0], self._expl_params)

    def _get_label(self, i):
        _ = '{}-v{}_{}'.format(self._label, self._iter_prepend(i), '_'.join([str(self._val(_)) for _ in self._static_parameters.keys()] + [str(self._val(_)) for _ in self._labeling]))
        if _[-1] == '_':
            return _[:-1]
        return _

    def _create_model(self, label):
        self._tm = self.trainer.model_factory.create_model1(label, self._val('nb_topics'), self._val('document_passes'), self._train_cfg, self._reg_cfg)

    def _set_parameters(self):
        if self._tau_traj_to_build:
            self._traj_param_name2reg_name = {traj_type: self._tm.get_reg_name[self._traj_param_name2reg_type[traj_type]] for traj_type in self._tau_traj_to_build}
            traj_definitions = filter(None, map(lambda x: x if x[1] else None, self._traj_param_name2reg_name.items()))
            names = filter(None, map(lambda x: x[1] if x[1] else None, traj_definitions))
            trajectories = map(lambda x: self._traj(self._traj_param_name2reg_name[x[0]]), traj_definitions)
            assert len(trajectories) == len(names)
            self._specs = TrainSpecs(self._val('collection_passes'), names, trajectories)
        else:
            self._specs = TrainSpecs(self._val('collection_passes'), [], [])

    def _val(self, parameter_name):
        """Parameter value extractor from currently generated parameter vector and from static parameters"""
        if parameter_name in self._static_parameters:
            return self._static_parameters[parameter_name]
        if parameter_name in self.explorable_parameters and parameter_name not in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory'):
            return self.parameter_vector[self.explorable_parameters.index(parameter_name)]
        return None

    def _traj(self, parameter_name):
        deact_iters = int(self._val('collection_passes')*self.parameter_vector[self.explorable_parameters.index(parameter_name)+self._traj_attr2index['deactivation_period_pct']])
        inter_iters = self._val('collection_passes') - deact_iters
        start_value = self.parameter_vector[self.explorable_parameters.index(parameter_name)+self._traj_attr2index['start']]
        end_value = self.parameter_vector[self.explorable_parameters.index(parameter_name)+self._traj_attr2index['end']]
        return trajectory_builder.begin_trajectory('tau').deactivate(deact_iters).interpolate_to(inter_iters, end_value, start=start_value).create()

    def _extract_simple_space(self):
        for i in self.explorable_parameters:
            if i not in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory'):
                yield self._explore_dict[i]

    def _extract_trajectory_space(self):
        for i in self.explorable_parameters:
            if i in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory'):
                yield [self._explore_dict[i][tau_traj_sub_param] for tau_traj_sub_param in self._traj_attr2index.keys()]

    def tune(self, search='grid-search'):
        # for i, self.parameter_vector in enumerate(self._vector_gen[search]):
        i = 1
        for self.parameter_vector in tqdm(self._vector_gen[search], total=len(self._vector_gen[search]), unit='models'):
            cur_label = self._get_label(i)
            # print i, cur_label, self.parameter_vector
            self._create_model(cur_label)
            self._set_parameters()
            self.experiment.init_empty_trackables(self._tm)
            self.trainer.train(self._tm, self._specs)
            self.experiment.save_experiment(save_phi=True)
            del self._tm
            i += 1


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Performs grid-search over the parameter space by creating and training topic models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('train_cfg', help='the fixed model parameters and regularizers to use. Regularizers can be turned on/off after consecutive train iterations through dynamic tau trajectory, but no new regularizer can be added after initialization')
    return parser.parse_args()

if __name__ == '__main__':
    ## CONSTANTS ##
    args = get_cli_arguments()
    regularizers_cfg = '/data/thesis/code/regularizers.cfg'

    nb_topics = [20, 40, 60, 80, 100]
    train_iters = [100]
    doc_iterations = [1, 5, 10, 15]

    sparse_tau_start = [-0.1, -0.2, -0.3, -0.4]
    sparse_tau_end = [-1, -2, -3, -4]
    deactivation_period = [10, 20, 30]
    deactivation_period_pct = [10, 20, 30]
    splines = ['linear']
    #########################################

    wlabel = 'plsa'
    static = {'collection_passes': 100}
    explorables = [('nb_topics', nb_topics), ('document_passes', doc_iterations)]

    tuner = Tuner(args.dataset, args.train_cfg, regularizers_cfg, static, explorables, model_type=wlabel, append=['nb_topics', 'document_passes'])
    tuner.tune()
