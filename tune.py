import os
import re
import sys
import argparse

from tqdm import tqdm
from collections import OrderedDict, Counter
from patm.utils import cfg2model_settings, load_results
from patm.modeling import trajectory_builder, regularizers_factory
from patm.modeling.parameters import ParameterGrid
from patm.definitions import collections_dir, regularizers_param_cfg, get_generic_topic_names
from patm import get_model_factory, trainer_factory, Experiment, TrainSpecs
from patm.modeling.regularizers import construct_regularizer


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
    def __init__(self, collection, train_config, regularizers_cfg, static_parameters, explorable_parameters, enable_ideology_labels=False, prefix_label=None, append_explorables='all', append_static=True):
        """

        :param str collection:
        :param str train_config:
        :param str regularizers_cfg:
        :param list_of_tuples static_parameters:
        :param list of tuples explorable_parameters: the order of this affects the order in which vectors of the parameter space are generated
        :param bool enable_ideology_labels:
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
        self._ideology_information_flag = enable_ideology_labels
        t = 0
        for k in self.explorable_parameters:
            self._expl_params_start_indices[k] = t
            if self._regex.match(k): # if trajectory parameter it takes 3 slots
                t += 3
            else:
                t += 1  # else it takes one slot
        print 'TUNER INIT', self._expl_params_start_indices
        self._prefix = prefix_label

        self._max_digits_version = 3
        self._labeling_params = []

        self.trainer = trainer_factory.create_trainer(collection)
        self.experiment = Experiment(self._dir, self.trainer.cooc_dicts)
        self.trainer.register(self.experiment)  # when the model_trainer trains, the experiment object listens to changes

        if append_static:
            self._labeling_params = [i for i,j in static_parameters if not self._regex.match(i)]
        print 'P', self._labeling_params
        self._versioning_needed = False
        if append_explorables == 'all':
            expl_labeling_params = [i for i,j in self._expl_params if not self._regex.match(i)]
        else:
            expl_labeling_params = [i for i in append_explorables if not self._regex.match(i)]
        if len(expl_labeling_params) != len(self._expl_params):
            self._versioning_needed = True
        self._labeling_params.extend(expl_labeling_params)
        print 'P', self._labeling_params

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
        if self._ideology_information_flag:
            # self._tm = self.trainer.model_factory.
            pass
        else:
            self._tm = self.trainer.model_factory.create_model1(self._cur_label, self._val('nb_topics'), self._val('document_passes'), self._train_cfg)

    def _set_parameters(self, reg_specs):
        """

        :param list reg_specs:
        """
        ns = []
        trs = []
        pool = []
        if reg_specs:
            ### SET REGULARIZERS
            _ = int(reg_specs[0] * self._val('nb_topics'))
            background_topics = get_generic_topic_names(self._val('nb_topics'))[:_]
            domain_topics = get_generic_topic_names(self._val('nb_topics'))[_:]

            for l in ('sparse-phi', 'sparse-theta', 'decorrelator-phi'):
                if l in reg_specs[1]:
                    reg_specs[1][l]['topic_names'] = domain_topics
            for l in ('smooth-phi', 'smooth-theta'):
                if l in reg_specs[1]:
                    reg_specs[1][l]['topic_names'] = background_topics
            if 'decorrelator-phi' in reg_specs[1]:
                reg_specs[1]['decorrelator-phi']['tau'] = self._val('decorrelate_topics_reg_coef')

            pool = map(lambda x: regularizers_factory.construct_reg_obj(x[0], x[1]['name'], x[1]), reg_specs[1].items())

            ### SPECIFY THE DOMAIN TOPICS FOR THE SCORERS
            for scorer_type in ('topic-kernel', 'top-tokens-10', 'top-tokens-100'):
                self._tm.get_scorer_by_type(scorer_type)._topic_names = domain_topics

        for reg_obj in pool:
            self._tm.add_regularizer(reg_obj, regularizers_factory.get_type(reg_obj))

        for tau_traj_parameter_name in map(lambda x: x+'_coef_trajectory', self._tau_traj_to_build):
            # print 'DEBUG', tau_traj_parameter_name
            # print type(self._val(tau_traj_parameter_name))
            if type(self._val(tau_traj_parameter_name)) == dict:
                # print 'A1', len(self._build_traj(self._val(tau_traj_parameter_name), self._val('collection_passes')))
                trs.append(self._build_traj(self._val(tau_traj_parameter_name), self._val('collection_passes')))
            else:
                # print 'A2', len(self._val(tau_traj_parameter_name))
                # print self._val(tau_traj_parameter_name)
                trs.append(self._val(tau_traj_parameter_name))
            reg_type = re.match('^(sparse_\w+)_reg_coef_trajectory$', tau_traj_parameter_name).group(1).replace('_', '-')
            ns.append(self._tm.regularizer_names[self._tm.regularizer_types.index(reg_type)])
        # print 'TUNE', ns, '\n', trs[0]._values, '\n', trs[1]._values
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


def get_model_settings(label, dataset):
    base = '/data/thesis/data/collections/{}/results'.format(dataset)
    results = load_results(os.path.join(base, label + '-train.json'))
    reg_set = results['reg_parameters'][-1][1]
    back_topics = {}
    domain_topics = {}
    reg_tau_trajectories = {}
    reg_specs = [0, {}]
    for k,v  in reg_set.items():
        print k
        reg_specs[1][k] = v
        if re.match('^smooth-(?:phi|theta)|^decorrelator-phi', k):
            print 'm1', reg_set[k].keys()
            back_topics[k] = reg_set[k]['topic_names']
        if re.match('^sparse-(?:phi|theta)', k):
            print 'm2', reg_set[k].keys()
            domain_topics[k] = reg_set[k]['topic_names']
            reg_tau_trajectories[k] = [_ for sublist in map(lambda x: [x[1][k]['tau']] * x[0], results['reg_parameters']) for _ in sublist]
    for k in sorted(back_topics.keys()):
        print k, back_topics[k]
    for k in sorted(domain_topics.keys()):
        print k, domain_topics[k]
    if back_topics:
        assert len(set(map(lambda x: '.'.join(x), back_topics.values()))) == 1  # assert that all sparsing adn smoothing regularizers
        # have the same domain and background topics respectively to comply with the restriction in 'tune' method that all regularizers "share" the same domain and background topics.
        assert len(set(map(lambda x: '.'.join(x), domain_topics.values()))) == 1
        reg_specs[0] = float(len(back_topics.values()[0])) / (len(back_topics.values()[0]) + len(domain_topics.values()[0]))
        return reg_specs, reg_tau_trajectories
    return None


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

    # nb_topics = [20, 40, 60, 80, 100]
    # train_iters = [100]
    # doc_iterations = [1, 5, 10, 15]
    #
    # sparse_tau_start = [-0.1, -0.2, -0.3, -0.4]
    # sparse_tau_end = [-1, -2, -3, -4]
    # deactivation_period = [0.1, 0.15, 0.2]
    # splines = ['linear']

    #### PLSA MODELS
    # tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
    #               [('collection_passes', 100)],
    #               [('document_passes', [5, 10, 15]), ('nb_topics', [20, 40, 60])],
    #               prefix_label='plsa',
    #               enable_ideology_labels=False,
    #               append_explorables='all',
    #               append_static=True)
    # tuner.tune([])

    # ############ GRID SEARCH ############
    tr1 = {'deactivation_period_pct': [0.1], 'start': [-0.2], 'end': [-4, -7]}
    tr2 = {'deactivation_period_pct': [0.1], 'start': [-0.1], 'end': [-3, -5]}

    # trajs = map(lambda x: ((x[0]+'_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), iit[1].items())

    # tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
    #               [('collection_passes', 20), ('nb_topics', 20)],
    #               [('document_passes', [5, 10])],
    #               prefix_label='test-b2',
    #               enable_ideology_labels=False,
    #               append_explorables='all',
    #               append_static=True)
    # tuner.tune([1.0, {'smooth-theta': {'name': 'sm_t', 'tau': 1},
    #                       'smooth-phi': {'name': 'sm_p', 'tau': 1}}])

    # ##### PLSA + SMOOTHING + SPARSING MODELS
    # tr1 = {'deactivation_period_pct': [0.1], 'start': [-0.2, -0.5, -0.8], 'end': [-1, -4, -7]} # more "aggressive" regularzation
    # tr2 = {'deactivation_period_pct': [0.1], 'start': [-0.1, -0.3, -0.5], 'end': [-1, -3, -5]} # more "quite" regularzation
    #
    # tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
    #               [('collection_passes', 100), ('document_passes', 5), ('nb_topics', 20)],
    #               [('sparse_phi_reg_coef_trajectory', tr1), ('sparse_theta_reg_coef_trajectory', tr2)],
    #               prefix_label='plsa+sm+sp',
    #               enable_ideology_labels=False,
    #               append_explorables='all',
    #               append_static=True)
    #
    # tuner.tune([0.1, {'sparse-theta': {'name': 'spd_t', 'tau': -0.1}, 'smooth-theta': {'name': 'smb_t', 'tau': 1},
    #                         'sparse-phi': {'name': 'spd_p', 'tau': -0.1}, 'smooth-phi': {'name': 'smb_p', 'tau': 1}}])

    ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL
    # num = [76, 73]
    # # pre_models = ['plsa+sm+sp_v076_20_5_20', 'plsa+sm+sp_v073_20_5_20']
    #
    # for ml in num:
    #     lbl1 = 'plsa+sm+sp_v0{}'.format(ml)
    #     l2 = '_20_5_20'
    #
    #     reg_specs, tau_trajs = get_model_settings(lbl1+l2, args.dataset)
    #     trajs = map(lambda x: ((x[0]+'_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), tau_trajs.items())
    #
    #     ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL AND ADD DECORELATION
    #
    #
    #     tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
    #                   [('collection_passes', 20), ('document_passes', 5), ('nb_topics', 20)] + trajs,
    #                   [('decorrelate_topics_reg_coef', [2e+5, 1.5e+5, 1e+5, 1e+4])],
    #                   prefix_label=lbl1+'+dec',
    #                   enable_ideology_labels=False,
    #                   append_explorables='all',
    #                   append_static=True)
    #     tuner.tune([reg_specs[0], dict(reg_specs[1], **{'decorrelator-phi': {'name': 'decd_p', 'tau': 1e+5}})])

    ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL
    num = [76, 46, 73]
    # pre = [ 'plsa+sm+sp_v076_100_5_20', 'plsa+sm+sp_v076_100_5_20', 'plsa+sm+sp_v073_100_5_20']
    for ml in num:
        lbl1 = 'plsa+sm+sp_v0{}'.format(ml)
        l2 = '_100_5_20'

        reg_specs, tau_trajs = get_model_settings(lbl1 + l2, args.dataset)
        trajs = map(lambda x: ((x[0] + '_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), tau_trajs.items())

        ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL AND ADD DECORELATION
        decor = {'name': 'decd_p', 'tau': 1e+5}
        reg_pool = [reg_specs[0], dict(reg_specs[1], **{'decorrelator-phi': decor})]

        tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
                      [('collection_passes', 100), ('document_passes', 5), ('nb_topics', 20)] + trajs,
                      [('decorrelate_topics_reg_coef', [2e+5, 1.5e+5, 1e+5, 1e+4])],
                      prefix_label=lbl1 + '+dec',
                      enable_ideology_labels=False,
                      append_explorables='all',
                      append_static=True)
        tuner.tune(reg_specs)

        # ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL AND ADD DECORELATION
        # reg_pool = [iit[0][0], dict(iit[0][1], **{'decorrelator-phi': decor})]
        # [('decorrelate_topics_reg_coef', [2e+5, 1e+5, 1e+4, 1e+3])],
        # decor = {'name': 'dec_p', 'tau': 1e+5}

    # ############ GRID SEARCH ############
    # tr1 = {'deactivation_period_pct': [0.2], 'start': [-0.2, -0.5, -0.8], 'end': [-1, -4, -7]}
    # tr2 = {'deactivation_period_pct': [0.2], 'start': [-0.1, -0.3, -0.5], 'end': [-1, -3, -5]}
    # decor = {'name': 'dec_p', 'tau': 1e+5}
    #
    # # iit = get_model_settings('plsa+ssp+sst_v0{}_100_5_20-train.json'.format(vold))
    # iit = get_model_settings('t3-plsa+ssp+sst+dec_v004_100_5_20')
    #
    # trajs = map(lambda x: ((x[0]+'_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), iit[1].items())
    #
    # reg_pool = [iit[0][0], dict(iit[0][1], **{'decorrelator-phi': {'name': 'dec_p', 'tau': 1e+5}})]
    #
    # tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
    #               [('collection_passes', 100), ('document_passes', 5), ('nb_topics', 20)] + trajs,
    #               [('decorrelate_topics_reg_coef', [2e+5, 1e+5, 1e+4, 1e+3])],
    #               prefix_label='test-ide',
    #               enable_ideology_labels=True,
    #               append_explorables=[],
    #               append_static=True)
    # tuner.tune(reg_pool)



    # ############ GRID SEARCH ############
    # tr1 = {'deactivation_period_pct': [0.2], 'start': [-0.2, -0.5, -0.8], 'end': [-1, -4, -7]}
    # tr2 = {'deactivation_period_pct': [0.2], 'start': [-0.1, -0.3, -0.5], 'end': [-1, -3, -5]}
    # decor = {'name': 'dec_p', 'tau': 1e+5}
    #
    # # iit = get_model_settings('plsa+ssp+sst_v0{}_100_5_20-train.json'.format(vold))
    # iit = get_model_settings('t3-plsa+ssp+sst+dec_v004_100_5_20')
    #
    # trajs = map(lambda x: ((x[0]+'_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), iit[1].items())
    #
    # reg_pool = [iit[0][0], dict(iit[0][1], **{'decorrelator-phi': {'name': 'dec_p', 'tau': 1e+5}})]
    #
    # tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
    #               [('collection_passes', 100), ('document_passes', 5), ('nb_topics', 20)] + trajs,
    #               [('decorrelate_topics_reg_coef', [2e+5, 1e+5, 1e+4, 1e+3])],
    #               prefix_label='test-ide',
    #               enable_ideology_labels=True,
    #               append_explorables=[],
    #               append_static=True)
    # tuner.tune(reg_pool)


#####################
    # REGULARIZERS POOL
    # reg_pool_specs = [1, {'sparse-theta': {'name': 'spd_t', 'tau': -0.1}, 'smooth-theta': {'name': 'smb_t', 'tau': 1},
    #                         'sparse-phi': {'name': 'spd_p', 'tau': -0.1}, 'smooth-phi': {'name': 'smb_p', 'tau': 1},
    #                         'decorrelator-phi': {'name': 'ddt', 'tau': 1e+5}}]
    # reg_pool_specs = [1, {'smooth-theta': {'name': 'smb_t', 'tau': 1},
    #                       'smooth-phi': {'name': 'smb_p', 'tau': 1}}]
    ####### EXPLORABLE TRAJECTORIES
    # tr1 = {'deactivation_period_pct': [0.2], 'start': [-0.2, -0.5], 'end': [-3]}
    # tr2 = {'deactivation_period_pct': [0.2], 'start': [-0.1, -0.7], 'end': [-5]}
    # tr0 = {'deactivation_period_pct': 0.2, 'start': -0.4, 'end': -8}
    ##############################
    # static = [('collection_passes', 100), ('document_passes', 5), ('sparse_theta_reg_coef_trajectory', tr2), ('sparse_phi_reg_coef_trajectory', traj1)]
    # wlabel =
    # static = [('collection_passes', 100)]
    # static = [('collection_passes', 100), ('document_passes', 5)]

    # tr1 = {'deactivation_period_pct': [0.1], 'start': [-0.2], 'end': [-4, -7]}
    # tr2 = {'deactivation_period_pct': [0.1], 'start': [-0.1], 'end': [-3, -5]}
