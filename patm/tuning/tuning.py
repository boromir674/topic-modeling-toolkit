import os
import re
import sys
from tqdm import tqdm
from collections import OrderedDict, Counter

from patm.modeling.parameters import ParameterGrid
from patm.definitions import COLLECTIONS_DIR, TRAIN_CFG
from patm import trainer_factory, Experiment, TrainSpecs
from patm.modeling import trajectory_builder, regularizers_factory
from patm.utils import get_standard_evaluation_definitions as get_standard_scores


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
    def __init__(self, collection, evaluation_definitions=None, prefix_label=None, append_explorables='all', append_static=True):
        """

        :param str collection: the name of the 'dataset'/collection to target the tuning process on
        :param str train_config: full path to 'the' train.cfg file used here only for initializing tracking evaluation scoring capabilities. Namely the necessary BaseScore objects of ARTM lib and the custom ArtmEvaluator objects are initialized
        :param list_of_tuples static_parameters: definition of parameters that will remain constant while exploring the parameter space. The order of this affects the order in which vectors of the parameter space are generated
            ie: [('collection_passes', 100), ('nb_topics', 20), ('document_passes', 5))].
        :param list of tuples explorable_parameters_names: definition of parameter "ranges" that will be explored; the order of list affects the order in which vectors of the parameter space are generated; ie
        :param bool enable_ideology_labels: Enable using the 'vowpal_wabbit' format to exploit extra modalities modalities. A modalitity is a disctreet space of values (tokens, class_labels, ..)
        :param str prefix_label: an optional alphanumeric that serves as a coonstant prefix used for the naming files (models, results) saved on disk
        :param list or str append_explorables: if list is given will use these exact parameters' names as part of unique names used for files saved on disk. If 'all' is given then all possible parameter names will be used
        :param bool append_static: indicates whether to use the values of the parameters remaining constant during space exploration, as part of the unique names used for files saved on disk
        """
        self._dir = os.path.join(COLLECTIONS_DIR, collection)
        if evaluation_definitions:
            self._score_defs = evaluation_definitions
        else:
            self._score_defs = get_standard_scores()

        self._regex = re.compile('^(\w+)_coef_trajectory$')
        self._max_digits_version = 3
        self.trainer = trainer_factory.create_trainer(collection)
        self.experiment = Experiment(self._dir, self.trainer.cooc_dicts)
        self.trainer.register(self.experiment)  # when the model_trainer trains, the experiment object listens to changes

        self._traj_attr2index = OrderedDict([('deactivation_period_pct', 0), ('start', 1), ('end', 2)])
        self._tm = None
        self._specs = None
        # self.initialize(prefix_label=prefix_label, append_explorables=append_explorables, append_static=append_static)
        self._reg_specs = {}


        ### NEW BODY
        self._allowed_labeling_params = ('collection_passes', 'nb_topics', 'document_passes', 'background_topics_pct', 'ideology_class_weight')

    @property
    def regularization_specs(self):
        return self._reg_specs

    @regularization_specs.setter
    def regularization_specs(self, regularizers_specs):
        self._reg_specs = regularizers_specs

    def _initialize(self, tuner_definition, static_parameters=None, explorable_parameters=None, prefix_label='', enable_ideology_labels=False, append_explorables='all', append_static=True):
        """
        :param patm.tuning.parameter_building.TunerDefinition tuner_definition:
        :param list_of_tuples static_parameters: definition of parameters that will remain constant while exploring the parameter space. The order of this affects the order in which vectors of the parameter space are generated
            ie: [('collection_passes', 100), ('nb_topics', 20), ('document_passes', 5))].
        :param list of tuples explorable_parameters: definition of parameter "ranges" that will be explored; the order of list affects the order in which vectors of the parameter space are generated; ie
        :param str prefix_label: an optional alphanumeric that serves as a coonstant prefix used for the naming files (models, results) saved on disk
        :param bool enable_ideology_labels: Enable using the 'vowpal_wabbit' format to exploit extra modalities modalities. A modalitity is a disctreet space of values (tokens, class_labels, ..)
        :param list or str append_explorables: if list is given will use these exact parameters' names as part of unique names used for files saved on disk. If 'all' is given then all possible parameter names will be used
        :param bool append_static: indicates whether to use the values of the parameters remaining constant during space exploration, as part of the unique names used for files saved on disk
        """
        self._static_params_hash = tuner_definition.static_parameters
        self._expl_params_hash = tuner_definition.explorable_parameters
        self._prefix = prefix_label
        self._experiments_saved = []
        self._label_groups = Counter()
        self._labeling_params = []

        # TO DELETE
        # self._ideology_information_flag = enable_ideology_labels
        # if append_static:
        #     self._labeling_params = [i for i,j in static_parameters if not self._regex.match(i)]
        # TO DELETE

        ## DECIDE
        self._expl_params_start_indices = {}  # OrderedDict([(k, 0) for k in self.explorable_parameters_names])

        t = 0
        for k in self.explorable_parameters_names:
            self._expl_params_start_indices[k] = t
            if self._regex.match(k):  # if trajectory parameter it takes 3 slots
                t += 3
            else:
                t += 1  # else it takes one slot
        ## DECIDE

        if append_static:
            self._labeling_params = [param_name for param_name in self._static_params_hash.keys() if param_name in self._allowed_labeling_params]

        self._versioning_needed = False
        if append_explorables == 'all':
            expl_labeling_params = [i for i in self._expl_params_hash.keys() if i in self._allowed_labeling_params]
        else:
            expl_labeling_params = [i for i in append_explorables if i in self._allowed_labeling_params]

        if len(expl_labeling_params) != len(self._expl_params_hash):
            self._versioning_needed = True

        self._labeling_params.extend(expl_labeling_params)
        print 'Initializing tuner, automatically labeling output using', self._labeling_params, 'parameters'

        self._check_parameters()
        self._tau_traj_to_build = tuner_definition.valid_trajectory_defs
        # self._tau_traj_to_build = map(lambda x: x.group(1), filter(None, map(lambda x: self._regex.match(x[0]),
        #                                                                      static_parameters + explorable_parameters)))

        ## REFACTOR
        # ses = self._extract_simple_space() + self._extract_trajectory_space()
        # if type(ses[0]) == list and type(ses[0][0]) == list:
        #     ses = [item for subl in ses for item in subl]
        ses = tuner_definition.parameter_spans
        print 'Search space', tuner_definition.explorable_parameters.keys()
        print 'Search space', ses

        self._parameter_grid_searcher = ParameterGrid(ses)

        ## REFACTOR
    def set_dataset(self, dataset_name):
        self._dir = os.path.join(COLLECTIONS_DIR, dataset_name)

    @property
    def explorable_parameters_names(self):
        return map(lambda x: x[0], self._expl_params_hash)

    def tune(self, parameters_mixture):
        """
        :param patm.modeling.tuning.parameter_building.TunerDefinition parameters_mixture:
        :return:
        """
        self._initialize(parameters_mixture)
        expected_iters = len(self._parameter_grid_searcher)
        print 'Doing grid-search, taking {} samples'.format(expected_iters)
        import sys
        sys.exit()
        for self.parameter_vector in tqdm(self._parameter_grid_searcher, total=expected_iters, unit='model'):
            self._build_label()
            self._set_parameters()
            tm, specs = self._create_model_n_specs()
            self.experiment.init_empty_trackables(tm)
            self.trainer.train(tm, specs)
            self.experiment.save_experiment(save_phi=True)
            del tm

    def tune_old(self, reg_specs):
        """
        Performs grid search over the parameter space while potentially utilizing regularizers.\n
        :param list reg_specs: el[0] in [0, 1] specifying the percentage of topics to be used for background topics; ie 0.2, 0.1. 0 indicates no usage of the "background-token feature"
            el[1] = dict with attributes of desired regularizers to use. Trajectories given through constructor or setters override any tau value defined here.
            ie [0.1, {'sparse-theta': {'name': 'spd_t'}, 'smooth-theta': {'name': 'smb_t', 'tau': 1}, 'sparse-phi': {'name': 'spd_p'}, 'smooth-phi': {'name': 'smb_p', 'tau': 1}}]
        """
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

    def _build_label(self):
        self._cached = map(lambda x: str(self._val(x)), self._labeling_params)
        self._cur_label = '_'.join(filter(None, [self._prefix] + self._cached))
        self._label_groups[self._cur_label] += 1
        if self._versioning_needed:
            self._cur_label = '_'.join(filter(None, [self._prefix] + ['v{}'.format(self._iter_prepend(self._label_groups[self._cur_label]))] + self._cached))

    def _create_model(self):

        self_tm = self.trainer
        if self._ideology_information_flag:
            # self._tm = self.trainer.model_factory.
            pass
        else:
            self._tm = self.trainer.model_factory.create_model1(self._cur_label, self._val('nb_topics'), self._val('document_passes'), self._train_cfg, modality_weights=None)
            # print dir(self._tm.artm_model)
            print self._tm.artm_model.scorer_tracker('')
            import sys
            sys.exit()

    def _set_parameters(self, reg_specs):
        # TODO refactor this using model_factory constructors accordingly
        """
        :param list reg_specs:
        """
        ns = []
        trs = []
        pool = []
        if reg_specs:
            ### SET REGULARIZERS
            background_topics, domain_topics = generic_topic_names_builder.define_nb_topics(self._val('nb_topics')).define_background_pct(reg_specs[0]).get_background_n_domain_topics()
            # _ = int(reg_specs[0] * self._val('nb_topics'))
            # background_topics = get_generic_topic_names(self._val('nb_topics'))[:_]
            # domain_topics = get_generic_topic_names(self._val('nb_topics'))[_:]

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
            #  type(self._val(tau_traj_parameter_name))
            if type(self._val(tau_traj_parameter_name)) == dict:
                # print 'A1', len(self._build_traj_from_unique_definition(self._val(tau_traj_parameter_name), self._val('collection_passes')))
                trs.append(self._build_traj_from_unique_definition(self._val(tau_traj_parameter_name), self._val('collection_passes')))
            else:
                # print 'A2', len(self._val(tau_traj_parameter_name)), '\n', self._val(tau_traj_parameter_name)
                trs.append(self._val(tau_traj_parameter_name))
            reg_type = re.match('^(sparse_\w+)_reg_coef_trajectory$', tau_traj_parameter_name).group(1).replace('_', '-')
            ns.append(self._tm.regularizer_names[self._tm.regularizer_types.index(reg_type)])
        # print 'TUNE', ns, '\n', trs[0]._values, '\n', trs[1]._values
        self._specs = TrainSpecs(self._val('collection_passes'), ns, trs)

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits == self._max_digits_version:
            return str(int_num)
        return '{}{}'.format((self._max_digits_version - nb_digits) * '0', int_num)

    def _val(self, parameter_name):
        """Parameter value extractor from currently generated parameter vector and from static parameters"""
        if parameter_name in self._static_params_hash:
            return self._static_params_hash[parameter_name]
        if parameter_name in self.explorable_parameters_names and not self._regex.match(parameter_name): #in ('sparse_phi_reg_coef_trajectory', 'sparse_theta_reg_coef_trajectory'):
            return self.parameter_vector[self._expl_params_start_indices[parameter_name]]
        if parameter_name in self.explorable_parameters_names:
            # print 'VAL', self.parameter_vector, parameter_name, self.explorable_parameters_names.index(parameter_name),
            return {'deactivation_period_pct': self.parameter_vector[self._expl_params_start_indices[parameter_name]+self._traj_attr2index['deactivation_period_pct']],
                    'start': self.parameter_vector[self._expl_params_start_indices[parameter_name]+self._traj_attr2index['start']],
                    'end': self.parameter_vector[self._expl_params_start_indices[parameter_name]+self._traj_attr2index['end']]}

    def _build_traj_from_unique_definition(self, traj_def, total_iters):
        for k, v in traj_def.items():
            if type(v) == list:
                assert len(v) == 1
                traj_def[k] = v[0]
        deactivation_span = int(traj_def['deactivation_period_pct'] * total_iters)
        return trajectory_builder.begin_trajectory('tau').deactivate(deactivation_span).interpolate_to(total_iters - deactivation_span, traj_def['end'], start=traj_def['start']).create()

    def _extract_simple_space(self):
        return [v for k, v in self._expl_params_hash.items() if not k.startswith('sparse_')]

    def _extract_trajectory_space(self):
        return map(lambda x: [x['deactivation_period_pct'], x['start'], x['end']], [v for k, v in self._expl_params_hash.items() if k.startswith('sparse_')])

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


class MissingRequiredParametersException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

class ParameterFoundInStaticAndExplorablesException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class TunerBuilder:
    def __init__(self, collection, train_config):
        self._col_name = collection
        self._train_cfg = train_config
    def transform(self, tuner, append_explorables='all', append_static=True):
        """:param tuner: patm.tuning.Tuner"""
        self._ideology = tuner._ideology_information_flag
        self._prefix_label = tuner._prefix
        self._append_explorables = append_explorables
        self._append_static = append_static
        self._static = []
        self._explorable = []
        return self
    def new_plsa(self):
        self._ideology = False
        self._prefix_label = 'plsa'
        self._append_explorables = 'all'
        self._append_static = True
        self._static = []
        self._explorable = []
        return self
    def new_lda(self):
        self._ideology = False
        self._prefix_label = 'lda'
        self._append_explorables = 'all'
        self._append_static = True
        self._static = []
        self._explorable = []
        return self
    def new(self, enable_ideology_labels=False, prefix_label='', append_explorables='all', append_static=True):
        self._ideology = enable_ideology_labels
        self._prefix_label = prefix_label
        self._append_explorables = append_explorables
        self._append_static = append_static
        self._static = []
        self._explorable = []
        return self
    def add_static(self, name, value):
        self._static.append((name, value))
        return self
    def add_explorable(self, name, value):
        self._explorable.append((name, value))
        return self
    def add_statics(self, list_of_static_parameters_tuples):
        self._static.extend(list_of_static_parameters_tuples)
        return self
    def add_explorables(self, list_of_explorable_parameters_tuples):
        self._explorable.extend(list_of_explorable_parameters_tuples)
        return self
    def build(self):
        return Tuner(self._col_name, self._train_cfg, self._static, self._explorable, enable_ideology_labels=self._ideology,
                     prefix_label=self._prefix_label, append_explorables=self._append_explorables, append_static=self._append_static)

builders = {}
def get_tuner_builder(collection, train_config):
    """

    :param collection:
    :param train_config:
    rreturn: patm.modeling.tuning.TunerBuilder
    """
    _ = '.'.join([collection, train_config])
    if _ not in builders:
        builders[_] = TunerBuilder(collection, train_config)
    return builders[_]


if __name__ == '__main__':
    tuner = Tuner('articles', prefix_label='gg')
    from parameter_building import tuner_definition_builder as tdb

    d2 = tdb.initialize().collection_passes(100).nb_topics([20, 40]).document_passes(1).background_topics_pct(0.1).ideology_class_weight(5).\
        sparse_phi().deactivate(10).kind(['quadratic', 'cubic']).start(-1).end([-10, -20, -30]).\
        sparse_theta().deactivate(10).kind(['quadratic', 'cubic']).start([-1, -2, -3]).end(-10).build()

    tuner.regularizer_defs = {'smooth-phi': {'tau': 1.0},
                              'smooth-theta': {'tau': 1.0},
                              'sparse-theta': {'alpha_iter': 1}}

    # tuner.regularizer_defs = {'smooth-phi': {'tau': 1.0},
    #                           'smooth-theta': {'tau': 1.0},
    #                           'sparse-theta': {'alpha_iter': 'linear_1_4'}}


    tuner.tune(d2)