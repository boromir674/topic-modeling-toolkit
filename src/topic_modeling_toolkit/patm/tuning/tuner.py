import warnings
from collections import OrderedDict, Counter
import attr
from functools import reduce
import pprint
from types import FunctionType
from tqdm import tqdm
from .parameters import ParameterGrid
from ..modeling import TrainerFactory, Experiment
from ..definitions import DEFAULT_CLASS_NAME, IDEOLOGY_CLASS_NAME  # this is the name of the default modality. it is irrelevant to class lebels or document lcassification


@attr.s
class Versioning(object):
    objects = attr.ib(init=True, default=[], converter=Counter, repr=True, cmp=True)
    _max_digits_version = attr.ib(init=True, default=2, converter=int, repr=True, cmp=True)
    _joiner = '_v'

    def __call__(self, data):
        self.objects[data] += 1
        if self.objects.get(data, 0) > 1:
            return '{}{}{}'.format(data, self._joiner, self._iter_prepend(self.objects[data]-1))
        return data

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits < self._max_digits_version:
            return '{}{}'.format((self._max_digits_version - nb_digits) * '0', int_num)
        if nb_digits == self._max_digits_version:
            return str(int_num)
        raise RuntimeError("More than 100 items are versioned. (but max_digit_length=2)")


@attr.s
class Tuner(object):
    dataset = attr.ib(init=True, repr=True)
    scores = attr.ib(init=True, default={
        'perplexity': 'per',
        'sparsity-phi-@dc': 'sppd',
        'sparsity-phi-@ic': 'sppi',
        'sparsity-theta': 'spt',
        # 'topic-kernel-0.25': 'tk25',
        'topic-kernel-0.60': 'tk60',
        'topic-kernel-0.80': 'tk80',
        'top-tokens-10': 'top10',
        'top-tokens-100': 'top100',
        'background-tokens-ratio-0.3': 'btr3',
        'background-tokens-ratio-0.2': 'btr2'
    })
    _training_parameters = attr.ib(init=True, default={}, converter=dict, repr=True)
    _reg_specs = attr.ib(init=True, default={}, converter=dict, repr=True)
    grid_searcher = attr.ib(init=True, default=None, repr=True)
    version = attr.ib(init=True, factory=Versioning, repr=True)

    _labeler = attr.ib(init=False, default=None)
    trainer = attr.ib(init=False, default=attr.Factory(lambda self: TrainerFactory().create_trainer(self.dataset, exploit_ideology_labels=True, force_new_batches=False), takes_self=True))
    experiment = attr.ib(init=False, default=attr.Factory(lambda self: Experiment(self.dataset), takes_self=True))

    def __attrs_post_init__(self):
        self.trainer.register(self.experiment)

    def __getitem__(self, item):
        if item == 'training':
            return self._training_parameters
        if item == 'regularization':
            return self._reg_specs
        raise KeyError

    @property
    def parameter_names(self):
        return self._training_parameters.parameter_names + self._reg_specs.parameter_names

    @property
    def constants(self):
        return self._training_parameters.steady + self._reg_specs.steady

    @property
    def explorables(self):
        return self._training_parameters.explorable + self._reg_specs.explorable

    @property
    def training_parameters(self):
        """The mixture of steady parameters and prameters to tune on"""
        return self._training_parameters

    @training_parameters.setter
    def training_parameters(self, training_parameters):
        """Provide a dict with the mixture of steady parameters and prameters to tune on"""
        self._training_parameters = ParametersMixture(training_parameters)

    @property
    def regularization_specs(self):
        """The specifications according to which regularization components should be activated, initialized and potentially evolved (see tau trajectory) during training"""
        return self._reg_specs

    @regularization_specs.setter
    def regularization_specs(self, regularization_specs):
        self._reg_specs = RegularizationSpecifications(regularization_specs)

    @property
    def current_reg_specs(self):
        return {reg_type:
                    {param_name:
                         self._val('{}.{}'.format(reg_type, param_name))
                     for param_name in params_mixture.parameter_names} for reg_type, params_mixture in self._reg_specs}

    def _set_verbosity_level(self, input_verbose):
        try:
            self._vb = int(input_verbose)
            if self._vb < 0:
                self._vb = 0
            elif 5 < self._vb:
                self._vb = 5
        except ValueError:
            self._vb = 3

    def tune(self, *args, **kwargs):
        self._set_verbosity_level(kwargs.get('verbose', 3))
        
        if args:
            if len(args) > 0:
                self.training_parameters = args[0]
            if len(args) > 1:
                self.regularization_specs = args[1]

        self._labeler = LabelingDefinition.from_tuner(self, prefix=kwargs.get('prefix_label', ''), labeling_params=kwargs.get('labeling_params', False),
                                                      append_static=kwargs.get('append_static', False), append_explorable=kwargs.get('append_explorables', True),
                                                      preserve_order=kwargs.get('preserve_order', True),
                                                      parameter_set=kwargs.get('parameter_set', 'training'))

        self.grid_searcher = ParameterGrid(self._training_parameters.parameter_spans + [span for _, reg_params_mixture in self._reg_specs for span in reg_params_mixture.parameter_spans])
        
        if 1 < self._vb:
            print('Taking {} samples for grid-search'.format(len(self.grid_searcher)))
        # if kwargs.get('force_overwrite', True):
        print('Overwritting any existing results and phi matrices found')
        if self._vb:
            print('Tuning..')
            generator = tqdm(self.grid_searcher, total=len(self.grid_searcher), unit='model')
        else:
            generator = iter(self.grid_searcher)
        
        for i, self.parameter_vector in enumerate(generator):
            self._cur_label = self.version(self._labeler(self.parameter_vector))
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                # warnings.simplefilter("always")
                tm, specs = self._model()
                # assert len(w) == 1
                # assert issubclass(w[-1].category, DeprecationWarning)
                # assert "The value of 'probability_mass_threshold' parameter should be set to 0.5 or higher" == str(w[-1].message)
                # Trigger a warning.
                # Verify some things
            tqdm.write(self._cur_label)
            tqdm.write("Background: [{}]".format(', '.join([x for x in tm.background_topics])))
            tqdm.write("Domain: [{}]".format(', '.join([x for x in tm.domain_topics])))
            
            if 4 < self._vb:
                tqdm.write(pprint.pformat({k: dict(v, **{k: v for k, v in {
                    'target topics': self._topics_str(tm.get_reg_obj(tm.get_reg_name(k)).topic_names, tm.domain_topics, tm.background_topics),
                    'mods': getattr(tm.get_reg_obj(tm.get_reg_name(k)), 'class_ids', None)}.items()}) for k, v in self.current_reg_specs.items()}))
            if 3 < self._vb:
                tqdm.write(pprint.pformat(tm.modalities_dictionary))
            self.experiment.init_empty_trackables(tm)
            self.trainer.train(tm, specs, cache_theta=kwargs.get('cache_theta', True))
            self.experiment.save_experiment(save_phi=True)

    def _topics_str(self, topics, domain, background):
        if topics == domain:
            return 'domain'
        if topics == background:
            return 'background'
        return '[{}]'.format(', '.join(topics))

    def _model(self):
        tm = self.trainer.model_factory.construct_model(self._cur_label, self._val('nb_topics'),
                                                        self._val('collection_passes'),
                                                        self._val('document_passes'),
                                                        self._val('background_topics_pct'),
                                                        {k: v for k, v in
                                                         {DEFAULT_CLASS_NAME: self._val('default_class_weight'),
                                                          IDEOLOGY_CLASS_NAME: self._val(
                                                              'ideology_class_weight')}.items() if v},
                                                        self.scores,
                                                        self._reg_specs.types,
                                                        reg_settings=self.current_reg_specs)  # a dictionary mapping reg_types to reg_specs
        tr_specs = self.trainer.model_factory.create_train_specs(self._val('collection_passes'))
        return tm, tr_specs

    def _val(self, parameter_name):
        return self.extract(self.parameter_vector, parameter_name.replace('_', '-'))

    def extract(self, parameters_vector, parameter_name):
        r = parameter_name.split('.')
        if len(r) == 1:
            return self._training_parameters.extract(parameters_vector, parameter_name)
        elif len(r) == 2:
            return self._reg_specs.extract(parameters_vector, r[0], r[1])
        else:
            raise ValueError("Either input a training parameter such as 'collection_passes', 'nb_topics', 'ideology_class_weight' or a regularizer's parameter in format such as 'sparse-phi.tau', 'label-regularization-phi-dom-def.tau'")

    def __format_reg(self, reg_specs, reg_type):
        if 'name' in reg_specs[reg_type]:
            return reg_type, reg_specs[reg_type].pop('name')
        return reg_type


############## PARAMETERS MIXTURE ##############


@attr.s
class RegularizationSpecifications(object):
    reg_specs = attr.ib(init=True, converter=lambda x: OrderedDict(
        [(reg_type, ParametersMixture([(param_name, value) for param_name, value in reg_specs])) for reg_type, reg_specs in x]), repr=True, cmp=True)
    parameter_spans = attr.ib(init=False, default=attr.Factory(
        lambda self: [span for reg_type, reg_specs in self.reg_specs.items() for span in reg_specs.parameter_spans],
        takes_self=True))
    parameter_names = attr.ib(init=False,
                              default=attr.Factory(lambda self: ['{}.{}'.format(reg_type, param_name) for reg_type, mixture in self.reg_specs.items() for param_name in mixture.parameter_names], takes_self=True))
    steady = attr.ib(init=False, default=attr.Factory(
        lambda self: ['{}.{}'.format(reg_type, param_name) for reg_type, mixture in self.reg_specs.items() for param_name in mixture.steady],
        takes_self=True))
    explorable = attr.ib(init=False, default=attr.Factory(
        lambda self: ['{}.{}'.format(reg_type, param_name) for reg_type, mixture in self.reg_specs.items() for param_name in mixture.explorable],
        takes_self=True))

    nb_combinations = attr.ib(init=False, default=attr.Factory(lambda self: reduce(lambda i, j: i * j, [v.nb_combinations for v in self.reg_specs.values()]), takes_self=True))

    def __getitem__(self, item):
        return self.reg_specs[item]

    def __iter__(self):
        return ((reg_type, reg_params_mixture) for reg_type, reg_params_mixture in self.reg_specs.items())

    def extract(self, parameter_vector, reg_name, reg_param):
        parameter_vector = list(parameter_vector[-len(self.parameter_spans):])
        if reg_name not in self.reg_specs:
            raise KeyError
        s = 0
        for k, v in self.reg_specs.items():
            if k == reg_name:
                break
            s += v.length
        return self.reg_specs[reg_name].extract(parameter_vector[s:], reg_param)

    @property
    def types(self):
        return list(self.reg_specs.keys())

def _conv(value):
    if type(value) != list:
        return [value]
    return value


def _build(tuple_list):
    if len(tuple_list) != len(set([x[0] for x in tuple_list])):
        raise ValueError("Input tuples should behave like a dict (unique elements as each 1st element)")
    return OrderedDict([(x[0], _conv(x[1])) for x in tuple_list])

@attr.s
class ParametersMixture(object):
    """An OrderedDict with keys parameter names and keys either a single object to initialize with or a list of objects intending to model a span/grid of values for grid-search"""
    _data_hash = attr.ib(init=True, converter=_build)

    length = attr.ib(init=False, default=attr.Factory(lambda self: len(self._data_hash), takes_self=True))
    parameter_names = attr.ib(init=False, default=attr.Factory(lambda self: list(self._data_hash.keys()), takes_self=True))
    nb_combinations = attr.ib(init=False, default=attr.Factory(lambda self: reduce(lambda i,j: i*j, [len(v) for v in self._data_hash.values()]), takes_self=True))
    steady = attr.ib(init=False, default=attr.Factory(lambda self: [name for name, assumables_values in self._data_hash.items() if len(assumables_values) == 1], takes_self=True))
    explorable = attr.ib(init=False, default=attr.Factory(lambda self: [name for name, assumables_values in self._data_hash.items() if len(assumables_values) > 1], takes_self=True))
    parameter_spans = attr.ib(init=False, default=attr.Factory(lambda self: [assumable_values for assumable_values in self._data_hash.values()], takes_self=True))

    def __contains__(self, item):
        return item in self._data_hash

    def __len__(self):
        return len(self._data_hash)

    def __getitem__(self, item):
        return self._data_hash[item]

    def __iter__(self):
        return ((k,v) for k,v in self._data_hash.items())

    def extract(self, parameter_vector, parameter):
        return parameter_vector[self.parameter_names.index(parameter)]

    @classmethod
    def from_regularization_settings(cls, reg_settings):
        return ParametersMixture([(k,v) for k, v in reg_settings])

##############  LABELING ##############

def _check_extractor(self, attr_attribute_obj, input_value):
    if not hasattr(input_value, '__call__'):
        raise ValueError("A callable is required")
    if input_value.__code__.co_argcount != 2:
        raise ValueError("Callable should accept exactly 2 arguments. First is a parameter vector (list) and second a parameter name")


@attr.s
class LabelingDefinition(object):
    parameters = attr.ib(init=True, converter=list, repr=True, cmp=True)
    extractor = attr.ib(init=True, validator=_check_extractor)
    _prefix = attr.ib(init=True, default='') #, converter=str)

    def __call__(self, parameter_vector):
        return '_'.join(x for x in [self._prefix] + [self._conv(self.extractor(parameter_vector, param_name)) for param_name in self.parameters] if x)

    def _conv(self, v):
        try:
            v1 = float(v)
            if v1 >= 1e4:
                return "{:.2}".format(v1)
            if int(v1) == v1:
                return str(int(v1))
            return str(v)
        except ValueError:
            return str(v)

    @classmethod
    def from_training_parameters(cls, training_parameters, prefix='', labeling_params=None, append_static=False, append_explorable=True, preserve_order=False):
        if labeling_params:
            if not type(labeling_params) == list:
                raise ValueError("If given the labeling_params argument should be a list")
        else:
            if type(append_static) == list:
                labeling_params = append_static
            elif append_static:
                labeling_params = training_parameters.steady.copy()
            else:
                labeling_params = []
            if type(append_explorable) == list:
                labeling_params.extend(append_explorable)
            elif append_explorable:
                labeling_params.extend(training_parameters.explorable)
        if preserve_order:
            labeling_params = [x for x in training_parameters.parameter_names if x in labeling_params]
        return LabelingDefinition(labeling_params, lambda vector, param: training_parameters.extract(vector, param), prefix)

    @classmethod
    def from_tuner(cls, tuner, prefix='', labeling_params=None, append_static=False, append_explorable=True, preserve_order=False, parameter_set='training|regularization'):
        return LabelingDefinition(cls.select(tuner, labeling_params=labeling_params, append_static=append_static, append_explorable=append_explorable, preserve_order=preserve_order, parameter_set=parameter_set),
                                  lambda vector, param: tuner.extract(vector, param),
                                  prefix)

    @classmethod
    def select(cls, tuner, **kwargs):
        labeling_params = []
        if kwargs.get('labeling_params', None):
            if not type(kwargs.get('labeling_params', None)) == list:
                raise ValueError("If given, the labeling_params keyword-argument should be a list")
            labeling_params = kwargs['labeling_params']
        else:
            if type(kwargs.get('append_static', False)) == list:
                labeling_params.extend(kwargs['append_static'])
            elif kwargs.get('append_static', False):
                labeling_params.extend([x for el in kwargs.get('parameter_set', 'training|regularization').split('|') for x in tuner[el].steady])
            if type(kwargs.get('append_explorable', False)) == list:
                labeling_params.extend(kwargs['append_explorable'])
            elif kwargs.get('append_explorable', False):
                labeling_params.extend([x for el in kwargs.get('parameter_set', 'training|regularization').split('|') for x in tuner[el].explorable])
        if kwargs.get('preserve_order', False):
            # labeling_params = [x for x in [y for el in kwargs.get('parameter_set', 'training|regularization').split('|') for y in tuner[el].parameter_names] if x in labeling_params]
            labeling_params = [x for x in tuner.parameter_names if x in labeling_params]
        return labeling_params
