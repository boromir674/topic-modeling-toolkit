from collections import Counter, OrderedDict
from functools import reduce
from .definitions import SCALE_PLACEMENT, DISCREETIZATION
import attr

import logging
logger = logging.getLogger(__name__)

from types import MethodType


def distr(ds_scheme, psm):
    outlet_id2name = OrderedDict([(_id, name) for name, _id in psm.scale.items()])
    datapoint_labels = [ds_scheme.poster_name2ideology_label[outlet_id2name[x]] for x in psm.datapoint_ids]
    c = Counter(datapoint_labels)
    n = sum(c.values())
    return [c[class_name] / float(n) for class_name, _ in ds_scheme]

def evolve(self, class_names, datapoint_ids, vectors_length, pool_size, prob=0.2, max_generation=100):
    self.population.evolve(datapoint_ids, vectors_length, pool_size, prob=prob, max_generation=max_generation)
    return DiscreetizationScheme.from_design(self.population.pool[0], [(k,v) for k,v in self.scale.items()], class_names)


class PoliticalSpectrumManager(object):
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = PoliticalSpectrum(SCALE_PLACEMENT, DISCREETIZATION)
            cls._instance.population = Population(cls._instance)
            cls._instance.balance_frequencies = MethodType(evolve, cls._instance)
        return cls._instance


def _correct_order(instance, attribute, converted_value):
    instance._schemes = {k: OrderedDict([(class_label, [_ for _ in instance.scale if _ in outlet_names]) for class_label, outlet_names in scheme.items()])
                         for k, scheme in converted_value.items()}


@attr.s
class PoliticalSpectrum(object):
    scale = attr.ib(init=True, converter=lambda x: OrderedDict(x), repr=True)
    _schemes = attr.ib(init=True, converter=lambda x: {name: DiscreetizationScheme(scheme) for name, scheme in x.items()})

    # _schemes = attr.ib(init=True, converter=lambda x: {scheme_name:
    #                                                        OrderedDict([(class_name.lower().replace(' ', '_'), outlet_names_list) for class_name, outlet_names_list in discretization_design])
    #                                                    for scheme_name, discretization_design in x.items()},
    #                    validator=_correct_order, repr=True)
    _cur = attr.ib(init=True, default='legacy-scheme', converter=str, repr=True)
    datapoint_ids = attr.ib(init=True, default=[], repr=False)
    _outlet_id2outlet_name = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict((v, k) for k, v in self.scale.items()), takes_self=True), repr=False)

    # _label_id2outlet_names = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict([(bin_name.lower().replace(' ', '_'), outlet_names_list) for bin_name, outlet_names_list in
    #      self._schemes[self._cur].values()]), takes_self=True), repr=False)
    # _outlet_id2outlet_name = attr.ib(init=False, default=attr.Factory(lambda self: {poster_id: poster_name for ide_bin_dict in self._schemes[self._cur].values() for
    #                                 poster_name, poster_id in ide_bin_dict.items()}, takes_self=True), repr=False)
    # _label_id2outlet_dict = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict([('_'.join(ide.split(' ')),
    #                                                                                             OrderedDict(sorted([(outlet, outlet_id) for outlet, outlet_id in
    #                                                                self._schemes[self._cur][ide].items()],
    #                                                               key=lambda x: x[0]))) for ide in
    #                                                                                            self._schemes[self._cur].keys()]), takes_self=True), repr=False)
    def _key(self, scheme):
        return '-'.join(str(class_name) for class_name, _ in scheme)

    def __getitem__(self, item):
        if item not in self._schemes:
            raise KeyError(
                "The schemes implemented are [{}]. Requested '{}' instead.".format(', '.join(self._schemes.keys()), item))
        return self._schemes[item]

    def __iter__(self):
        return ((k, v) for k, v in self._schemes.items())

    @property
    def discreetization_scheme(self):
        return self._schemes[self._cur]

    @discreetization_scheme.setter
    def discreetization_scheme(self, scheme):
        if type(scheme) == str:
            if scheme not in self._schemes:
                raise KeyError("The schemes implemented are [{}]. Requested '{}' instead.".format(', '.join(self._schemes.keys()), scheme))
            self._cur = scheme

        elif type(scheme) == list:
            if type(scheme[0]) == str:
                k = scheme[0]
                scheme = scheme[1]
            else:
                k = self._key(scheme)
            self._schemes[k] = DiscreetizationScheme(scheme)
            self._cur = k
            logger.info("Registered new discreetization scheme '{}' with doc classes [{}].".format(k, ', '.join(
                class_name for class_name, _ in scheme)))
            if self.datapoint_ids:
                logger.info("Classes' distribution: [{}]".format(
                    ', '.join(['{:.2f}'.format(x) for x in distr(self._schemes[k], self)])))
        elif type(scheme) == DiscreetizationScheme:
            k = self._key(scheme)
            self._schemes[k] = DiscreetizationScheme(scheme)
            self._cur = k

            logger.info("Registered new discreetization scheme '{}' with doc classes [{}].".format(k, ', '.join(class_name for class_name, _ in scheme)))
            if self.datapoint_ids:
                logger.info("Classes' distribution: [{}]".format(', '.join(['{:.2f}'.format(x) for x in distr(self._schemes[k], self)])))
        else:
            raise ValueError("Input should be either a string or a DiscreetizationScheme or a [str, DiscreetizationScheme] list (1st element is the name/key)")

    def distribution(self, scheme):
        return distr(scheme, self)

    @property
    def class_distribution(self):
        return distr(self._schemes[self._cur], self)

    @property
    def poster_id2ideology_label(self):
        return OrderedDict([(self.scale[name], class_label) for class_label, outlet_names in self._schemes[self._cur] for name in outlet_names])

    @property
    def class_names(self):
        """The normalized class names matching the discrete bins applied on the 10-point scale of ideological consistency"""
        return self._schemes[self._cur].class_names

    # def optimize_class_distribution(self, outletnames, class_names, pool_size=50, max_iterations=100, gene_mutation_prob=0.3):
    #     self.population = Population(self)
    #     self.population.evolve(outletnames, len(class_names)-1, pool_size, prob=gene_mutation_prob, max_generation=max_iterations)
    #
    #     if len(set(outletnames)) != len(self.scale):
    #         logger.warning("There is a mismatch between the distinct number of outlets supplied ({}) and the number of items in (global and constant) ideological scale ({})".format(len(set(outletnames)), len(self.scale)))
    #     ds = DiscreetizationScheme.from_design(self.population.pool[0], list(self.scale.items()), class_names)
        # self._schemes['evolved'] = ds


@attr.s(cmp=True, repr=True, slots=True)
class DiscreetizationScheme(object):
    _bins = attr.ib(init=True, converter=lambda x: OrderedDict([(class_name, outlet_names_list) for class_name, outlet_names_list in x]), repr=True, cmp=True)
    poster_name2ideology_label = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict([(name, class_label) for class_label, outlet_names in self._bins.items() for name in outlet_names]), takes_self=True), repr=False)
    # bins = attr.ib(init=True, converter=list, cmp=True, repr=True)
    # _class_names = attr.ib(init=True, default=attr.Factory(lambda self: ['my_class_{}'.format(x) for x in range(len(self.bins))], takes_self=True),
    #                       converter=lambda x: [_.lower().replace(' ', '_') for _ in x], cmp=True, repr=True)
    def __iter__(self):
        for class_name, items in self._bins.items():
            yield class_name, items

    @property
    def class_names(self):
        return list(self._bins.keys())

    @class_names.setter
    def class_names(self, class_names):
        if len(class_names) != len(self._bins):
            raise RuntimeError("Please give equal number of class names ({} given) as the number of defined bins ({})".format(len(class_names), len(self._bins)))
        self._bins = OrderedDict([(class_name, outlet_list) for class_name, outlet_list in zip(class_names, self._bins.items())])

    @classmethod
    def from_design(cls, design, scale, class_names=None):
        """
        :param design:
        :param list of tuples scale:
        :param class_names:
        :return:
        """
        if not class_names:
            class_names = ['bin_{}'.format(x) for x in range(len(design) + 1)]
        return DiscreetizationScheme([(k,v) for k,v in zip(class_names, list(Bins.from_design(design, scale)))])

    # @property
    # def poster_name2ideology_label(self):
    #     return OrderedDict([(name, class_label) for class_label, outlet_names in self._bins.items() for name in outlet_names])

    # @classmethod
    # def from_tuples(cls, data):
    #     return DiscreetizationScheme([_[1] for _ in data], [_[0] for _ in data])
        # a, b = (list(x) for x in zip(*[[doc, label] for doc, label in zip(self.corpus, self.outlet_ids) if doc]))
        # return DiscreetizationScheme(*reversed([list(x) for x in zip(*data)]))


def _check_nb_bins(instance, attribute, value):
    if not 0 < len(value) <= 100:
        raise ValueError("Resulted in {} bins but they should be in [1,100]".format(len(value)))
    if sum(len(x) for x in value) != len(reduce(lambda x,y: x.union(y), [set(_) for _ in value])):
        raise ValueError("Found same item in multiple bins")


@attr.s(cmp=True, repr=True, str=True, slots=True, frozen=True)
class Bins(object):
    bins = attr.ib(init=True, cmp=True, repr=True, validator=_check_nb_bins)
    # __max_el_length = attr.ib(init=False, default=attr.Factory(lambda self: max([len(x) for outlet_names in self.bins for x in outlet_names]), takes_self), repr=False, cmp=False)

    def __getitem__(self, item):
        return self.bins[item]

    @classmethod
    def from_design(cls, design, scale_placement):
        """
        :param design:
        :param list of tuples scale_placement: the ordering of outlets from liberal to conservative that the design bins (discreetizes to classes)
        :return:
        """
        return Bins([[scale_placement[i][0] for i in a_range] for a_range in BinDesign(list(design)).ranges(len(scale_placement))])

    def __str__(self):
        return '\n'.join(' '.join(self.__str_els(b, i) for b in self.bins) for i in range(max(len(x) for x in self.bins)))
        # for i in range(max(len(x) for x in self.bins)):
        #     line = ' '.join(self.__str_els(b, i) for b in self.bins)

    def __str_els(self, bin, index):
        if index < len(bin):
            return bin[index] + ' ' * (max(len(x) for x in bin) - len(bin[index]))
        return ' ' * self.__max_el_length


def _check_design(instance, attribute, value):
    if not 0 < len(value) < 100:
        raise ValueError("Resulted in {} bins but they should be in [1,100]".format(len(value)+1))
    for i in range(1, len(value)):
        if value[i] <= value[i - 1]:
            raise ValueError("Invalid design list. Each element should be greater than the previous one; prev: {}, current: {}".format(value[i - 1], value[i]))


@attr.s(cmp=True, repr=True, str=True)
class BinDesign(object):
    seps = attr.ib(init=True, converter=list, validator=_check_design, cmp=True, repr=True)

    def ranges(self, nb_elements):
        if self.seps[-1] >= nb_elements:
            raise ValueError("Last bin starts from index {}, but requested to build range indices for {} elements".format(self.seps[-1], nb_elements))
        yield range(self.seps[0])
        for i, item in enumerate(self.seps[1:]):
            yield range(self.seps[i], item)
        yield range(self.seps[-1], nb_elements)

    def __getitem__(self, item):
        return self.seps[item]
    def __len__(self):
        return len(self.seps)


import random
import numpy as np
from scipy.stats import entropy as scipy_entropy


@attr.s
class Population(object):
    psm = attr.ib(init=True)
    # doc_ids = attr.ib(init=True, converter=list)
    # outlet_id2name = attr.ib(init=False, default=attr.Factory(lambda self: OrderedDict([(_id, name) for name, _id in self.psm.scale.items()]), takes_self=True), repr=False)
    pool = attr.ib(init=False, repr=True, cmp=True)

    def create_random(self, vector_length, scale_length):
        inds = [_ for _ in range(1, scale_length)]
        res = []
        for i in range(vector_length):
            c = random.choice(inds)
            inds.remove(c)
            res.append(c)
        if res[-1] == scale_length:
            raise RuntimeError("Vector's last element should be maximum {}. Found {}".format(scale_length-1, res[-1]))
        return sorted(res, reverse=False)

    def distr(self, design):
        ds = DiscreetizationScheme.from_design(design, list(self.psm.scale.items()))
        outlet_id2name = OrderedDict([(_id, name) for name, _id in self.psm.scale.items()])
        datapoint_labels = [ds.poster_name2ideology_label[outlet_id2name[x]] for x in self.datapoint_ids]
        c = Counter(datapoint_labels)
        n = sum(c.values())
        return [c[class_name] / float(n) for class_name, _ in ds]

    def compute_fitness(self, design):
        """The smaller the better in our case!"""
        design.fitness = jensen_shannon_distance(self.distr(design), self.ideal)
        return design.fitness

    def init_random(self, pool_size, vector_length, elements, max_generation=100, convergence=(1, 10)):
        self.pool = [BinDesign(self.create_random(vector_length, elements)) for _ in range(pool_size)]
        self.ideal = [float(1)/(vector_length+1)] * (vector_length + 1)
        self.sorted = False
        self._generation_counter = 0
        self._max_generation = max_generation
        self._convergence = convergence

    def selection(self):
        self._inds = [x for x in range(len(self.pool))]

    def operators(self, nb_elements, prob=0.2):
        # MUTATE
        self._new = [_f for _f in [self.mutate(design, nb_elements, prob=prob) for design in self.pool] if _f]

    def mutate(self, design, elements, prob=0.2):
        # available = [x for x in range(1, elements) if x not in design]
        _ = BinDesign([x for x in self._gen_genes(design, elements, prob=prob)])
        if _ == design:
            return None
        return _
        # res = []
        # for i in range(len(res)):
        #     if random.random() < prob:
        #         available = [x for x in range(1, elements) if (x not in design and x not in res)]
        #         assert available[-1] == elements - 1
        #         c = random.choice(available)
        #         res.append(c)
        # _ = BinDesign(sorted(res, reverse=False))
        # if _[-1] == elements:
        #     raise RuntimeError("Vector: {}, scale elements: {}".format(list(_), elements))
        # return _

    def replacement(self):
        self.pool = sorted(self.pool + self._new, key=lambda x: getattr(x, 'fitness', self.compute_fitness(x)))[:len(self.pool)]  # sors from smaller to bigger values so best to worst because smaller fitness value the better
        self._generation_counter += 1
        self.sorted = True

    def evolve(self, datapoint_ids, vectors_length, pool_size, prob=0.2, max_generation=100, convergence=(1, 10)):
        self.datapoint_ids = datapoint_ids
        self.psm.datapoint_ids = datapoint_ids
        scale_length = len(self.psm.scale)
        self.init_random(pool_size, vectors_length, scale_length, max_generation=max_generation, convergence=convergence)
        while not self.condition():
            self.selection()
            self.operators(scale_length, prob=prob)
            self.replacement()

    def condition(self):
        """Until convergence or max generation count. Returns True when evolution should stop"""
        return self._generation_counter >= self._max_generation

    def _gen_genes(self, design, elements, prob=0.5):
        if len(design) == 1:
            yield self._toss(design[0], self._new_index([x for x in range(1, elements)], design[0]), prob, random.random())
        elif len(design) == 2:
            prev =  self._toss(design[0], self._new_index([x for x in range(1, design[1])], design[0]), prob, random.random())
            yield prev
            yield self._toss(design[1], self._new_index([x for x in range(prev + 1, elements)], design[1]), prob, random.random())
        else:
            prev = self._toss(design[0], self._new_index([x for x in range(1, design[1])], design[0]), prob,
                              random.random())
            yield prev
            for i in range(1, len(design) - 1):
                _ = self._toss(design[i], self._new_index([x for x in range(prev+1, design[i+1])], design[i]), prob, random.random())
                yield _
                prev = _
            yield self._toss(design[-1], self._new_index([x for x in range(prev + 1, elements)], design[-1]), prob, random.random())

    def _new_index(self, indices_list, current_index):
        if len(indices_list) == 1:
            return current_index
        c = random.randint(1, len(indices_list)-1) - 1
        if c >= indices_list.index(current_index):
            return indices_list[c+1]
        return indices_list[c]

    def _toss(self, v1, v2, prob, toss):
        if toss < prob:
            return v2
        return v1


def jensen_shannon_distance(p, q):
    """Jenson-Shannon Distance between two probability distributions"""
    p = np.array(p)
    q = np.array(q)
    m = (p + q) / 2
    divergence = (scipy_entropy(p, m) + scipy_entropy(q, m)) / 2
    distance = np.sqrt(divergence)
    return distance
