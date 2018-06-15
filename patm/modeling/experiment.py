import os
import cPickle as pickle
import json

from .regularizers import parameter_name2encoder
from .persistence import ResultsWL, ModelWL


class Experiment:
    """
    This class encapsulates experimental related activities such as tracking of various quantities during model training
    and also has capabilities of persisting these tracked quantities as experimental results.\n
    Experimental results include:
    - number of topics: the number of topics inffered
    - document passes: the inner iterations over each document. Phi matrix gets updated 'document_passes' times during one path through the whole collection
    - root dir: the base directory of the colletion, on which experiments were conducted
    - model label: the unique identifier of the model used for the experiments
    - trackable metrics
    - regularization parameters
    - evaluators
    """

    def __init__(self, root_dir):
        """
        Encapsulates experimentation by doing topic modeling on a 'collection' in the given root_dir. A 'collection' is a proccessed document collection into BoW format and possibly split into 'train' and 'test' splits
        """
        self._dir = root_dir
        self._loaded_dictionary = None # artm.Dictionary object. Data population happens uppon artm.Artm object creation in model_factory; dictionary.load(bin_dict_path) is called there
        self._topic_model = None
        self.collection_passes = []
        self.specs_instances = []
        self.trackables = {}
        self.reg_params = []
        self.model_params = {'nb_topics': [], 'document_passes': []}

        self.train_results_handler = ResultsWL(self, 'train')
        self.phi_matrix_handler = ModelWL(self, 'train')

    @property
    def topic_model(self):
        return self._topic_model

    def set_topic_model(self, topic_model, empty_trackables=False):
        self._topic_model = topic_model
        if empty_trackables:
            self.trackables = {key: {inner_k: [] for inner_k in self._topic_model.evaluators[key].attributes} for key in self._topic_model.evaluators.keys()}

    @property
    def dictionary(self):
        return self._loaded_dictionary

    def set_dictionary(self, artm_dictionary):
        self._loaded_dictionary = artm_dictionary

    def update(self, model, specs):
        self.collection_passes.append(specs['collection_passes']) # iterations
        self.specs_instances.append(specs)
        self.model_params['nb_topics'].append(tuple((specs['collection_passes'], model.num_topics)))
        self.model_params['document_passes'].append(tuple((specs['collection_passes'], model.num_document_passes)))
        self.reg_params.append(tuple((specs['collection_passes'], self._topic_model.get_regs_param_dict())))

        for evaluator_type, evaluator_instance in self._topic_model.evaluators.items():
            current_eval = evaluator_instance.evaluate(model)
            for inner_k, value in current_eval.items():
                try:
                    self.trackables[evaluator_type][inner_k] = value
                except RuntimeError as e:
                    print e, '\n', type(value)
                    try:
                        print len(value)
                    except TypeError:
                        print 'does not have __len__ implemented'
                    raise EvaluationOutputLoadingException("Could not assign the value of type '{}' with key '{}' as an item in self.trackables'".format(type(value), inner_k))

    @property
    def current_root_dir(self):
        return self._dir

    def get_results(self):
        return {
            'collection_passes': self.collection_passes,  # eg [20, 20, 40, 100]
            'trackables': self.trackables,  # TODO try list of tuples [('perplexity'), dict), ..]
            'root_dir': self.current_root_dir,  # eg /data/blah/
            'model_label': self.topic_model.label,
            'model_parameters': self.model_params,
            'reg_parameters': self.reg_params,
            'evaluators': sorted(map(lambda x: x[0], x[1].name, self._topic_model.evaluators.items()), key=lambda x: x[0])
        }

    def save_experiment(self, save_phi=True):
        """
        Dumps the dictionary-type accumulated results with the given file name. The file is saved in the directory specified by the latest train specifications (TrainSpecs).\n
        """
        if not self.collection_passes:
            raise DidNotReceiveTrainSignalException('Model probably hasn\'t been fitted since len(self.collection_passes) = {}, len(self.specs_instances) = {}'.format(len(self.collection_passes), len(self.specs_instances)))
        # asserts that the number of observations recorded per tracked metric variables is equal to the number of "collections pass"; training iterations over the document dataset
        # artm.ARTM.scores satisfy this
        assert all(map(lambda x: len(x) == sum(self.collection_passes), [values_list for eval2scoresdict in self.trackables.values() for values_list in eval2scoresdict.values()]))
        # the above will fail when metrics outside the artm library start to get tracked, because these will be able to only capture the last state of the metric trajectory due to fitting by "junks
        self.train_results_handler.save(self.topic_model.label)
        if save_phi:
            self.phi_matrix_handler.save(self.topic_model.label)

    def load_experiment(self, model_label):
        """
        Given a unigue model label, restores the state of the experiment from disk. Loads all tracked values of the experimental results
        and the state of the TopicModel inferred so far: namely the phi p_wt matrix.
        In details loads settings:
        - doc collection fit iteration chunks
        - eval metrics/measures trackes per iteration
        - regularization parameters
        - document passes
        \n
        :param str model_label: a unigue identifier of a topic model
        :return: the latest train specification used in the experiment
        :rtype: patm.modeling.topic_model.TrainSpecs
        """
        results = self.train_results_handler.load(model_label)
        assert model_label == results['model_label']
        self.collection_passes = results['collection_passes']
        self.specs_instances = []
        self.trackables = results['trackables']
        self.reg_params = results['reg_parameters']
        self.model_params = results['model_parameters']
        my_tm, train_specs = self.phi_matrix_handler.load(name, results=results)
        self.set_topic_model(my_tm, empty_trackables=False)
        return train_specs

        # get_model_factory(self._loaded_dictionary).create_model_with_phi_from_disk()

    # def set_parameters(self, expressions_list):
    #     """
    #     Allows setting new values to already existing
    #     :param expressions_list:
    #     :return:
    #     """
    #     components_lists = [expression.split('.') for expression in expressions_list]
    #     for reg_name, param, value in components_lists:
    #         self._topic_model.set_parameter(reg_name, param, value)

class EvaluationOutputLoadingException(Exception):
    def __init__(self, msg):
        super(EvaluationOutputLoadingException, self).__init__(msg)


class DidNotReceiveTrainSignalException(Exception):
    def __init__(self, msg):
        super(DidNotReceiveTrainSignalException, self).__init__(msg)
