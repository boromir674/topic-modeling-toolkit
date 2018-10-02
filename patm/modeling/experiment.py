from .model_factory import get_model_factory
from .persistence import ResultsWL, ModelWL


class Experiment:
    """
    This class encapsulates experimental related activities such as tracking of various quantities during model training
    and also has capabilities of persisting these tracked quantities as experimental results.\n
    Experimental results include:
    - number of topics: the number of topics inffered
    - document passes: the inner iterations over each document. Phi matrix gets updated 'document_passes' x 'nb_docs' times during one path through the whole collection
    - root dir: the base directory of the colletion, on which experiments were conducted
    - model label: the unique identifier of the model used for the experiments
    - trackable "evaluation" metrics
    - regularization parameters
    - _evaluator_name2definition
    """
    def __init__(self, root_dir, cooc_dict):
        """
        Encapsulates experimentation by doing topic modeling on a 'collection' in the given root_dir. A 'collection' is a proccessed document collection into BoW format and possibly split into 'train' and 'test' splits
        """
        self._dir = root_dir
        self.cooc_dict = cooc_dict
        self._loaded_dictionary = None # artm.Dictionary object. Data population happens uppon artm.Artm object creation in model_factory; dictionary.load(bin_dict_path) is called there
        self._topic_model = None
        self.collection_passes = []
        self.trackables = None
        self.reg_params = []
        self.model_params = {'nb_topics': [], 'document_passes': []}
        self.train_results_handler = ResultsWL(self, 'train')
        self.phi_matrix_handler = ModelWL(self, 'train')

    def init_empty_trackables(self, model):
        self._topic_model = model
        self.trackables = {self._topic_model.evaluator_definitions[self._topic_model.evaluator_names.index(evaluator_name)]: {inner_k: [] for inner_k in self._topic_model.get_evaluator(evaluator_name).attributes} for evaluator_name in self._topic_model.evaluator_names}
        self.collection_passes = []
        self.reg_params = []
        self.model_params = {'nb_topics': [], 'document_passes': []}

    @property
    def model_factory(self):
        return get_model_factory(self.dictionary, self.cooc_dict)

    @property
    def topic_model(self):
        return self._topic_model

    @property
    def dictionary(self):
        return self._loaded_dictionary

    def set_dictionary(self, artm_dictionary):
        self._loaded_dictionary = artm_dictionary

    # TODO refactor this; remove dubious exceptions
    def update(self, topic_model, span):
        self.collection_passes.append(span) # iterations
        self.model_params['nb_topics'].append(tuple((span, topic_model.nb_topics)))
        self.model_params['document_passes'].append(tuple((span, topic_model.document_passes)))
        self.reg_params.append(tuple((span, topic_model.get_regs_param_dict())))
        for evaluator_name, evaluator_definition in zip(topic_model.evaluator_names, topic_model.evaluator_definitions):
            current_eval = topic_model.get_evaluator(evaluator_name).evaluate(topic_model.artm_model)
            for eval_reportable, value in current_eval.items():
                try:
                    if type(value) == list:
                        self.trackables[evaluator_definition][eval_reportable].extend(value[-span:])  # append only the newly produced tracked values
                    else:
                        raise RuntimeError
                except RuntimeError as e:
                    print e, '\n', type(value)
                    try:
                        print len(value)
                        raise e
                    except TypeError as er:
                        print 'does not have __len__ implemented'
                        print er
                    raise EvaluationOutputLoadingException("Could not assign the value of type '{}' with key '{}' as an item in self.trackables'".format(type(value), eval_reportable))

    @property
    def current_root_dir(self):
        return self._dir

    def get_results(self):
        return {
            'collection_passes': self.collection_passes,  # eg [20, 20, 40, 100]
            'trackables': self.trackables,  # TODO try list of tuples [('perplexity'), dict), ..]
            'root_dir': self.current_root_dir,  # eg /data/blah/
            'model_label': self._topic_model.label,
            'model_parameters': self.model_params,  # nb_topics and document_passes
            'reg_parameters': self.reg_params,
            'eval_definition2eval_name': dict(zip(self._topic_model.evaluator_definitions, self._topic_model.evaluator_names)),
            'background_topics': self.topic_model.background_topics,
            'domain_topics': self.topic_model.domain_topics,
            'modalities': self.topic_model.modalities_dictionary
        }

    def save_experiment(self, save_phi=True):
        """
        Dumps the dictionary-type accumulated experimental results with the given file name. The file is saved in the directory specified by the latest train specifications (TrainSpecs).\n
        """
        if not self.collection_passes:
            raise DidNotReceiveTrainSignalException('Model probably hasn\'t been fitted since len(self.collection_passes) = {}'.format(len(self.collection_passes)))
        # asserts that the number of observations recorded per tracked metric variables is equal to the number of "collections pass"; training iterations over the document dataset
        # artm.ARTM.scores satisfy this
        # print 'Saving model \'{}\', train set iterations: {}'.format(self.topic_model.label, self.collection_passes)
        assert all(map(lambda x: len(x) == sum(self.collection_passes), [values_list for eval2scoresdict in self.trackables.values() for values_list in eval2scoresdict.values()]))
        # the above will fail when metrics outside the artm library start to get tracked, because these will be able to only capture the last state of the metric trajectory due to fitting by "junks
        self.train_results_handler.save(self._topic_model.label)
        if save_phi:
            self.phi_matrix_handler.save(self._topic_model.label)

    def load_experiment(self, model_label):
        """
        Given a unigue model label, restores the state of the experiment from disk. Loads all tracked values of the experimental results
        and the state of the TopicModel inferred so far: namely the phi p_wt matrix.
        In details loads settings:\n
        - doc collection fit iteration steady_chunks\n
        - eval metrics/measures trackes per iteration\n
        - regularization parameters\n
        - document passes\n
        :param str model_label: a unigue identifier of a topic model
        :return: the latest train specification used in the experiment
        :rtype: patm.modeling.topic_model.TrainSpecs
        """
        results = self.train_results_handler.load(model_label)
        assert model_label == results['model_label']
        # print len(results['trackables']['per']['value'])
        # print sum(results['collection_passes'])
        assert len(results['trackables']['perplexity']['value']) == sum(results['collection_passes'])
        self.collection_passes = results['collection_passes']
        self.trackables = results['trackables']
        self.reg_params = results['reg_parameters']
        self.model_params = results['model_parameters']
        self._topic_model = self.phi_matrix_handler.load(model_label, results)
        return self._topic_model

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
