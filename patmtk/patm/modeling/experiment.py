from .model_factory import get_model_factory
from .persistence import ResultsWL, ModelWL
from patm.modeling.regularization.regularizers import regularizer_type2dynamic_parameters as dyn_coefs
from patm.modeling.experimental_results import ExperimentalResults


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
    # tracked_entities = {
    #     'perplexity': ['value', 'class_id_info'],
    #     'sparsity-phi': ['value'],
    #     'sparsity-theta': ['value'],
    #     'topic-kernel': ['average_coherence', 'average_contrast', 'average_purity', 'average_size', 'coherence',
    #                      'contrast', 'purity'],
    # # tokens are not tracked over time; they will be saved only for the lastest state of the inferred topics
    #     'top-tokens': ['average_coherence', 'coherence'],
    # # tokens are not tracked over time; they will be saved only for the lastest state of the inferred topics
    #     'background-tokens-ratio': ['value']
    # # tokens are not tracked over time; they will be saved only for the lastest state of the inferred topics
    # }

    def __init__(self, root_dir, cooc_dict):
        """
        Encapsulates experimentation by doing topic modeling on a 'collection' in the given patm_root_dir. A 'collection' is a proccessed document collection into BoW format and possibly split into 'train' and 'test' splits
        """
        self._dir = root_dir
        self.cooc_dict = cooc_dict
        self._loaded_dictionary = None # artm.Dictionary object. Data population happens uppon artm.Artm object creation in model_factory; dictionary.load(bin_dict_path) is called there
        self._topic_model = None
        self.collection_passes = []
        self.trackables = None
        self.model_params = {'nb_topics': [], 'document_passes': []}
        self.train_results_handler = ResultsWL(self, None)
        self.phi_matrix_handler = ModelWL(self, None)

    def init_empty_trackables(self, model):
        self._topic_model = model
        self.trackables = {}
        for evaluator_definition in model.evaluator_definitions:
            if self._strip_parameters(evaluator_definition) in ('perplexity', 'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio'):
                self.trackables[evaluator_definition] = []
            elif evaluator_definition.startswith('topic-kernel-'):
                self.trackables[evaluator_definition] = [[], [], [], {t_name: {'coherence': [], 'contrast': [], 'purity': []} for t_name in model.domain_topics}]
            elif evaluator_definition.startswith('top-tokens-'):
                self.trackables[evaluator_definition] = [[], {t_name: [] for t_name in model.domain_topics}]
        self.collection_passes = []
        self.reg_params = {reg_type: {attr: [] for attr in dyn_coefs[reg_type]} for reg_type in model.regularizer_types}
        self.model_params = {'nb_topics': [], 'document_passes': []}

    @property
    def model_factory(self):
        return get_model_factory(self.dictionary, self.cooc_dict)

    @property
    def topic_model(self):
        """
        :rtype: patm.modeling.topic_model.TopicModel
        """
        return self._topic_model

    @staticmethod
    def _strip_parameters(score_definition):
        tokens = []
        for el in score_definition.split('-'):
            try:
                _ = float(el)
            except ValueError:
                if el[0] != '@':
                    tokens.append(el)
        return '-'.join(tokens)

    @property
    def dictionary(self):
        return self._loaded_dictionary

    def set_dictionary(self, artm_dictionary):
        self._loaded_dictionary = artm_dictionary

    # TODO refactor this; remove dubious exceptions
    def update(self, topic_model, span):
        self.collection_passes.append(span) # iterations performed on the train set for the current 'steady' chunk
        print 'TOP', topic_model.domain_topics, 'AND', topic_model.background_topics

        for reg_type, reg_settings in topic_model.get_regs_param_dict().items():
            for param_name, param_value in (_ for _ in reg_settings.items() if _[1]):
                self.reg_params[reg_type][param_name].extend([param_value]*span)
        print topic_model.evaluator_names
        print topic_model.evaluator_definitions
        print "MODEL TOP-TOKENS-EVALS: [{}]".format(', '.join(_ for _ in topic_model.evaluator_definitions if _.startswith('top-tokens-')))
        # print "MODEL TOP-TOKENS-EVALS: [{}]".format(', '.join(topic_model.ge_ for _ in topic_model.evaluator_definitions if _.startswith('top-tokens-')))
        for evaluator_name, evaluator_definition in zip(topic_model.evaluator_names, topic_model.evaluator_definitions):
            # print 'REPORTABLES:', topic_model.get_evaluator(evaluator_name).reportable_attributes

            # for eval_name, evaluator_definition2 in zip(topic_model.evaluator_names, topic_model.evaluator_definitions):
            #     reportable_to_results = topic_model.get_evaluator(evaluator_name).evaluate(topic_model.artm_model)
            #     if hasattr(topic_model.get_evaluator(eval_name), 'topic_names') and evaluator_definition2.startswith('top-tokens-'):
            #         print 'NAME {}, DEF {}, topics:'.format(eval_name, evaluator_definition2), topic_model.get_evaluator(eval_name).topic_names
            #         print "top-tokens-dict-keys: [{}]".format(', '.join(sorted(reportable_to_results.keys())))
            #         print 'value type', type(reportable_to_results['value'])
            #         print reportable_to_results['coherence'][0].keys()

            reportable_to_results = topic_model.get_evaluator(evaluator_name).evaluate(topic_model.artm_model)
            # import pprint
            # pprint.pprint(reportable_to_results, indent=1)
            if self._strip_parameters(evaluator_definition) in ('perplexity', 'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio'):
                self.trackables[evaluator_definition].extend(reportable_to_results['value'][-span:])
            elif evaluator_definition.startswith('topic-kernel-'):
                self.trackables[evaluator_definition][0].extend(reportable_to_results['average_coherence'][-span:])
                self.trackables[evaluator_definition][1].extend(reportable_to_results['average_contrast'][-span:])
                self.trackables[evaluator_definition][2].extend(reportable_to_results['average_purity'][-span:])
                for topic_name, topic_metrics in self.trackables[evaluator_definition][3].items():
                    # print type(reportable_to_results['coherence'])
                    topic_metrics['coherence'].extend(map(lambda x: x[topic_name], reportable_to_results['coherence'][-span:]))
                    topic_metrics['contrast'].extend(map(lambda x: x[topic_name], reportable_to_results['contrast'][-span:]))
                    topic_metrics['purity'].extend(map(lambda x: x[topic_name], reportable_to_results['purity'][-span:]))
            elif evaluator_definition.startswith('top-tokens-'):
                self.trackables[evaluator_definition][0].extend(reportable_to_results['average_coherence'][-span:])

                print type(reportable_to_results['coherence'][0])
                print 'ELA', reportable_to_results['coherence'][0].keys()
                print self.trackables[evaluator_definition][1].keys()
                # import pprint
                # pprint.pprint(reportable_to_results, indent=2)

                for topic_name, topic_metrics in self.trackables[evaluator_definition][1].items():
                    topic_metrics.extend(map(lambda x: x[topic_name], reportable_to_results['coherence'][-span:]))

            # for eval_reportable, value in reportable_to_results.items():
            #     if evaluator_definition == 'perplexity':
            #         print value
            #         # self.trackables['perplexity'].extend()

                # print ' REPORTABLE', eval_reportable
                # # pprint.pprint(value, indent=2)
                # self.trackables[evaluator_definition][eval_reportable].extend(value[-span:])  # append only the newly produced tracked values
                # try:
                #     if type(value) == list:
                #         self.trackables[evaluator_definition][eval_reportable].extend(value[-span:])  # append only the newly produced tracked values
                #     else:
                #         raise RuntimeError
                # except RuntimeError as e:
                #     print e, '\n', type(value)
                #     try:
                #         print len(value)
                #         raise e
                #     except TypeError as er:
                #         print 'does not have __len__ implemented'
                #         print er
                #     raise EvaluationOutputLoadingException("Could not assign the value of type '{}' with key '{}' as an item in self.trackables'".format(type(value), eval_reportable))

    @property
    def current_root_dir(self):
        return self._dir

    def get_results(self):

        # res = ExperimentalResults(self._dir, self._topic_model.label, self._topic_model.nb_topics, self._topic_model.document_passes, self.topic_model.background_topics, self.topic_model.domain_topics, self.topic_model.modalities_dictionary, self.trackables)
        res = ExperimentalResults(self)
        return {
            'collection_passes': self.collection_passes,  # eg [20, 20, 40, 100]
            'trackables': self.trackables,  # TODO try list of tuples [('perplexity'), dict), ..]
            'patm_root_dir': self.current_root_dir,  # eg /data/blah/
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


class EvaluationOutputLoadingException(Exception):
    def __init__(self, msg):
        super(EvaluationOutputLoadingException, self).__init__(msg)


class DidNotReceiveTrainSignalException(Exception):
    def __init__(self, msg):
        super(DidNotReceiveTrainSignalException, self).__init__(msg)
