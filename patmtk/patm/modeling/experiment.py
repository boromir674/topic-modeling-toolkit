from .model_factory import get_model_factory
from .persistence import ResultsWL, ModelWL
from patm.modeling.regularization.regularizers import regularizer_type2dynamic_parameters as dyn_coefs
from patm.modeling.experimental_results import ExperimentalResults

from experimental_results import experimental_results_factory


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
        self.failed_top_tokens_coherence = {}
        self._total_passes = 0

    def init_empty_trackables(self, model):
        self._topic_model = model
        self.trackables = {}
        self.failed_top_tokens_coherence = {}
        self._total_passes = 0
        for evaluator_definition in model.evaluator_definitions:
            if self._strip_parameters(evaluator_definition) in ('perplexity', 'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio'):
                self.trackables[evaluator_definition] = []
            elif evaluator_definition.startswith('topic-kernel-'):
                self.trackables[evaluator_definition] = [[], [], [], {t_name: {'coherence': [], 'contrast': [], 'purity': []} for t_name in model.domain_topics}]
            elif evaluator_definition.startswith('top-tokens-'):
                self.trackables[evaluator_definition] = [[], {t_name: [] for t_name in model.domain_topics}]
                self.failed_top_tokens_coherence[evaluator_definition] = {t_name: [] for t_name in model.domain_topics}
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

        for reg_type, reg_settings in topic_model.get_regs_param_dict().items():
            for param_name, param_value in (_ for _ in reg_settings.items() if _[1]):
                self.reg_params[reg_type][param_name].extend([param_value]*span)

        for evaluator_name, evaluator_definition in zip(topic_model.evaluator_names, topic_model.evaluator_definitions):
            reportable_to_results = topic_model.get_evaluator(evaluator_name).evaluate(topic_model.artm_model)

            if self._strip_parameters(evaluator_definition) in ('perplexity', 'sparsity-phi', 'sparsity-theta', 'background-tokens-ratio'):
                self.trackables[evaluator_definition].extend(reportable_to_results['value'][-span:])
            elif evaluator_definition.startswith('topic-kernel-'):
                self.trackables[evaluator_definition][0].extend(reportable_to_results['average_coherence'][-span:])
                self.trackables[evaluator_definition][1].extend(reportable_to_results['average_contrast'][-span:])
                self.trackables[evaluator_definition][2].extend(reportable_to_results['average_purity'][-span:])
                for topic_name, topic_metrics in self.trackables[evaluator_definition][3].items():
                    topic_metrics['coherence'].extend(map(lambda x: x[topic_name], reportable_to_results['coherence'][-span:]))
                    topic_metrics['contrast'].extend(map(lambda x: x[topic_name], reportable_to_results['contrast'][-span:]))
                    topic_metrics['purity'].extend(map(lambda x: x[topic_name], reportable_to_results['purity'][-span:]))
            elif evaluator_definition.startswith('top-tokens-'):
                self.trackables[evaluator_definition][0].extend(reportable_to_results['average_coherence'][-span:])

                assert 'coherence' in reportable_to_results
                for topic_name, topic_metrics in self.trackables[evaluator_definition][1].items():
                    try:
                        topic_metrics.extend(map(lambda x: x[topic_name], reportable_to_results['coherence'][-span:]))
                    except KeyError:
                        if len(self.failed_top_tokens_coherence[evaluator_definition][topic_name]) == 0:
                            self.failed_top_tokens_coherence[evaluator_definition][topic_name].append((self._total_passes, span))
                        else:
                            if span + self.failed_top_tokens_coherence[evaluator_definition][topic_name][-1][0] == self._total_passes:
                                perv_tuple = self.failed_top_tokens_coherence[evaluator_definition][topic_name][-1]
                                self.failed_top_tokens_coherence[evaluator_definition][topic_name][-1] = (self._total_passes, perv_tuple[1]+span)
                self._total_passes += span

    @property
    def current_root_dir(self):
        return self._dir

    def get_results(self):
        return experimental_results_factory.create_from_experiment(self)

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
        # the above will fail when metrics outside the artm library start to get tracked, because these will be able to only capture the last state of the metric trajectory due to fitting by "chunks"
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
        assert len(results['trackables']['perplexity']['value']) == sum(results['collection_passes'])
        self.collection_passes = results['collection_passes']
        self._total_passes = sum(self.collection_passes)
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
