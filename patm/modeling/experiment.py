import os
import cPickle as pickle
import json

from .regularizers import parameter_name2encoder

from .regularizers import parameter_name2encoder


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
    """

    def __init__(self):
        """

        """
        self._topic_model = None
        self.collection_passes = []
        self.specs_instances = []
        self.trackables = {}
        self.reg_params = []
        self.model_params = {'nb_topics': [], 'document_passes': []}

    @property
    def topic_model(self):
        return self._topic_model

    def set_topic_model(self, topic_model):
        self._topic_model = topic_model
        self.trackables = {key: {inner_k: [] for inner_k in self._topic_model.evaluators[key].attributes} for key in self._topic_model.evaluators.keys()}

    # def set_parameters(self, expressions_list):
    #     """
    #     Allows setting new values to already existing
    #     :param expressions_list:
    #     :return:
    #     """
    #     components_lists = [expression.split('.') for expression in expressions_list]
    #     for reg_name, param, value in components_lists:
    #         self._topic_model.set_parameter(reg_name, param, value)

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
                    print e, '\n'
                    print type(value)
                    try:
                        print len(value)
                    except TypeError:
                        print 'does not have __len__ implemented'
                    raise EvaluationOutputLoadingException("Could not assign the value of type '{}' with key '{}' as an item in self.trackables'".format(type(value), inner_k))

    @property
    def current_root_dir(self):
        return self.specs_instances[-1]['output_dir']

    def get_results(self):
        return _stringify_trackable_dicts({
            'collection_passes': self.collection_passes,  # eg [20, 20, 40, 100]
            'trackables': self.trackables,  # TODO try list of tuples [('perplexity'), dict), ..]
            'root_dir': self.current_root_dir,  # eg /data/blah/
            'model_label': self.topic_model.label,
            'model_parameters': self.model_params,
            'reg_parameters': self.reg_params})

    def save_results(self, filename):
        """
        Dumps the dictionary-type accumulated results with the given file name. The file is saved in the directory specified by the latest train specifications (TrainSpecs).\n
        :param str filename: the name with possible extension to use
        """
        if not self.collection_passes:
            raise DidNotReceiveTrainSignalException('Model probably hasn\'t been fitted since len(self.collection_passes) = {}, len(self.specs_instances) = {}'.format(len(self.collection_passes), len(self.specs_instances)))
        for k, v in self.trackables.items():
            print(k)
            for ik, iv in v.items():
                print ik, len(iv)
        print(len(self.collection_passes), self.collection_passes)
        assert all(map(lambda x: len(x) == sum(self.collection_passes), [values_list for eval2scoresdict in self.trackables.values() for values_list in eval2scoresdict.values()]))
        # assert all(map(lambda x: len(x) == sum(self.collection_passes), [score_list for scores_dict in self.trackables.values() for score_list in scores_dict]))
        results_dir = os.path.join(self.current_root_dir, 'results')
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
            print 'Created {} dir to pickle experiment results'.format(results_dir)
        target_name = os.path.join(results_dir, filename)
        res = self.get_results()
        with open(target_name, 'w') as results_file:
            json.dump(res, results_file)
        print('Saved results as', target_name)


def _stringify_trackable_dicts(results):
    tr = results['trackables']
    for k, v in tr.items():
        if type(v) == list:
            tr[k] = map(lambda x: str(x), tr[k])
        elif type(v) == dict:
            for in_k, in_v in v.items():
                if type(in_v) == list:
                    v[in_k] = map(lambda x: str(x), v[in_k])
    return results


def load_results(path_file):
    with open(path_file, 'rb') as results_file:
        results = json.load(results_file, encoding='utf-8')
    assert 'collection_passes' in results and 'trackables' in results, 'root_dir' in results
    return results


class EvaluationOutputLoadingException(Exception):
    def __init__(self, msg):
        super(EvaluationOutputLoadingException, self).__init__(msg)


class DidNotReceiveTrainSignalException(Exception):
    def __init__(self, msg):
        super(DidNotReceiveTrainSignalException, self).__init__(msg)

