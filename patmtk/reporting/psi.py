import os
from collections import defaultdict
import attr


@attr.s
class PsiReporter(object):
    dataset_path = attr.ib(init=True)
    topics = attr.ib(init=False, default={'all': lambda x: x.topic_names,
                                          'domain': lambda x: x.domain_topics,
                                          'background': lambda x: x.background_topics})

    def pformat(self, model_paths, topics_set='domain', show_class_names=True, show_topic_names=True, precision=2):
        pass
    def artifacts(self):
        new_exp_obj = Experiment(os.path.join(collections_root_dir, test_dataset.name))
        trainer.register(new_exp_obj)
        loaded_model = new_exp_obj.load_experiment(model.label)

    def topic_names(self, topics_set):
        return {'all': lambda x: x.topic_names}.get(topics_set, )

    def paths(self, *args):
        if os.path.isfile(args[0]):  # is a full path
            return args[0], os.path.join(os.path.dirname(args[0]), '../results', os.path.basename(args[0]))
        return os.path.join(self.dataset_path, 'models', args[0]), os.path.join(self.dataset_path, 'results', args[0])  # input is model label
