import os
import sys
import argparse
from patm.definitions import collections_dir
from patm import get_model_factory, trainer_factory, Experiment, TrainSpecs
from patm.utils import cfg2model_settings

from patm.modeling import trajectory_builder

def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='train.py', description='Trains a artm _topic_model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the topic_model')
    parser.add_argument('label', metavar='id', default='def', help='a unique identifier used for a newly created model')
    parser.add_argument('--save', default=False, action='store_true', help='saves the state of the model and experimental results after the training iterations finish')
    parser.add_argument('--load', default=False, action='store_true', help='restores the model state and progress of tracked entities from disk')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()
    root_dir = os.path.join(collections_dir, args.collection)
    regularizers_param_cfg = '/data/thesis/code/regularizers.cfg'

    experiment = Experiment(root_dir)
    model_trainer = trainer_factory.create_trainer(args.collection)
    model_trainer.register(experiment)  # when the model_trainer trains, the experiment object listens to changes

    if args.load:
        topic_model = experiment.load_experiment(args.label)
        print '\nLoaded experiment and model state'
        settings = cfg2model_settings(args.config)
        train_specs = TrainSpecs(settings['learning']['collection_passes'], [], [])
    else:
        topic_model, train_specs = model_trainer.model_factory.create_model(args.label, args.config, regularizers_param_cfg)
        experiment.init_empty_trackables(topic_model)
        print 'Initialized new experiment and model'
    # train_specs = {'collection_passes': 30}
    model_trainer.train(topic_model, train_specs)
    print 'Iterated {} times through the collection and {} times over each document: total phi updates = {}'.format(train_specs.collection_passes, topic_model.document_passes, train_specs.collection_passes * topic_model.document_passes)
    if args.save:
        experiment.save_experiment(save_phi=True)
