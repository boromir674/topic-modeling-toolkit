import os
import sys
import argparse
from patm.definitions import collections_dir
from patm import get_model_factory, trainer_factory, Experiment


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='train.py', description='Trains a artm _topic_model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the topic_model')
    parser.add_argument('label', metavar='id', default='def', help='a unique identifier used for a newly created model')
    parser.add_argument('--save', default=False, type=bool, action='store_true', help='saves the state of the model and experimental results after the training iterations finish')
    parser.add_argument('--load', default=False, type=bool, action='store_true', help='restores the model state and progress of tracked entities from disk')
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

    if args.load:
        train_specs = experiment.load_experiment(args.label)
        topic_model = experiment.topic_model
    else:
        topic_model, train_specs = model_trainer.model_factory.create_model(args.label, args.config, regularizers_param_cfg)
        experiment.set_topic_model(topic_model, empty_trackables=True)

    # when the trainer trains, the experiment object listens to changes
    model_trainer.register(experiment)
    model_trainer.train(topic_model.artm_model, train_specs)

    if args.save:
        experiment.save_experiment(save_phi=True)
