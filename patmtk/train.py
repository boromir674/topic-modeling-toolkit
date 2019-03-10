#!/home/kostas/software_and_libs/anaconda2/bin/python

import os
import sys
import argparse

from patm.utils import cfg2model_settings
from patm import get_model_factory, trainer_factory, Experiment, TrainSpecs
from patm.definitions import COLLECTIONS_DIR_PATH, REGULARIZERS_CFG


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='train.py', description='Trains an artm topic model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the topic_model')
    parser.add_argument('label', metavar='id', default='def', help='a unique identifier used for a newly created model')
    parser.add_argument('--reg-config', '--r-c', dest='reg_config', default=REGULARIZERS_CFG, help='the .cfg file containing initialization parameters for the active regularizers')
    parser.add_argument('--save', default=False, action='store_true', help='saves the state of the model and experimental results after the training iterations finish')
    # parser.add_argument('--load', default=False, action='store_true', help='restores the model state and progress of tracked entities from disk')
    parser.add_argument('--new-batches', '--n-b', default=False, dest='new_batches', action='store_true', help='whether to force the creation of new batches, regardless of finding batches already existing')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()
    root_dir = os.path.join(COLLECTIONS_DIR_PATH, args.collection)

    model_trainer = trainer_factory.create_trainer(args.collection, exploit_ideology_labels=True, force_new_batches=args.new_batches)
    experiment = Experiment(root_dir, model_trainer.cooc_dicts)
    model_trainer.register(experiment)  # when the model_trainer trains, the experiment object keeps track of evaluation metrics

    # if args.load:
    #     topic_model = experiment.load_experiment(args.label)
    #     print '\nLoaded experiment and model state'
    #     settings = cfg2model_settings(args.config)
    #     train_specs = TrainSpecs(15, [], [])
    # else:
    topic_model = model_trainer.model_factory.create_model(args.label, args.config, reg_cfg=args.reg_config)
    train_specs = model_trainer.model_factory.create_train_specs()
    experiment.init_empty_trackables(topic_model)
    print 'Initialized new experiment and model'
    model_trainer.train(topic_model, train_specs, effects=True, cache_theta=True)
    print 'Iterated {} times through the collection and {} times over each document: total phi updates = {}'.format(train_specs.collection_passes, topic_model.document_passes, train_specs.collection_passes * topic_model.document_passes)

    if args.save:
        experiment.save_experiment(save_phi=True)
        print "Saved results and model '{}'".format(args.label)