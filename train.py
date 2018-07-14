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

def get_trajs_specs(iters):
    tr1 = trajectory_builder.begin_trajectory('tau').deactivate(deact).interpolate_to(iters - deact, -0.1, start=-1).create()
    tr2 = trajectory_builder.begin_trajectory('tau').deactivate(deact).interpolate_to(iters - deact, -0.1, start=-0.8).create()
    return TrainSpecs(train_iters, ['ssp', 'sst'], [tr1, tr2])

if __name__ == '__main__':
    args = get_cl_arguments()
    root_dir = os.path.join(collections_dir, args.collection)
    regularizers_param_cfg = '/data/thesis/code/regularizers.cfg'

    model_trainer = trainer_factory.create_trainer('articles', exploit_ideology_labels=False, force_new_batches=True)
    # experiment = Experiment(root_dir, model_trainer.cooc_dicts)
    # model_trainer.register(experiment)  # when the model_trainer trains, the experiment object listens to changes
    #
    # train_iters = 50
    # deact = 10
    # train_specs = get_trajs_specs(train_iters)
    #
    # print train_specs.tau_trajectory_list[0][1]
    # print train_specs.tau_trajectory_list[1][1]
    #
    # if args.load:
    #     topic_model = experiment.load_experiment(args.label)
    #     print '\nLoaded experiment and model state'
    #     settings = cfg2model_settings(args.config)
    #     # train_specs = TrainSpecs(15, [], [])
    #
    # else:
    #     topic_model, train_specs = model_trainer.model_factory.create_model(args.label, args.config, regularizers_param_cfg)
    #     experiment.init_empty_trackables(topic_model)
    #     print 'Initialized new experiment and model'
    #
    # print topic_model.regularizer_names
    # print topic_model.regularizer_types
    # print topic_model.evaluator_names
    # print topic_model.evaluator_types
    #
    # # train_specs = {'collection_passes': 30}
    # model_trainer.train(topic_model, train_specs)
    # print 'Iterated {} times through the collection and {} times over each document: total phi updates = {}'.format(train_specs.collection_passes, topic_model.document_passes, train_specs.collection_passes * topic_model.document_passes)
    # if args.save:
    #     experiment.save_experiment(save_phi=True)
