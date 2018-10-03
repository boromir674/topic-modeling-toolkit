import os
import sys
import argparse
from patm.definitions import collections_dir, REGULARIZERS_CFG
from patm import get_model_factory, trainer_factory, Experiment, TrainSpecs
from patm.utils import cfg2model_settings


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='train.py', description='Trains an artm topic model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the topic_model')
    parser.add_argument('label', metavar='id', default='def', help='a unique identifier used for a newly created model')
    parser.add_argument('--reg-config', '--r-c', dest='reg_config', default=REGULARIZERS_CFG, help='the .cfg file containing initialization parameters for the active regularizers')
    parser.add_argument('--save', default=False, action='store_true', help='saves the state of the model and experimental results after the training iterations finish')
    parser.add_argument('--load', default=False, action='store_true', help='restores the model state and progress of tracked entities from disk')
    parser.add_argument('--new-batches', '--n-b', default=False, dest='new_batches', action='store_true', help='whether to force the creation of new batches, regardless of finding batches already existing')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def get_trajs_specs(iters):
    traj_1 = trajectory_builder.begin_trajectory('tau').deactivate(deact).interpolate_to(iters - deact, -0.1, start=-1).create()
    traj_2 = trajectory_builder.begin_trajectory('tau').deactivate(deact).interpolate_to(iters - deact, -0.1, start=-0.8).create()
    return TrainSpecs(train_iters, ['ssp', 'sst'], [traj_1, traj_2])

if __name__ == '__main__':
    args = get_cl_arguments()
    root_dir = os.path.join(collections_dir, args.collection)

    model_trainer = trainer_factory.create_trainer(args.collection, exploit_ideology_labels=False, force_new_batches=args.new_batches)
    experiment = Experiment(root_dir, model_trainer.cooc_dicts)
    model_trainer.register(experiment)  # when the model_trainer trains, the experiment object listens to changes

    if args.load:
        topic_model = experiment.load_experiment(args.label)
        print '\nLoaded experiment and model state'
        settings = cfg2model_settings(args.config)
        train_specs = TrainSpecs(15, [], [])
    else:
        topic_model = model_trainer.model_factory.create_model00(args.label, args.config, reg_cfg=args.reg_config)
        train_specs = model_trainer.model_factory.create_train_specs()
        experiment.init_empty_trackables(topic_model)
        print 'Initialized new experiment and model'

    for name, tr in train_specs.tau_trajectory_list:
        print 'REG NAME:', name, '\n', [type(i) for i in tr]

    for reg in topic_model.regularizer_names:
        params = topic_model.get_reg_wrapper(reg).static_parameters
        print 'AA', reg, params.items()
    # for x in train_specs.tau_trajectory_list:
    #     print x[0], x[1]

    model_trainer.train(topic_model, train_specs, effects=True)

    print topic_model.domain_topics

    # ## TRAIN WITH DYNAMICALLY CHANGING TAU COEEFICIENT VALUE
    # nexta = [10, 10, 10, 10, 10, 10]
    # train_iters = 100
    # deact = 2
    #
    # from patm.modeling import trajectory_builder
    # tr1 = trajectory_builder.begin_trajectory('tau').deactivate(deact).\
    #     interpolate_to(nexta[0], -10).interpolate_to(nexta[1], -10). \
    #     interpolate_to(nexta[2], -10).interpolate_to(nexta[3], -10). \
    #     interpolate_to(nexta[4], -10).interpolate_to(nexta[5], -10).steady_prev(train_iters - deact - sum(map(lambda x: eval('nexta' + str(x)), range(1, 7)))).create()
    # print tr1
    # # tr2 = trajectory_builder.begin_trajectory('tau').deactivate(deact).interpolate_to(iters - deact, -0.1, start=-0.8).create()

    # train_specs = get_trajs_specs(train_iters)
    #
    # print train_specs.tau_trajectory_list[0][1]
    # print train_specs.tau_trajectory_list[1][1]

    # print topic_model.regularizer_names
    # print topic_model.regularizer_types
    # print topic_model.evaluator_names
    # print topic_model.evaluator_definitions
    #
    # # train_specs = {'collection_passes': 30}
    # model_trainer.train(topic_model, train_specs)
    # print 'Iterated {} times through the collection and {} times over each document: total phi updates = {}'.format(train_specs.collection_passes, topic_model.document_passes, train_specs.collection_passes * topic_model.document_passes)
    # if args.save:
    #     experiment.save_experiment(save_phi=True)
