import os
import sys
import argparse
from patm.definitions import COLLECTIONS_DIR, REGULARIZERS_CFG
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
    root_dir = os.path.join(COLLECTIONS_DIR, args.collection)

    model_trainer = trainer_factory.create_trainer(args.collection, exploit_ideology_labels=False, force_new_batches=args.new_batches)
    experiment = Experiment(root_dir, model_trainer.cooc_dicts)
    model_trainer.register(experiment)  # when the model_trainer trains, the experiment object listens to changes

    if args.load:
        topic_model = experiment.load_experiment(args.label)
        print '\nLoaded experiment and model state'
        settings = cfg2model_settings(args.config)
        train_specs = TrainSpecs(15, [], [])
    else:
        topic_model = model_trainer.model_factory.create_model(args.label, args.config, reg_cfg=args.reg_config)
        train_specs = model_trainer.model_factory.create_train_specs()
        experiment.init_empty_trackables(topic_model)
        print 'Initialized new experiment and model'

    for w in topic_model.regularizer_wrappers:
        print w.name, w.label

    for e in topic_model.evaluators:
        print e.name, e.label, e.settings

    t = topic_model.get_regs_param_dict()
    for k, v in t.items():
        print 'K', k, v

    print
    print topic_model.modalities_dictionary
    print topic_model.domain_topics
    print topic_model.background_topics
    print topic_model.regularizer_names
    print topic_model.regularizer_types
    print topic_model.evaluator_names
    print topic_model.evaluator_definitions

    import sys
    sys.exit()
    model_trainer.train(topic_model, train_specs, effects=True)

    for w in topic_model.regularizer_wrappers:
        print w.name, w.label

    for e in topic_model.evaluators:
        r = e.artm_score.class_ids
        print e.name, e.label, r


    t = topic_model.get_regs_param_dict()
    for k, v in t.items():
        print 'K', k, v
    print
    print topic_model.modalities_dictionary
    print topic_model.domain_topics
    print topic_model.background_topics
    print topic_model.regularizer_names
    print topic_model.regularizer_types
    print topic_model.evaluator_names
    print topic_model.evaluator_definitions
    print 'Iterated {} times through the collection and {} times over each document: total phi updates = {}'.format(train_specs.collection_passes, topic_model.document_passes, train_specs.collection_passes * topic_model.document_passes)
    print
    print

    if args.save:
        experiment.save_experiment(save_phi=True)
