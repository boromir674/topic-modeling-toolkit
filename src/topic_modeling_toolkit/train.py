#!/usr/bin/env python

import os
import sys
import argparse

from topic_modeling_toolkit.patm import TrainerFactory, Experiment


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='train.py', description='Trains an artm topic model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the topic_model')
    parser.add_argument('label', metavar='id', default='def', help='a unique identifier used for a newly created model')
    parser.add_argument('--reg-config', '--r-c', dest='reg_config', help='the .cfg file containing initialization parameters for the active regularizers')
    parser.add_argument('--save', default=True, action='store_true', help='saves the state of the model and experimental results after the training iterations finish')
    # parser.add_argument('--load', default=False, action='store_true', help='restores the model state and progress of tracked entities from disk')
    parser.add_argument('--new-batches', '--n-b', default=False, dest='new_batches', action='store_true', help='whether to force the creation of new batches, regardless of finding batches already existing')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main():
    args = get_cl_arguments()
    collections_dir = os.getenv('COLLECTIONS_DIR')
    if not collections_dir:
        raise RuntimeError(
            "Please set the COLLECTIONS_DIR environment variable to the directory containing collections/datasets")
    root_dir = os.path.join(collections_dir, args.collection)
    model_trainer = TrainerFactory().create_trainer(root_dir, exploit_ideology_labels=True,
                                                    force_new_batches=args.new_batches)
    experiment = Experiment(root_dir)
    model_trainer.register(
        experiment)  # when the model_trainer trains, the experiment object keeps track of evaluation metrics

    # if args.load:
    #     topic_model = experiment.load_experiment(args.label)
    #     print '\nLoaded experiment and model state'
    #     settings = cfg2model_settings(args.config)
    #     train_specs = TrainSpecs(15, [], [])
    # else:
    topic_model = model_trainer.model_factory.create_model(args.label, args.config, reg_cfg=args.reg_config,
                                                           show_progress_bars=False)
    train_specs = model_trainer.model_factory.create_train_specs()
    experiment.init_empty_trackables(topic_model)
    # static_reg_specs = {}  # regularizers' parameters that should be kept constant during data fitting (model training)
    # import pprint
    # pprint.pprint({k: dict(v, **{setting_name: setting_value for setting_name, setting_value in {'target topics': (lambda x: 'all' if len(x) == 0 else '[{}]'.format(', '.join(x)))(topic_model.get_reg_obj(topic_model.get_reg_name(k)).topic_names), 'mods': getattr(topic_model.get_reg_obj(topic_model.get_reg_name(k)), 'class_ids', None)}.items()}) for k, v in self.static_regularization_specs.items()})
    # pprint.pprint(tm.modalities_dictionary)
    print("Initialized Model:")
    print(topic_model.pformat_regularizers)
    print(topic_model.pformat_modalities)
    model_trainer.train(topic_model, train_specs, effects=True, cache_theta=True)
    print('Iterated {} times through the collection and {} times over each document: total phi updates = {}'.
          format(train_specs.collection_passes, topic_model.document_passes,
                 train_specs.collection_passes * topic_model.document_passes))

    if args.save:
        experiment.save_experiment(save_phi=True)
        print("Saved results and model '{}'".format(args.label))


if __name__ == '__main__':
    main()