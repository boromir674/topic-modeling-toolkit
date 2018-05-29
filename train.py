import os
import sys
import argparse
from patm.definitions import collections_dir
from patm import get_model_factory, trainer_factory, Experiment


def get_cl_arguments():
<<<<<<< 13ad3d25a58ef152e5b746c70be2845250d56b00
    parser = argparse.ArgumentParser(prog='train.py', description='Trains a artm _topic_model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the _topic_model')
=======
    parser = argparse.ArgumentParser(prog='train.py', description='Trains a artm topic_model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the topic_model')
>>>>>>> WIP
    # parser.add_argument('category', help='the category of data to use')
    parser.add_argument('label', metavar='id', default='def', help='a unique identifier used for a newly created model')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()
<<<<<<< 13ad3d25a58ef152e5b746c70be2845250d56b00
    regularizers_param_cfg = '/data/thesis/code/regularizers.cfg'

    model_trainer = trainer_factory.create_trainer(args.collection)
    topic_model, train_specs = model_trainer.model_factory.create_model(args.label, args.config, regularizers_param_cfg, os.path.join(collections_dir, args.collection))


    experiment = Experiment()
    experiment.set_topic_model(topic_model)

    # when the trainer trains, the experiment object listens to changes
    model_trainer.register(experiment)

    model_trainer.train(topic_model.artm_model, train_specs)

    experiment.save_results(topic_model.label + '-train.pkl')
=======

    model_trainer = trainer_factory.create_trainer(args.collection)

    topic_model, train_specs = model_trainer.model_factory.create_model(args.config, os.path.join(collections_dir, args.collection))

    experiment = Experiment(topic_model)

    model_trainer.register(experiment)

    model_trainer.train(topic_model.model, train_specs)

    experiment.make_plots()

    # model2, collection_passes = model_trainer.model_factory.create_model(args.config)
    # mdl_trg = trainer_factory.create_trainer('plsa', 'n1c')
    # model3, collection_passes = mdl_trg.model_factory.create_model(args.config)
    # model3, collection_passes = mdl_trg.model_factory.create_model(args.config)
    # topic_model = mdl_fct.create_model(args.config)
>>>>>>> WIP
