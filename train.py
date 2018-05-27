import sys
import argparse
from patm import get_model_factory, trainer_factory


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='train.py', description='Trains a artm model and stores \'evaluation\' scores', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the model')
    # parser.add_argument('category', help='the category of data to use')
    # parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()
    mdl_tr = trainer_factory.create_trainer(args.collection)
    # mdl_fct = get_model_factory(mdl_tr.dictionary)
    model, train_specs = mdl_tr.model_factory.create_model(args.config)
    mdl_tr.register(model)
    # model2, collection_passes = mdl_tr.model_factory.create_model(args.config)
    # mdl_trg = trainer_factory.create_trainer('plsa', 'n1c')
    # model3, collection_passes = mdl_trg.model_factory.create_model(args.config)
    # model3, collection_passes = mdl_trg.model_factory.create_model(args.config)
    # model = mdl_fct.create_model(args.config)
    mdl_tr.train(model, train_specs)
