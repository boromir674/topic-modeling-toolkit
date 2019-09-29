#!/usr/bin/env python

import os
import click
from .reporting import PsiReporter


import logging
logger = logging.getLogger(__name__)


@click.command()
@click.argument('dataset')
@click.option('-m', '--model')
def main(dataset, model):
    collections_dir = os.getenv('COLLECTIONS_DIR')
    if not collections_dir:
        raise RuntimeError(
            "Please set the COLLECTIONS_DIR environment variable with the path to a directory containing collections/datasets")
    reporter = PsiReporter()
    dataset_path = os.path.join(collections_dir, dataset)
    reporter.dataset = dataset_path

    print("Classes: [{}]".format(', '.join("'{}'".format(x) for x in reporter.dataset.class_names)))

    models_dir = os.path.join(dataset_path, 'models')
    if not os.path.isdir(models_dir):
        print("No models found")
        sys.exit(0)
    if model:
        phi_file_names = [model]
    else:
        phi_file_names = os.listdir(models_dir)

    # b = pr.pformat(glob('{}/*.phi'.format(models_dir)), topics_set='domain', show_class_names=True)
    b = reporter.pformat(phi_file_names, topics_set='domain', show_model_name=True, show_class_names=True)
    print(b)


if __name__ == '__main__':
    main()