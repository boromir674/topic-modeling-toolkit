COLLECTIONS_DIR_PATH = '/data/thesis/data/collections'
RESULTS_DIR_NAME = 'results'
PLOTS_DIR_NAME = 'graphs'

import os

from .model_selection import ResultsHandler

results_handler = ResultsHandler(collection_root_path=COLLECTIONS_DIR_PATH, results_dir_name=RESULTS_DIR_NAME)

from .reporter import ModelReporter
model_reporter = ModelReporter(collections_root_path=COLLECTIONS_DIR_PATH, results_dir_name=RESULTS_DIR_NAME)

from .graph_maker import GraphMaker
graph_maker = GraphMaker(collections_root_path=COLLECTIONS_DIR_PATH, plot_dir_name=PLOTS_DIR_NAME)
