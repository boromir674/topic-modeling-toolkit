import argparse
from reporting import ModelReporter


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on existing saved model topic model instances developed on the specified dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    # parser.add_argument('--details', '-d', default=False, action='store_true', help='Switch to show details about the models')
    parser.add_argument('--sort', '-s', default='perplexity', help='Whether to sort the found experiments by checking the desired metric against the corresponding models')
    return parser.parse_args()

if __name__ == '__main__':
    # from patm.definitions import COLLECTIONS_DIR_PATH, RESULTS_DIR_NAME
    from reporting import model_reporter

    COLUMNS = ['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity',
               'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi',
               'sparsity-theta',
               'background-tokens-ratio',
               'regularizers']

    # COLUMNS = ['nb-topics', 'collection-passes', 'perplexity']
               # 'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi',
               # 'sparsity-theta',
               # 'background-tokens-ratio',
               # 'regularizers']

    cli_args = get_cli_arguments()

    # reporter = ModelReporter(COLLECTIONS_DIR_PATH, results_dir_name=RESULTS_DIR_NAME)

    s = model_reporter.get_formatted_string(cli_args.dataset, columns=COLUMNS, metric=cli_args.sort, verbose=True)
    # s = reporter.get_formatted_string(cli_args.dataset, columns=None, metric=cli_args.sort)
    print('\n{}'.format(s))

    # from pprint import pprint
    # print '\n', pprint(dict(zip(reporter._columns_titles, reporter._max_col_lens)))

    # spinner = Spinner(delay=0.2)
    # spinner.start()
    # try:
    #     if cli_args.details:
    #         b1 = model_reporter.get_highlighted_detailed_string(fitness_function, model_reporter.UNDERLINE)
    #     else:
    #         b1 = model_reporter.get_model_labels_string(fitness_function)
    # except RuntimeError as e:
    #     spinner.stop()
    #     raise e
    # spinner.stop()
    # print(b1)


# class Spinner:
#     busy = False
#     delay = 0.1
#     @staticmethod
#     def spinning_cursor():
#         while 1:
#             for cursor in '|/-\\': yield cursor
#     def __init__(self, delay=None):
#         self.spinner_generator = self.spinning_cursor()
#         if delay and float(delay): self.delay = delay
#     def spinner_task(self):
#         while self.busy:
#             sys.stdout.write(next(self.spinner_generator))
#             sys.stdout.flush()
#             time.sleep(self.delay)
#             sys.stdout.write('\b')
#             sys.stdout.flush()
#     def start(self):
#         self.busy = True
#         threading.Thread(target=self.spinner_task).start()
#     def stop(self):
#         self.busy = False
#         time.sleep(self.delay)
