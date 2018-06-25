import os
import glob
import argparse
from patm.definitions import results_root, models_root
from patm.utils import load_results

def load_reportables(results_path):
    d = load_results(results_path)
    try:
        rr = [d['model_label'], d['model_parameters']['nb_topics'][-1][1], sum(d['collection_passes']), d['model_parameters']['document_passes'][-1][1], sorted(i for i in set([item for subl in [param_dict.keys() for _, param_dict in d['reg_parameters']] for item in subl])),
            d['trackables']['perplexity']['value'][-1],
            d['trackables']['sparsity-phi']['value'][-1],
            d['trackables']['sparsity-theta']['value'][-1],
            d['trackables']['background-tokens-ratio']['value'][-1],
            d['trackables']['topic-kernel']['average_coherence'][-1],
            d['trackables']['topic-kernel']['average_contrast'][-1],
            d['trackables']['topic-kernel']['average_purity'][-1],
            d['trackables']['top-tokens-10']['average_coherence'][-1],
            d['trackables']['top-tokens-100']['average_coherence'][-1]
            ]
    except KeyError:
        rr = [d['model_label'], d['model_parameters']['nb_topics'][-1][1], sum(d['collection_passes']), d['model_parameters']['document_passes'][-1][1], sorted(i for i in set([item for subl in [param_dict.keys() for _, param_dict in d['reg_parameters']] for item in subl]))]
    return rr

class ModelReporter(object):
    def __init__(self):
        self._r = None

    def to_string(self, reportables):
        r = reportables
        return 'NAME: {}, nb_topics: {}\ncol_iters: {}, doc_iters: {}, total: {}\nregs: {}'.format(r[0], r[1], r[2], r[3], r[2] * r[3], '[{}]'.format(', '.join(str(_) for _ in r[4])))


model_reporter = ModelReporter()


def get_models(collection_dir, sort='', details=True):
    collection_names = os.listdir(collection_dir)
    results_dir = os.path.join(collection_dir, results_root)
    models_dir = os.path.join(collection_dir, models_root)

    result_paths = glob.glob('{}/*-train.json'.format(results_dir))

    stripped_phi_names = [os.path.basename(phi_path).replace('.phi', '') for phi_path in glob.glob('{}/*-train.phi'.format(models_dir))]
    stripped_result_names = map(lambda x: os.path.basename(x).replace('.json', ''), result_paths)
    if stripped_phi_names != stripped_result_names:
        print '{} phi matrices do not correspond to resuls jsons'.format([_ for _ in stripped_phi_names if _ not in stripped_result_names], )
        print '{} results jsons do not correspond to phi matrices'.format([_ for _ in stripped_result_names if _ not in stripped_phi_names])
    if not details:
        return stripped_result_names
    return map(lambda x: model_reporter.to_string(x), map(lambda x: load_reportables(x), result_paths))


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on existing saved model topic model instances developed on the specified dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('--details', '-d', default=False, action='store_true', help='Switch to show details about the models')
    return parser.parse_args()

if __name__ == '__main__':
    collections_root_dir = '/data/thesis/data/collections'
    args = get_cli_arguments()

    dataset_dir = os.path.join(collections_root_dir, args.dataset)

    infos = get_models(dataset_dir, details=args.details)
    for i in infos:
        print i
