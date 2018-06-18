#!/home/kostas/software_and_libs/anaconda2/bin/python

import os
import pickle
import glob
import argparse
from patm.dataset import Dataset

supported_datasets = tuple(['uci'])

def get_datasets(collections_root, details=True):
    collection_names = os.listdir(collections_root)
    info_list = []
    for col_name in collection_names:
        for sup_dt_type in supported_datasets:
            dataset_pattern = '{}/*.{}.pkl'.format(os.path.join(collections_root, col_name), sup_dt_type)
            for pickled_dataset in glob.glob(dataset_pattern):
                dataset_object = Dataset.load(pickled_dataset)
                if details:
                    info_list.append(str(dataset_object))
                else:
                    info_list.append(col_name + '/' + os.path.basename(pickled_dataset))
    return info_list


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Reports on topic-modeling datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--details', '-d', dest='details', default=False, action='store_true', help='Switch to show details about the datasets')
    return parser.parse_args()


if __name__ == '__main__':
    collection_root_dir = '/data/thesis/data/collections'
    args = get_cli_arguments()
    dataset_info_list = get_datasets(collection_root_dir, details=args.details)
    b = '\n'.join(dataset_info_list)
    print b
