#!/usr/bin/env python

import argparse
import os
import sys
import re
from PyInquirer import prompt

from patm.definitions import COLLECTIONS_DIR_PATH
from processors import PipeHandler
from patm import political_spectrum
from patm.build_coherence import CoherenceFilesBuilder


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='transform.py', description='Extracts from pickled data, preprocesses text and saves all necessary files to train an artm model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('category', help='the category of data to use')
    parser.add_argument('config', help='the .cfg file to use for constructing a pipeline')
    parser.add_argument('collection', help='a given name for the collection')
    parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    parser.add_argument('--window', '-w', default=10, type=int, help='number of tokens around specific token, which are used in calculation of cooccurrences')
    parser.add_argument('--min_tf', default=0, type=int,
                        help='Minimal value of cooccurrences of a pair of tokens that are saved in dictionary of cooccurrences.')
                             # 'For each int value a file is built to be used for coherence computation. By default builds one with min_tf=0')
    parser.add_argument('--min_df', default=0, type=int,
                        help='Minimal value of documents in which a specific pair of tokens occurred together closely.')
                             # 'For each int value a file is built to be used for coherence computation. By default builds one with min_df=0')
    parser.add_argument('--exclude_class_labels_from_vocab', '--ecliv', action='store_true', default=False, help='Whether to ommit adding document labels in the vocabulary file generated. If set to False document labels are treated as valid registered vocabulary tokens.')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#
# def naming(psm):
#     """
#
#     :param patm.discreetization.PoliticalSpectrum psm:
#     :return:
#     """
#     from patm.discreetization import Bins
#     psm.discreetization_scheme = 'legacy-scheme'
#
#     namings = [['liberal', 'centre', 'conservative'],
#                ['liberal', 'centre_liberal', 'centre_conservative', 'conservative'],
#                ['liberal', 'centre_liberal', 'centre', 'centre_conservative', 'conservative'],
#                ['more_liberal', 'liberal', 'centre_liberal', 'centre', 'centre_conservative', 'conservative']]
#
#     questions = [
#         {
#             'type': 'list',  # navigate with arrows through choices
#             'name': 'discreetization-scheme',
#             'message': 'Use a registered discreetization scheme or create a new one.',  #  of classes [{}] with distribution [{}]? and bins:\n\n{}'.format(
#             'choices': ["{}: [{}] with class frequencies [{}]".format(name, ', '.join(c for c in scheme.class_names),
#                                                                   ', '.join('{:.2f}'.format(x) for x in scheme.class_distribution)) for name, scheme in psm] + ['Create new discreetization scheme']
#                 # psm.class_names,
#                 # ', '.join('{:.2f}'.format(x) for x in psm.class_distribution),
#                 # Bins([_[1] for _ in psm.discreetization_scheme])),
#             # 'choices': []
#         },
#         {
#             'type': 'confirm',  # navigate with arrows through choices
#             'name': 'use-scheme',
#             'message': "Create dataset using the '{}' discreetization scheme with classses [{}] and frquencies distribution [{}]?".format(),
#             # Use legacy discreetization scheme of classes [{}] with distribution [{}]? and bins:\n\n{}'.format(psm.class_names,
#             #                                                                                                               ', '.join('{:.2f}'.format(x) for x in psm.class_distribution),
#             #                                                                                                               Bins([_[1] for _ in psm.discreetization_scheme])),
#             'when': lambda x: x['discreetization-scheme'] != 'Create new discreetization scheme'
#         },
#         {
#             'type': 'list',  # navigate with arrows through choices
#             'name': 'naming-scheme',
#             'message': 'You can pick one of the pre made class names or define your own custom',
#             'choices': [', '.join(names) for names in namings] + ['Create custom names'],
#             'when': lambda x: x['discreetization-scheme'] == 'Create new discreetization scheme'
#         },
#         {
#             'type': 'input',  # navigate with arrows through choices
#             'name': 'custom-class-names',
#             'message': 'Give space separated class names',
#             'when': lambda x: x['use-default-naming-scheme'] == 'Create custom names'
#         }
#     ]
#     answers = {}
#     while not answers.get('use-scheme', False):
#         answers = prompt(questions)
#         if answers['discreetization-scheme'] == 'Create new discreetization scheme':
#             class_names = ['{}_Class'.format(x) for x in answers.get('naming-scheme', 'custom-class-names')]
#
#     return answers

def ask_discreetization(spectrum, pipe_handler, pool_size=100, prob=0.3, max_generation=100):
    # from patm.discreetization import Bins
    # psm.discreetization_scheme = 'legacy-scheme'

    namings = [['liberal', 'centre', 'conservative'],
               ['liberal', 'centre_liberal', 'centre_conservative', 'conservative'],
               ['liberal', 'centre_liberal', 'centre', 'centre_conservative', 'conservative'],
               ['more_liberal', 'liberal', 'centre_liberal', 'centre', 'centre_conservative', 'conservative']]

    questions = [
        {
            'type': 'list',  # navigate with arrows through choices
            'name': 'discreetization-scheme',
            'message': 'Use a registered discreetization scheme or create a new one.',
            'choices': ['Classes: [{}] with distribution [{}]'.format(' '.join(scheme.class_names), ' '.join('{:.2f}'.format(x) for x in spectrum.distribution(scheme))) for name, scheme in spectrum]
                       + ['Create new']

        },
        {
            'type': 'list',  # navigate with arrows through choices
            'name': 'naming-scheme',
            'message': 'You can pick one of the pre made class names or define your own custom',
            'choices': ['{}: {}'.format(len(names), ' '.join(names)) for names in namings] + ['Create custom names'],
            'when': lambda x: x['discreetization-scheme'] == 'Create new',
            # 'filter': lambda x: x.split(' ')
        },
        {
            'type': 'input',  # navigate with arrows through choices
            'name': 'custom-class-names',
            'message': 'Give space separated class names',
            'when': lambda x: x.get('naming-scheme', None) == 'Create custom names',
            # 'filter': lambda x: x.split(' ')
        }
    ]
    answers = prompt(questions)
    if answers['discreetization-scheme'] == 'Create new':
        print("Evolving discreetization scheme ..")
        class_names = _class_names(answers.get('custom-class-names', answers['naming-scheme']))
        return spectrum.balance_frequencies(class_names, pipe_handler.outlet_ids, len(class_names) - 1, pool_size, prob=prob, max_generation=max_generation)
    return spectrumspectrum[answers['discreetization-scheme']]

def _class_names(string):
    return ['{}_Class'.format(x) for x in re.sub(r'\d+:\s+', '', string).split(' ')]

def ask_persist(pol_spctrum):
    return prompt([{'type': 'confirm',
                    'name': 'create-dataset',
                    'message': "Use scheme [{}] with resulting distribution [{}]?".format(' '.join(pol_spctrum.class_names), ', '.join('{:.2f}'.format(x) for x in political_spectrum.class_distribution)),
                    'default': True}])['create-dataset']


if __name__ == '__main__':
    args = get_cl_arguments()
    nb_docs = args.sample
    if nb_docs != 'all':
        nb_docs = int(nb_docs)


    ph = PipeHandler()
    ph.process(args.config, args.category, sample=nb_docs, verbose=True)
    political_spectrum.datapoint_ids = ph.outlet_ids

    while 1:
        scheme = ask_discreetization(political_spectrum, ph, pool_size=100, prob=0.3, max_generation=100)
        print("Scheme with classes: [{}]".format(' '.join(x for x, _ in scheme)))
        try:
            political_spectrum.discreetization_scheme = scheme
        except ValueError as e:
            raise ValueError("{}. {}".format(e, type(scheme).__name__))

        if ask_persist(political_spectrum):
            uci_dt = ph.persist(os.path.join(COLLECTIONS_DIR_PATH, args.collection), political_spectrum.poster_id2ideology_label, political_spectrum.class_names, add_class_labels_to_vocab=not args.exclude_class_labels_from_vocab)
            break

    print(uci_dt)

    print('\nBuilding coocurences information')
    coherence_builder = CoherenceFilesBuilder(os.path.join(COLLECTIONS_DIR_PATH, args.collection))
    coherence_builder.create_files(cooc_window=args.window,
                                   min_tf=args.min_tf,
                                   min_df=args.min_df,
                                   apply_zero_index=False)
