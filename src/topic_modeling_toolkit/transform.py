#!/usr/bin/env python

import argparse
import os
import sys
import re
from PyInquirer import prompt

from topic_modeling_toolkit.patm import PipeHandler, CoherenceFilesBuilder, political_spectrum


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


def ask_discreetization(spectrum, pipe_handler, pool_size=100, prob=0.3, max_generation=100):

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
        },
        {
            'type': 'input',
            'name': 'custom-class-names',
            'message': 'Give space separated class names',
            'when': lambda x: x.get('naming-scheme', None) == 'Create custom names',
        }
    ]
    answers = prompt(questions)
    if not all(x in answers for x in ['discreetization-scheme']):
        raise KeyboardInterrupt

    if answers['discreetization-scheme'] == 'Create new':
        evolution_specs = ask_evolution_specs()
        print("Evolving discreetization scheme ..")
        class_names = _class_names(answers.get('custom-class-names', answers['naming-scheme']))
        spectrum.init_population(class_names, pipe_handler.outlet_ids, pool_size)
        return spectrum.evolve(int(evolution_specs['nb-generations']), prob=float(evolution_specs['probability']))
    else:
        for scheme_name, scheme in spectrum:
            if ' '.join(scheme.class_names) in answers['discreetization-scheme']:
                return scheme
    # return spectrum[answers['discreetization-scheme']]


def _class_names(string):
    return ['{}_Class'.format(x) for x in re.sub(r'\d+:\s+', '', string).split(' ')]

def ask_persist(pol_spctrum):
    return prompt([{'type': 'confirm',
                    'name': 'create-dataset',
                    'message': "Use scheme [{}] with resulting distribution [{}]?".format(' '.join(pol_spctrum.class_names), ', '.join('{:.2f}'.format(x) for x in political_spectrum.class_distribution)),
                    'default': True}])['create-dataset']


def ask_evolution_specs():
    return prompt([{'type': 'input',
                    'name': 'nb-generations',
                    'message': 'Give the number of generations to evolve solution (optimize)',
                    'default': '50'},
                    {'type': 'input',
                     'name': 'probability',
                     'message': 'Give mutation probability',
                     'default': '0.35'}])

def what_to_do():
    return prompt([{
        'type': 'list',  # navigate with arrows through choices
        'name': 'to-do',
        'message': 'How to proceed?',
        'choices': ['Evolve more', 'Use scheme to persist dataset', 'back'],
    }])['to-do']


def main():
    args = get_cl_arguments()
    nb_docs = args.sample
    if nb_docs != 'all':
        nb_docs = int(nb_docs)
    collections_dir = os.getenv('COLLECTIONS_DIR')
    if not collections_dir:
        raise RuntimeError(
            "Please set the COLLECTIONS_DIR environment variable with the path to a directory containing collections/datasets")

    ph = PipeHandler()
    ph.process(args.config, args.category, sample=nb_docs, verbose=True)
    political_spectrum.datapoint_ids = ph.outlet_ids

    while 1:
        try:
            scheme = ask_discreetization(political_spectrum, ph, pool_size=100, prob=0.3, max_generation=100)
        except KeyboardInterrupt:
            print("Exiting ..")
            sys.exit(0)
        print("Scheme with classes: [{}]".format(' '.join(x for x, _ in scheme)))
        try:
            political_spectrum.discreetization_scheme = scheme
        except ValueError as e:
            raise ValueError("{}. {}".format(e, type(scheme).__name__))

        print("Scheme [{}] with resulting distribution [{}]".format(' '.join(political_spectrum.class_names), ', '.join(
            '{:.2f}'.format(x) for x in political_spectrum.class_distribution)))
        print("Bins: {}".format(' '.join('[{}]'.format(', '.join(class_bin) for _, class_bin in scheme))))
        while 1:
            answer = what_to_do()
            if answer == 'back':
                break
            if answer == 'Evolve more':
                evolution_specs = ask_evolution_specs()
                print("Evolving discreetization scheme ..")
                scheme = political_spectrum.evolve(int(evolution_specs['nb-generations']),
                                                   prob=float(evolution_specs['probability']))
                political_spectrum.discreetization_scheme = scheme
                print("Scheme [{}] with resulting distribution [{}]".format(' '.join(scheme.class_names),
                                                                            ', '.join('{:.2f}'.format(x) for x in
                                                                                      political_spectrum.class_distribution)))
                print("Bins: {}".format(
                    ' '.join('[{}]'.format(', '.join(outlet for outlet in class_bin) for _, class_bin in scheme))))
            else:
                uci_dt = ph.persist(os.path.join(collections_dir, args.collection),
                                    political_spectrum.poster_id2ideology_label, political_spectrum.class_names,
                                    add_class_labels_to_vocab=not args.exclude_class_labels_from_vocab)
                print(uci_dt)
                print("Discreetization scheme\n{}".format(political_spectrum.discreetization_scheme))

                # print("Add the below to the DISCREETIZATION_SCHEMES_HASH")
                # print("[{}]".format())
                print('\nBuilding coocurences information')
                coherence_builder = CoherenceFilesBuilder(os.path.join(collections_dir, args.collection))
                coherence_builder.create_files(cooc_window=args.window,
                                               min_tf=args.min_tf,
                                               min_df=args.min_df,
                                               apply_zero_index=False)
                sys.exit(0)


if __name__ == '__main__':
    main()

    # if ask_persist(political_spectrum):
    #
    #         break
    #
    # print(uci_dt)
    #
    # print('\nBuilding coocurences information')
    # coherence_builder = CoherenceFilesBuilder(os.path.join(collections_dir, args.collection))
    # coherence_builder.create_files(cooc_window=args.window,
    #                                min_tf=args.min_tf,
    #                                min_df=args.min_df,
    #                                apply_zero_index=False)
