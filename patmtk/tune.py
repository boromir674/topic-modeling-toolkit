#!/usr/bin/env python

import argparse
from patm.tuning import Tuner
from patm.tuning.building import tuner_definition_builder as tdb

def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Performs grid-search over the parameter space by creating and training topic models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('--prefix', '-p', default='', help='a custom label to prepend to every model created; useful to indicate same grouping')
    parser.add_argument('--force-overwrite', '--f-o', action='store_true', dest='overwrite', help='whether to overwrite existing files in case of name colision happens with the newly generated ones')
    parser.add_argument('--include-constants-to-label', '--i-c', action='store_true', default=False, dest='append_static',
                        help='whether to use a concatenation of the (constants) static parameter names as part of the automatically generated labels that are used to name the artifacts of the training process')
    parser.add_argument('--include-explorables-to-label', '--i-e', action='store_true', default=False, dest='append_explorables',
                        help='whether to use a concatenation of the explorable parameter names as part of the automatically generated labels that are used to name the artifacts of the training process')
    parser.add_argument('--verbose', '-v', type=int, default=3, help='controls the amount of outputing to stdout. Sensible values are {1,2,3,4,5}')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_cli_arguments()
    from patm.definitions import COLLECTIONS_DIR_PATH
    from os import path
    tuner = Tuner(path.join(COLLECTIONS_DIR_PATH, args.dataset))
    tuning_definition = tdb.initialize()\
        .nb_topics(20)\
        .collection_passes(50)\
        .document_passes(1)\
        .background_topics_pct(0.2) \
        .ideology_class_weight(1, 5) \
        .build()

        # .sparse_phi()\
        #     .deactivate(8)\
        #     .kind('linear')\
        #     .start(-1)\
        #     .end(-10, -100)\
        # .sparse_theta()\
        #     .deactivate(10)\
        #     .kind('linear')\
        #     .start(-1)\
        #     .end(-10, -100)\
    tuner.active_regularizers = [
        # 'smooth-phi',
        # 'smooth-theta',
        'label-regularization-phi-dom-cls',
        # 'smooth-phi-dom-cls'   # ENABLE TO smooth the the p(class=c|topic=t) distributions. Crucial for computing the symmetric KL divergence; eg KL(p(c1|t)||p(c2|t)
        # 'label-regularization-phi-dom-def',
        # 'decorrelate-phi-domain',
        # 'decorrelate-phi-background'
    ]
    tuner._default_regularizer_parameters = {'smooth-phi': {'tau': 1.0},
                                            'smooth-theta': {'tau': 1.0, 'alpha_iter': 1.0},
                                            'sparse-phi': {'tau': 'linear_-5_-15', 'start': 4},
                                            'sparse-theta': {'alpha_iter': 1, 'tau': 'linear_-3_-13', 'start': 4},
                                            'label-regularization-phi': {'tau': 1.0},
                                            'label-regularization-phi-dom-def': {'tau': 1e5},
                                            'label-regularization-phi-dom-cls': {'tau': 1e5},
                                            'decorrelate-phi-def': {'tau': 10000},
                                            'decorrelate-phi-dom-def': {'tau': 10000},
                                            'decorrelate-phi-class': {'tau': 10000},
                                            'decorrelate-phi-domain': {'tau': 10000},
                                            'decorrelate-phi-background': {'tau': 10000},
                                            'improve-coherence': {'tau': 1.0},
                                            'smooth-phi-dom-cls': {'tau': 1}}
    tuner.tune(tuning_definition,
               prefix_label=args.prefix,
               append_explorables=args.append_explorables,
               append_static=args.append_static,
               force_overwrite=args.overwrite,
               verbose=args.verbose)

    #LDA
    # tuner.activate_regularizers.smoothing.phi.theta.done()
    # DLDA
    # tuner.activate_regularizers.smoothing.phi.theta.decorrelate_phi_all.done()
    # CLDA
    # "label-regularization-phi-def"
    # tuner.activate_regularizers.smoothing.phi.theta.label_regularization.done()


    #ILDA
    # tuner.activate_regularizers.smoothing.phi.theta.improve_coherence_phi.done()

    #DCLDA
    # tuner.activate_regularizers.smoothing.phi.theta.decorrelate_phi_domain.label_regularization.done()

    # SLDA
    # tuner.activate_regularizers \
    #     .smoothing \
    #         .phi \
    #         .theta \
    #     .sparsing \
    #         .theta \
    #         .phi \
    # .done()

    # SCLDA
    # tuner.activate_regularizers \
    #     .smoothing \
    #         .phi \
    #         .theta \
    #     .label_regularization \
    #     .sparsing \
    #         .theta \
    #         .phi \
    # .done()

    #     .label_regularization \
    # .done()
        # .sparsing\
        #     .phi\
        #     .theta\
    # .done()

    #     .label_regularization\
    # .done()


    # tuner.regularizers_data = {'smooth-phi': {'tau': 1.0},
    #                                      'smooth-theta': {'tau': 1.0},
    #                                      'sparse-theta': {'alpha_iter': 1}}
    # 'sparse-theta': {'alpha_iter': 'linear_1_4'}}
