import argparse

from patm.tuning import Tuner
from patm.tuning.building import tuner_definition_builder as tdb


# def get_model_settings(label, dataset):
#     results = load_results(os.path.join(collection_dir, dataset, '/results', label + '-train.json'))
#     reg_set = results['reg_parameters'][-1][1]
#     back_topics = {}
#     domain_topics = {}
#     reg_tau_trajectories = {}
#     reg_specs = [0, {}]
#     for k,v  in reg_set.items():
#         print k
#         reg_specs[1][k] = v
#         if re.match('^smooth-(?:phi|theta)|^decorrelator-phi', k):
#             print 'm1', reg_set[k].keys()
#             back_topics[k] = reg_set[k]['topic_names']
#         if re.match('^sparse-(?:phi|theta)', k):
#             print 'm2', reg_set[k].keys()
#             domain_topics[k] = reg_set[k]['topic_names']
#             reg_tau_trajectories[k] = [_ for sublist in map(lambda x: [x[1][k]['tau']] * x[0], results['reg_parameters']) for _ in sublist]
#     for k in sorted(back_topics.keys()):
#         print k, back_topics[k]
#     for k in sorted(domain_topics.keys()):
#         print k, domain_topics[k]
#     if back_topics:
#         assert len(set(map(lambda x: '.'.join(x), back_topics.values()))) == 1  # assert that all sparsing adn smoothing regularizers
#         # have the same domain and background topics respectively to comply with the restriction in 'tune' method that all regularizers "share" the same domain and background topics.
#         assert len(set(map(lambda x: '.'.join(x), domain_topics.values()))) == 1
#         reg_specs[0] = float(len(back_topics.values()[0])) / (len(back_topics.values()[0]) + len(domain_topics.values()[0]))
#         return reg_specs, reg_tau_trajectories
#     return None

def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Performs grid-search over the parameter space by creating and training topic models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    parser.add_argument('--prefix', '-p', default='', help='a custom label to prepend to every model created; useful to indicate same grouping')
    parser.add_argument('--force-overwrite', '--f-o', action='store_true', dest='overwrite', help='whether to overwrite existing files in case of name colision happens with the newly generated ones')
    parser.add_argument('--include-constants-to-label', '--i-c', action='store_true', default=False, dest='append_static',
                        help='whether to use a concatenation of the (constants) static parameter names as part of the automatically generated labels that are used to name the artifacts of the training process')
    parser.add_argument('--include-explorables-to-label', '--i-e', action='store_true', default=False, dest='append_explorables',
                        help='whether to use a concatenation of the explorable parameter names as part of the automatically generated labels that are used to name the artifacts of the training process')
    parser.add_argument('--verbose', '-v', type=int, default=3, help='controls the amount of outputing to stdout')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()

    tuner = Tuner(args.dataset)
    tuning_definition = tdb.initialize().nb_topics([20]).collection_passes(50).document_passes([1, 3]).background_topics_pct([0.2]).\
        ideology_class_weight([5.0]). \
        sparse_phi().deactivate(2).kind(['linear']).start(-1).end([-40, -60]). \
        sparse_theta().deactivate(2).kind('linear').start([-5]).end([-25, -50]).build()

    tuner.activate_regularizers.smoothing.phi.theta.sparsing.phi.theta.done()
    # tuner.static_regularization_specs = {'smooth-phi': {'tau': 1.0},
    #                                      'smooth-theta': {'tau': 1.0},
    #                                      'sparse-theta': {'alpha_iter': 1}}
    # 'sparse-theta': {'alpha_iter': 'linear_1_4'}}

    tuner.tune(tuning_definition,
               prefix_label=args.prefix,
               append_explorables=(lambda x: 'all' if x == True else [])(args.append_explorables),
               append_static=(lambda x: 'all' if x == True else [])(args.append_static),
               force_overwrite=args.overwrite,
               verbose=args.verbose)
