import os
import argparse

from patm.tuning import Tuner


def get_model_settings(label, dataset):
    results = load_results(os.path.join(collection_dir, dataset, '/results', label + '-train.json'))
    reg_set = results['reg_parameters'][-1][1]
    back_topics = {}
    domain_topics = {}
    reg_tau_trajectories = {}
    reg_specs = [0, {}]
    for k,v  in reg_set.items():
        print k
        reg_specs[1][k] = v
        if re.match('^smooth-(?:phi|theta)|^decorrelator-phi', k):
            print 'm1', reg_set[k].keys()
            back_topics[k] = reg_set[k]['topic_names']
        if re.match('^sparse-(?:phi|theta)', k):
            print 'm2', reg_set[k].keys()
            domain_topics[k] = reg_set[k]['topic_names']
            reg_tau_trajectories[k] = [_ for sublist in map(lambda x: [x[1][k]['tau']] * x[0], results['reg_parameters']) for _ in sublist]
    for k in sorted(back_topics.keys()):
        print k, back_topics[k]
    for k in sorted(domain_topics.keys()):
        print k, domain_topics[k]
    if back_topics:
        assert len(set(map(lambda x: '.'.join(x), back_topics.values()))) == 1  # assert that all sparsing adn smoothing regularizers
        # have the same domain and background topics respectively to comply with the restriction in 'tune' method that all regularizers "share" the same domain and background topics.
        assert len(set(map(lambda x: '.'.join(x), domain_topics.values()))) == 1
        reg_specs[0] = float(len(back_topics.values()[0])) / (len(back_topics.values()[0]) + len(domain_topics.values()[0]))
        return reg_specs, reg_tau_trajectories
    return None

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
    from patm.tuning.building import tuner_definition_builder as tdb

    tuner = Tuner(args.dataset)
    tuning_definition = tdb.initialize().nb_topics([20]).collection_passes(50).document_passes(1).background_topics_pct([0.1]).\
        ideology_class_weight([5]). \
        sparse_phi().deactivate(5).kind(['linear']).start(-10).end([-20, -30]). \
        sparse_theta().deactivate(5).kind('linear').start([-5]).end([-15, -25]).build()

    tuner.activate_regularizers.smoothing.phi.theta.sparsing.phi.theta.done()
    # tuner.static_regularization_specs = {'smooth-phi': {'tau': 1.0},
    #                                      'smooth-theta': {'tau': 1.0},
    #                                      'sparse-theta': {'alpha_iter': 1}}
    # 'sparse-theta': {'alpha_iter': 'linear_1_4'}}
    print 'TUNE', (lambda x: 'all' if x == True else [])(args.append_explorables)
    tuner.tune(tuning_definition,
               prefix_label=args.prefix,
               append_explorables=(lambda x: 'all' if x == True else [])(args.append_explorables),
               append_static=(lambda x: 'all' if x == True else [])(args.append_static),
               force_overwrite=args.overwrite,
               verbose=args.verbose)



# def tune_plsa(static_params, explorable_params):
#     tuner = tuner_builder.new_plsa().add_statics(static_params).add_explorables(explorable_params).build()
#     tuner.tune([])
#     return tuner
#     # tuner = eval('tuner_builder.new_plsa().{}{}build()'.format(
#     #     ''.join(map(lambda x: 'add_static({}, {}).'.format(x[0], x[1]), static_params)),
#     #     ''.join(map(lambda x: 'add_explorable({}, {}).'.format(x[0], x[1]), explorable_params))))
# def tune_lda(static_params, explorable_params, smooth_phi_tau=1, smooth_theta_tau=1):
#     tuner = tuner_builder.new_lda().add_statics(static_params).add_explorables(explorable_params).build()
#     tuner.tune([1.0, {'smooth-phi': {'name': 'sm_p', 'tau': smooth_phi_tau}, 'smooth-theta': {'name': 'sm_t', 'tau': smooth_theta_tau}}])
#     return tuner
# def tune_smooth_n_sparse(static_params, explorable_params, prefix, background_topics_ratio=0.1, smooth_phi_tau=1, smooth_theta_tau=1):
#     tuner = tuner_builder.new(prefix_label=prefix).add_statics(static_params).add_explorables(explorable_params).build()
#     tuner.tune([background_topics_ratio, {'sparse-theta': {'name': 'spd_t'}, 'smooth-theta': {'name': 'smb_t', 'tau': smooth_theta_tau},
#                             'sparse-phi': {'name': 'spd_p'}, 'smooth-phi': {'name': 'smb_p', 'tau': smooth_phi_tau}}])
#     return tuner




    # tuner.active_regularizers = {'smooth-phi': 'smph', 'smooth-theta': 'smth', 'sparse-phi': 'spph', 'sparse-theta': 'spth'}

    # tuner.static_regularization_specs = {'smooth-phi': {'tau': 1.0},
    #                           'smooth-theta': {'tau': 1.0},
    #                           'sparse-theta': {'alpha_iter': 1}}

    #                           'sparse-theta': {'alpha_iter': 'linear_1_4'}}

    # tr1 = {'deactivation_period_pct': [0.1], 'start': [-2], 'end': [-9]} # more "aggressive" regularization
    # tr2 = {'deactivation_period_pct': [0.1], 'start': [-1, -2, -3], 'end': [-4, -4.5, -5]} # more "quite" regularization
    # tr4 = {'deactivation_period_pct': [0.1], 'start': [-3], 'end': [-5, -10]}  # more "quite" regularization
    #
    # static = [('collection_passes', 20), ('document_passes', 5)]
    # explorable = [('_nb_topics', [20, 30, 40])]
    #
    # #### MODELS
    # # t1 = tune_plsa(static, explorable)
    # t2 = tune_lda(static, explorable)
    #
    # t3 = tune_smooth_n_sparse(static.append(('sparse_theta_reg_coef_trajectory', tr1)), explorable.append(('sparse_theta_reg_coef_trajectory', tr4)), background_topics_ratio=0.1)


    #### SNIPPETS

    # ############ GRID SEARCH ############
    # tr1 = {'deactivation_period_pct': [0.1], 'start': [-0.2], 'end': [-4, -7]}
    # tr2 = {'deactivation_period_pct': [0.1], 'start': [-0.1], 'end': [-3, -5]}


    ##### PLSA + SMOOTHING + SPARSING MODELS
    # tr1 = {'deactivation_period_pct': [0.1], 'start': [-2, -3, -4], 'end': [-5, -7, -9]} # more "aggressive" regularzation
    # tr2 = {'deactivation_period_pct': [0.1], 'start': [-1, -2, -3], 'end': [-4, -4.5, -5]} # more "quite" regularzation


    ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL
    # pre_models = ['plsa+sm+sp_v076_20_5_20', 'plsa+sm+sp_v073_20_5_20']

    # lbl1 = 'plsa+sm+sp_v076_100_5_20'
    # old_model = 'plsa+sm+sp_v076_20_5_20'
    # l2 = '_20_5_20'
    #
    # reg_specs, tau_trajs = get_model_settings(old_model, args.dataset)
    # print 'REGS', reg_specs
    # print 'TRAJS', tau_trajs
    #
    # trajs = map(lambda x: ((x[0]+'_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), tau_trajs.items())


    #
    # tuner = Tuner(args.dataset, args.train_cfg, reg_cfg,
    #                   [('collection_passes', 100), ('document_passes', 5)] + trajs,
    #                   [('nb_topics', [10, 20])], # ('decorrelate_topics_reg_coef', [2e+5, 1.5e+5, 1e+5, 1e+4])],
    #                   prefix_label='cm1',
    #                   enable_ideology_labels=False,
    #                   append_explorables='all',
    #                   _append_static=True)
    # tuner.tune([0.1, {'sparse-theta': {'name': 'spd_t', 'tau': -0.1}, 'smooth-theta': {'name': 'smb_t', 'tau': 1},
    #                         'sparse-phi': {'name': 'spd_p', 'tau': -0.1}, 'smooth-phi': {'name': 'smb_p', 'tau': 1}}])
    # tuner.tune([reg_specs[0], dict(reg_specs[1], **{'decorrelator-phi': {'name': 'decd_p', 'tau': 1e+5}})])

    # ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL
    # num = [76, 46, 73]
    # # pre = ['plsa+sm+sp_v076_100_5_20', 'plsa+sm+sp_v076_100_5_20', 'plsa+sm+sp_v073_100_5_20']
    # for ml in num:
    #     lbl1 = 'plsa+sm+sp_v0{}'.format(ml)
    #     l2 = '_100_5_20'

    #     reg_specs, tau_trajs = get_model_settings(lbl1 + l2, args.dataset)
    #     trajs = map(lambda x: ((x[0] + '_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), tau_trajs.items())

    #     ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL AND ADD DECORELATION
    #     decor = {'name': 'decd_p', 'tau': 1e+5}
    #     reg_pool = [reg_specs[0], dict(reg_specs[1], **{'decorrelator-phi': decor})]

    # ##### LOAD REGULARIZATION SETTINGS FROM SELECTED MODEL AND ADD DECORELATION
    # reg_pool = [iit[0][0], dict(iit[0][1], **{'decorrelator-phi': decor})]
    # [('decorrelate_topics_reg_coef', [2e+5, 1e+5, 1e+4, 1e+3])],
    # decor = {'name': 'dec_p', 'tau': 1e+5}

    # ############ GRID SEARCH 2 ############
    # tr1 = {'deactivation_period_pct': [0.2], 'start': [-0.2, -0.5, -0.8], 'end': [-1, -4, -7]}
    # tr2 = {'deactivation_period_pct': [0.2], 'start': [-0.1, -0.3, -0.5], 'end': [-1, -3, -5]}
    # decor = {'name': 'dec_p', 'tau': 1e+5}
    #
    # # iit = get_model_settings('plsa+ssp+sst_v0{}_100_5_20-train.json'.format(vold))
    # iit = get_model_settings('t3-plsa+ssp+sst+dec_v004_100_5_20')
    #
    # trajs = map(lambda x: ((x[0]+'_reg_coef_trajectory').replace('-', '_'), trajectory_builder.create_tau_trajectory(x[1])), iit[1].items())
    #
    # reg_pool = [iit[0][0], dict(iit[0][1], **{'decorrelator-phi': {'name': 'dec_p', 'tau': 1e+5}})]

