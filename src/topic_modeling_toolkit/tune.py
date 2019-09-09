#!/usr/bin/env python

from os import path
import re
import click
from topic_modeling_toolkit.patm import Tuner

#parser = argparse.ArgumentParser(description='Performs grid-search over the parameter space by creating and training topic models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

class ParameterNamesParser(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return re.compile(r'([\w\.\-]+)').findall(value)
        except:
            raise click.BadParameter(value)

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('dataset')
@click.option('--prefix', '-p', help='a custom label to prepend to every model created; useful to indicate same grouping')
@click.option('--overwrite/--no-overwrite', default=False, show_default=True, help='whether to overwrite existing files in case of name colision happens with the newly generated ones')
@click.option('--labeling-parameters', '-lp', cls=ParameterNamesParser, default='', help='If given, then model names use the input parameter names to  uses the Selects model(s) solely based on the names provided and ignores all other selection and sorting options below')
@click.option('--constants/--no-constants', default=False, show_default=True, help='whether to use a concatenation of the (constants) static parameter names as part of the automatically generated labels that are used to name the artifacts of the training process')
@click.option('--explorables/--no-explorables', default=True, show_default=True, help='whether to use a concatenation of the explorable parameter names as part of the automatically generated labels that are used to name the artifacts of the training process')
@click.option('--parameter-set', '-ps', type=click.Choice(['all', 'training', 'regularization']), default='training', show_default=True, help="If no 'labeling-parameters' are inputted then it can narrow down where to look for the static and explorable parameters to include in name building")
@click.option('--verbose', '-v', type=int, default=5, help='controls the amount of outputing to stdout. Sensible values are {1,2,3,4,5}')
def main(dataset, prefix, overwrite, labeling_parameters, constants, explorables, parameter_set, verbose):
    collections_dir = os.getenv('COLLECTIONS_DIR')
    if not collections_dir:
        raise RuntimeError("Please set the COLLECTIONS_DIR environment variable with the path to a directory containing collections/datasets")

    tuner = Tuner(path.join(collections_dir, dataset), {
        'perplexity': 'per',
        'sparsity-phi-@dc': 'sppd',
        'sparsity-phi-@ic': 'sppi',
        'sparsity-theta': 'spt',
        'topic-kernel-0.60': 'tk60',
        'topic-kernel-0.65': 'tk65',
        'topic-kernel-0.80': 'tk80',
        'top-tokens-10': 'top10',
        'top-tokens-100': 'top100',
        'background-tokens-ratio-0.3': 'btr3',
        'background-tokens-ratio-0.2': 'btr2'
    })

    tuner.training_parameters = [('nb-topics', 40),
                                 ('collection-passes', 100),
                                 ('document-passes', 1),
                                 ('background-topics-pct', 0.1),
                                 ('default-class-weight', 1),
                                 ('ideology-class-weight', [1, 2, 5, 10])]
    tuner.regularization_specs = [
        # ('smooth-phi', [('tau', 1.0)]),
        # ('smooth-theta', [('tau', 1.0)]),
        # ('sparse-theta', [('tau', ['0_linear_-1_-10', '2_linear_-4_-20'])]),
        # ('sparse-phi', [('tau', [1.0, 2])]),
        ('label-regularization-phi-dom-cls', [('tau', [1e3, 1e4, 1e5])]),
        # ('label-regularization-phi-dom-cls', [('tau', [1e3])]),
        ('smooth-phi-dom-cls', [('tau', 1)]),
        # ('decorrelate-phi-dom-def', [('tau', 1e4)])
    ]

    tuner.tune(prefix_label=prefix,
               append_static=constants,
               append_explorables=explorables,
               force_overwrite=overwrite,
               cache_theta=True,
               verbose=verbose,
               interactive = False,
               labeling_params=labeling_parameters,
               preserve_order=True,
               parameter_set=parameter_set
               )

if __name__ == '__main__':
    main()
