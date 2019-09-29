#!/usr/bin/env python

import click
from topic_modeling_toolkit.reporting import TopicsHandler


@click.command()
@click.option('--dataset', '-d', required=True, prompt="Which i the dataset the model was trained on? (input dataset string label)",
              help="The dataset that was used to train the model on.")
@click.option('--model-label', '-m-l', required=True, prompt="Which model do you want to query for its topics? (input model label)",
              help="The model label to use searching for stored experimental results.")
@click.option('--topics-set', '-t-s', default='domain', show_default=True, type=click.Choice(["background", "domain"]),
              help="Common lexis should be collected in 'background' topics. 'Domain' topics should be free of common lexis.")
@click.option('--tokens-type', '-t-t', default='top-tokens', show_default=True,
              help="'top-tokens' is a list sorted on p(w|t). 'kernel' is a list sorted on p(t|w); should be accompanied by "
                   "threshold, ie 'kernel60' -> 0.60, 'kernel25' -> 0.25")
@click.option('--sort', '-s', default='name', show_default=True,
              help="Reports back the list of topics sorted on the metric. 'name': alphabetically by name, 'coh': by kernel "
                   "coherence, 'con': by kernel contrast, 'pur': by kernel purity. The last 3 options require a threshold similar to the "
                   "'tokens-type' arguments. Example syntaxes are: 'coh-80', 'con-25', 'pur-90'.")
@click.option('--columns', '-c', default=10, show_default=True,
              help="The number of columns (each corresponding to a topic's tokens group) to include per row'")
@click.option('--number-of-tokens', '-nb-tokens', default=15, show_default=True,
              help="The maximum number of tokens to show per topic. If requested background tokens to report then this "
                   "argument correspond to the total amount of bg tokens to show.")
@click.option('--show_metrics/--no-show_metrics', show_default=True, default=True,
              help="Whether to print kernel coherence, contrast and purity for each individual topic. It requires a kernel "
                   "definition (threshold) to be inputted from '--tokens-type' or '--sort', else it has no effect.")
@click.option('--show_title/--no-show_title', show_default=True, default=False,
              help="Whether to print a title on top of the table of topics ")
def main(dataset, model_label, topics_set, tokens_type, sort, columns, number_of_tokens, show_metrics, show_title):
    collections_dir = os.getenv('COLLECTIONS_DIR')
    if not collections_dir:
        raise RuntimeError("Please set the COLLECTIONS_DIR environment variable with the path to a directory containing collections/datasets")
    topic_handler = TopicsHandler(collections_dir)
    if topics_set == 'background':
        b = topic_handler.pformat_background([dataset, model_label],
                                             columns=columns,
                                             nb_tokens=number_of_tokens,
                                             show_title=show_title)
    else:
        b = topic_handler.pformat([dataset, model_label],
                                  topics_set,
                                  tokens_type,
                                  sort,
                                  number_of_tokens,
                                  columns,
                                  topic_info=show_metrics,
                                  show_title=show_title)
    print(b) # '--s_m/--no-s_m'

if __name__ == '__main__':
    main()
