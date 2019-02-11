#!/usr/bin/python3

import click

from reporting import topic_handler


@click.command()
@click.option('--dataset', '-d', required=True, prompt="Which i the dataset the model was trained on? (input dataset string label)",
              help="The dataset that was used to train the model on.")
@click.option('--model-label', '-m-l', required=True, prompt="Which model do you want to query for its topics? (input model label)",
              help="The model label to use searching for stored experimental results.")
@click.option('--topics-set', '-t-s', required=True, default='domain', show_default=True, type=click.Choice(["background", "domain"]),
              help="Common lexis should be collected in 'background' topics. 'Domain' topics should be free of common lexis.")
@click.option('--tokens-type', '-t-t', default='top-tokens', show_default=True,
              help="'top-tokens' is a list sorted on p(w|t). 'kernel' is a list sorted on p(t|w); should be accompanied by "
                   "threshold, ie 'kernel60' -> 0.60, 'kernel25' -> 0.25")
@click.option('--sort', '-s', default='name', show_default=True,
              help="Reports back the list of topics sorted on the metric. 'name': alphabetically by name, 'coh': by kernel "
                   "coherence, 'con': by kernel contrast, 'pur': by kernel purity. The last 3 options require a threshold similar to the "
                   "'tokens-type' arguments. Example syntaxes are: 'coh-80', 'con-25', 'pur-90'.")
@click.option('--columns', '-c', default=10, show_default=True, help="The number of columns (each corresponding to a topic's tokens group) to include per row'")
@click.option('--number-of-tokens', '-nb-tokens', default=15, show_default=True, help="The maximum number of tokens to show per topic")
def main(dataset, model_label, topics_set, tokens_type, sort, columns, number_of_tokens):
    b = topic_handler.pformat([dataset, model_label],
                              topics_set,
                              tokens_type,
                              sort,
                              number_of_tokens,
                              columns)
    print(b)

if __name__ == '__main__':
    main()
