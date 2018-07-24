import os
import sys
import json
import argparse

results_root = 'results'
models_root = 'models'
collections_dir = '/data/thesis/data/collections'


def load_trackables(path_file):
    with open(path_file, 'r') as results_file:
        results_text = results_file.read()
    return json.loads(results_text)['trackables']

def get_latest_top_tokens(trackables_dict):
    score_types = ['top-tokens' + str(_) for _ in ['-100', '-10', '']]
    for token_score in score_types:
        if token_score in trackables_dict:
            return sorted(eval(trackables_dict[token_score]['tokens'][-1]).items(), key=lambda x: x[0])
    return None

def get_table(caption, column_headers, row_headers, data_row_entries, title=None):
    """
    Generates the equivalent latex code of a table encoded by the input parameters.

    :param caption: the string to be used as a caption for the table.
    :param column_headers: a list of elements being either strings or tuples with first element a string and second the
        number of columns it spans. This data structure serves as potential headers of the columns of the table.
    :param row_headers: a list of (strictly) the first element of every row
    :param data_row_entries: all the rows to appear after the table's column headers. Supports integers, strings and
        tuples with the same format as above (element, number of columns it spans).
    :param title: the string to be used as the top row of the table spanning all columns representing the table's
        desired title.
    """
    all_row_entries = []
    rules = []
    # print str(data_row_entries)
    num_columns = column_span(data_row_entries[0]) + 1
    if title is not None:
        rules.append('mid')
    rules.append('bottom')
    rules.extend(['' for _ in xrange(len(data_row_entries) - 1)])
    rules.append('mid')
    if title is not None:
        all_row_entries.append([(title, num_columns)])
    if column_span(column_headers) == column_span(data_row_entries[0]):
        all_row_entries.append([''] + column_headers)
    else:
        all_row_entries.append(column_headers)
    for row_label, data_row in zip(row_headers, data_row_entries):
        if type(data_row) is not list:
            data_row = [data_row]
        all_row_entries.append([row_label] + data_row)
    table = get_latex_table(caption, all_row_entries, rules=rules)
    return table


def get_latex_table(caption, row_entries, caption_pos='below', rules=None):
    """
    Generates latex code to create a table given the input parameters.\n
    :param str caption: the text to be used as a caption for the table
    :param row_entries: a list of lists. Each inner list holds the table's data for a specific row from top to bottom.
        Each of these lists can contain a mixture of single variables (numerical/alpharithmetic) and tuples. See transform_row_entry for more.\n
    :param str caption_pos: position of caption. Can be either 'bottom' or 'below'.
    :param list rules: list with rules corresponding to each line in row_entries. Elements of the list can be one of None, 'top', 'mid', 'bottom'.
    :return the generated latex code representing the table.
    :rtype: str
    """
    # print '\n', str(row_entries), '\n'
    table = ''
    num_columns = column_span(row_entries[0])
    if rules is None:
        rules = ['' for _ in row_entries]
    table += "\\begin{table}[]\n\\centering\n"
    if caption_pos == 'above':
        table += "\\caption{" + caption + "}\n"
        table += "\\label{Please fill in the label}\n"
    table += "\\begin{tabular}{@{}" + 'l'*num_columns + "@{}}\n\\toprule\n"
    for row, rule in zip(row_entries, rules):
        table += transform_row_entry(row, rule=rule)
    table += '\\end{tabular}\n'
    if caption_pos == 'below':
        table += '\\caption{' + caption + '}\n'
        table += "\\label{Please fill in the label}\n"
    table += '\\end{table}'
    return table


def transform_row_entry(data_structure, rule=''):
    """
    Transforms the input list into the latex code that will populate a latex table's to form a row according to the information packed in the input data_structure. Input contains the sequence of row entries from left to right order, along with information about spanning and alignment. The input list supports having both single numerical or alpharithmetic
    variables as row entries, as well as tuples of 3 or 2 elements. Three-element tuples have as 1st element the actual
    row entry, as 2nd the number of desired number of columns to span and as 3rd the cell alignment; one of 'l', 'c', 'r'. Two-element tuples have as 1st element the row entry and as 2nd either the column span or the alignment. Intuitively, single variables in the input list span one column and align as defined by the column they belong by default.

    :param list entry: A list of a mixture of single variables (arithmetic/alpharithmetic) and tuples. The tuples will have
        as first element a numerical or alpharithmetic value followed by either one or two more elements to include
        information about column span and/or alignment. For the single elements appearing in the iterable it is
        explicitly assumed that they span a single column and inherit the alignmet of the column they belong.\n
        * Eg1 [("This is the table's title"), 3, 'c')]; a row with the title text, aligned centrally and spanning 3 columns.\n
        * Eg2 [5.34, 5.50, 3.59, 7.81]; a four column row with each item aligning as defined by the  column it belongs.\n
        * Eg3 [("label description", 2, 'l'), (0.91, 'c'), (0.76, 'c')]; a row with 3 entries: the first is a 2-column
        alpharithmetic aligned left and the rest 2 are numerical entries aligned centrally.
    :param str rule: can be one of 'top', 'mid', 'bottom'
    :return: the latex code that generates a table's row with the input data
    :rtype: str
    """
    if rule != '':
        rule = '\\' + rule + 'rule'
    row = ''
    for i, item in enumerate(data_structure):
        if type(item) == tuple:
            if len(item) == 1:
                row += str(item[0]) + ' & '
            elif len(item) == 2:
                if type(item[1]) == str:
                    row += r'\multicolumn{' + str(item[0]) + '}{' + item[1] + '}{' + '1' + '} & '
                else:  # if type(item[1]) == int
                    row += r'\multicolumn{' + str(item[0]) + '}{' + 'c' + '}{' + str(item[1]) + '} & '
            else:  # len(item) == 3
                row += r'\multicolumn{' + str(item[0]) + '}{' + item[2] + '}{' + str(item[1]) + '} & '
        else:
            row += str(item) + ' & '
    row = row[:len(row) - 2]
    row += r'\\ ' + rule + '\n'
    return row


def column_span(row_entry):
    length = 0
    for el in row_entry:
        if type(el) == tuple and type(el[1]) == int:
            length += el[1]
        else:
            length += 1
    return length


def create_topics_tokens_table(caption, topic_names, tokens_lists, nb_tokens=10, width=5, math=False):
    methods = [lambda x: x.replace('_', '\_'), lambda x: '$' + x[:x.index('_') + 1] + '{' + x[x.index('_') + 1:] + '}$']
    max_tokens_len = max(len(_) for _ in tokens_lists)
    def get_token_str(index, token_list):
        if index < len(token_list):
            return token_list[index]
        else:
            return ' '
    nb_topics = len(topic_names)
    rows = []
    rules = []
    processed = 0
    while processed < nb_topics:
        rows.append(map(lambda x: methods[1](x) if math else methods[0](x), topic_names[processed:processed+width]))
        rules.append('bottom')
        for row_index in range(nb_tokens):
            if row_index == max_tokens_len: # in case nb_tokens requested exceeds the maximum token recording capacity
                break
            current_token_lists = tokens_lists[processed:processed + width]
            row_token_list = map(lambda x: get_token_str(row_index, x), current_token_lists)
            rows.append(row_token_list)
            rules.append('')
        if processed + width > nb_topics:
            processed = nb_topics
        else:
            processed += width
        if processed < nb_topics:
            rows.append([' '] * width)
            rules.append('top')
    table = get_latex_table(caption, rows, caption_pos='below', rules=rules)
    return table


class TopicReporter:
    def __init__(self, collections_root_dir):
        self.collections_dir = collections_root_dir
        self.results_root = 'results'
        self._topics = []
        self._model_path = ''
        self._nb_tokens = 0
        self._max_token_lens = []
        self._max_headers = []
        # self.quality_metrics = ['']
        # self._topic_quality_metrics = {'kernel-coherence': 'coherence', 'kernel-contrast': 'contrast', 'kernel-purity': 'purity'}
        self._transf_metrics = {'coh': 'coherence', 'con': 'contrast', 'pur': 'purity'}
        self._quality_labels = ['coh', 'con', 'pur']

    def get_topics_string(self, collection_name, model_name, metric='coherence', nb_tokens=10, top_n_topics=3):
        self._model_path = os.path.join(self.collections_dir, collection_name, self.results_root, model_name + '-train.json')
        if metric =='all':
            self.get_top_topic_per_metric(nb_tokens)
        else:
            self.get_top_topics(metric, nb_tokens, top_n_topics)
        self._max_token_lens = list(map(lambda x: max(map(len, x[1])), self._topics))
        header = self._get_header()
        body = self._get_rows()
        return header + body

    def get_top_topics(self, metric, nb_tokens, top_n_topics):
        self._topics = self._get_topics(self._model_path, metric, nb_tokens, top_n_topics)

    def get_top_topic_per_metric(self, nb_tokens):
        self._topics = []
        for metric in self._quality_labels:
            self._topics.extend(self._get_topics(self._model_path, self._transf_metrics[metric], nb_tokens, 1))

    def _get_topics(self, model_path, metric, nb_tokens, nb_topics):
        self._nb_tokens = nb_tokens
        self._tracked = load_trackables(model_path)
        return list(map(lambda x: (x[0], x[1][:nb_tokens]), sorted(eval(self._tracked['top-tokens-100']['tokens'][-1]).items(),
                                                              key=lambda x: eval(self._tracked['topic-kernel'][metric][-1])[x[0]],
                                                              reverse=True)[:nb_topics]))

    def _get_header(self):
        self._headers = list(map(lambda y: [y[0]] + list(map(lambda x: '{}:{:.4f}'.format(x, self._report_quality(self._transf_metrics[x], y[0])), self._quality_labels)), self._topics))
        self._max_headers = list(map(lambda x: max(map(lambda y: len(y), x)), self._headers))
        gaps = []
        for i, el in enumerate(map(lambda x: [h[x] for h in self._headers], range(4))): # 1..4: one per row in header
            gaps.append([max(self._max_token_lens[j], self._max_headers[j])-len(self._headers[j][i]) if self._max_token_lens[j] > len(self._headers[j][i]) else 0 for j, topic in enumerate(self._headers)])
        b = ''
        for i, el in enumerate(map(lambda x: [h[x] for h in self._headers], range(4))):
            b += '{}\n'.format('  '.join(map(lambda x: '{}{} '.format(x[1], ' '*(gaps[i][x[0]])), enumerate(el))))  # el, ' '*(1+list(gaps)[i])) + '\n'
        b += '{}\n'.format('-' * sum(map(lambda x: max(self._max_headers[x[0]], self._max_token_lens[x[0]]) + 3, enumerate(self._headers))))
        return b

    def _report_quality(self, metric, topic):
        return eval(self._tracked['topic-kernel'][metric][-1])[topic]

    def _get_rows(self):
        b = ''
        for i in range(self._nb_tokens):
            try:
                b += ' | '.join(['{}{}'.format(self._get_token_string(j, i), ' '*(max(self._max_token_lens[j], self._max_headers[j]) - len(str(self._get_token_string(j, i))))) for j, g in enumerate(self._topics)]) + '\n'
            except IndexError as ie:
                print(ie, i)
        return b

    def _get_token_string(self, topic_index, token_index):
        if token_index < len(self._topics[topic_index][1]):
            return self._topics[topic_index][1][token_index]
        return ''


def _assert_natural_order_is_descending(trackables_dictionary):
    tt = eval(trackables_dictionary['top-tokens-100']['tokens'][-1])
    tw = eval(trackables_dictionary['top-tokens-100']['weights'][-1])
    c1 = list(map(lambda x: (x[0], x[1]), zip(tt['top_00'], tw['top_00'])))
    assert sorted(c1, key=lambda x: x[1], reverse=True) == c1
    c1 = list(map(lambda x: (x[0], x[1]), zip(tt['top_03'], tw['top_03'])))
    assert sorted(c1, key=lambda x: x[1], reverse=True) == c1
    c1 = list(map(lambda x: (x[0], x[1]), zip(tt['top_09'], tw['top_09'])))
    assert sorted(c1, key=lambda x: x[1], reverse=True) == c1

def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Creates various components for reporting on inffered topics', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', metavar='name', help='the name of the collection on which experiments were conducted')
    parser.add_argument('model', metavar='label', help='the name of model that was trained to infer topics')
    parser.add_argument('--topic-tokens', '-tt', dest='topic_tokens', action='store_true', help='enable creating table in latex code, showing top tokens per topic, according to p(w|t)')
    parser.add_argument('--topics', '-t', dest='nb_topics', type=int,  default=5, help='report on the top n topics sorted by the quality metric selected')
    parser.add_argument('--num-tokens', '-nt', dest='nb_tokens', type=int, default=10, help='report on the top n tokens sorted by the quality metric selected')
    parser.add_argument('--metric', '-m', default='coherence', help="sort the retrieved topics according to this metric. If 'all' is given reports the top quality topic per metric")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_cli_arguments()
    col_root = '/data/thesis/data/collections/'
    tracked_metrics_dict = load_trackables(os.path.join(col_root, args.collection, 'results', args.model + '-train.json'))
    _assert_natural_order_is_descending(tracked_metrics_dict)
    # top_toks = get_latest_top_tokens(tracked_metrics_dict)
    tr = TopicReporter(col_root)
    st = tr.get_topics_string('arts', 'cm1_100_5_20', metric=args.metric, nb_tokens=args.nb_tokens, top_n_topics=args.nb_topics)
    print(st)

    # table = create_topics_tokens_table('Top tokens per topic', map(lambda k: k[0], top_toks), map(lambda z: z[1], top_toks), width=4, math=True)
    # print table
