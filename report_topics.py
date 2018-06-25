import os
import sys
import json
import argparse


def load_results(path_file):
    with open(path_file, 'r') as results_file:
        results_text = results_file.read()
    return json.loads(results_text)


def get_latest_top_tokens(results_dict):
    score_types = ['top-tokens' + str(_) for _ in ['-100', '-10', '']]
    for token_score in score_types:
        if token_score in results_dict['trackables']:
            return sorted(eval(results_dict['trackables'][token_score]['tokens'][-1]).items(), key=lambda x: x[0])
    return None

def _dictify_results(results):
    tr = results['trackables']
    for k, v in tr.items():
        if type(v) == str:
            tr[k] = eval(v)
        elif type(v) == dict:
            for in_k, in_v in v.items():
                if type(in_v) == list:
                    try:
                        # print(type(constructor[in_k]))
                        v[in_k] = [eval(_) for _ in in_v]
                        # print(type(constructor[in_k]))
                    except RuntimeError as e:
                        print(e)
    return results



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


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Creates various components for reporting on inffered topics', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', metavar='name', help='the name of the collection on which experiments were conducted')
    parser.add_argument('model', metavar='name', help='the name of model used to infer topics')
    parser.add_argument('--topic-tokens', '-tt', dest='topic_tokens', action='store_true', help='enable creating table showing top tokens per topic, according to p(w|t)')
    parser.add_argument('--topics', '-t', metavar='indices', type=list, default='all', help='filter topics to consider')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    col_root = '/data/thesis/data/collections/'
    res_dict = load_results(os.path.join(col_root, args.collection, 'results', args.model + '-train.json'))
    top_toks = get_latest_top_tokens(res_dict)

    table = create_topics_tokens_table('Top tokens per topic', map(lambda k: k[0], top_toks), map(lambda z: z[1], top_toks), width=4, math=True)

    print table
    

    # llt = map(lambda x: map(lambda y: str(y), x[1]), sorted([(topic_name, str(tok)) for topic_name, top_tokens in tok_dic.items() for tok in top_tokens], key=lambda x: x[0]))

    # tt = eval(res_dict['trackables']['top-tokens-100']['tokens'][-1])
    # tw = eval(res_dict['trackables']['top-tokens-100']['weights'][-1])

    # print tt['top_01']
    # print tw['top_01']

    # tuples = []
    # for top in tt.keys():
    #     tuples.append(sorted([(token, score) for token, score in zip(tt[top], tw[top])], key=lambda x: x[1], reverse=True))  # descending order accoring to weight: p(w|t)
    # for l in tuples:
    #     print
    #     print [x[0] for x in l]
    #     print
    #     print [_ for _ in map(lambda y: y[1], top_toks)]
    #     print
    #     assert [x[0] for x in l] == [_ for _ in map(lambda y: y[1], top_toks)]



    # rows = [[("Trial row", 2), '50'], ['alpha', 'beta', 'gamma']]
    # tab = get_latex_table('Test table', rows)
    # print tab, '\n'
    #
    # rows1 = [[0.3, 0.6, 0.6, 0.6],
		# 	[0.4, 0.5, 0.6, 0.7],
		# 	[0.3, 0.4, 0.7, 0.8]]
    # caption = 'Performance comparison'
    # col_h = ['pre', 'rec']
    # row_h = ['autoenc', 'base0', 'base1']
    # data = [[0.3, (0.5, 'r')], [0.4, (0.5, 'r')], [0.7, (0.8, 'r')]]
    # table = get_table(caption, col_h, row_h, data)
    # print table
    # get_latex_table(caption, )

    # table = create_topics_tokens_table('Top tokens per topic', ['topic{}'.format(i) for i in range(3)], [['a', 'b', 'c'], ['r', 't', 'y'], ['sdf', 't', 'as']], width=2)



