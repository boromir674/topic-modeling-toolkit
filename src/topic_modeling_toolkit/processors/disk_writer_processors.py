from .processor import BaseDiskWriter, BaseDiskWriterWithPrologue


class UciFormatWriter(BaseDiskWriterWithPrologue):
    """
    Injests a doc_vector at a time. For example [('2', 1), ('15', 4), ('18', 5), ('11', 3)]
    """
    def __init__(self, fname='/data/thesis/data/myuci'):
        super(UciFormatWriter, self).__init__(fname, lambda x: write_uci(self.file_handler, map(lambda word_id_weight_tuple: (word_id_weight_tuple[0]+1, word_id_weight_tuple[1]), x), self.doc_num))
        # super(UciFormatWriter, self).__init__(fname, lambda x: write_uci(self.fname, x, self.doc_num))
        # + 1 to align with the uci format, which states that the minimum word_id = 1; implemented with map-lambda combo on every element of a document vector

    def to_id(self):
        return 'uci'

class VowpalFormatWriter(BaseDiskWriter):
    """
    Injests a (doc_vector, classes_ditct) at a time. For example ([('_builder', 1), ('alpha', 4)], {'author': 'Ivan Sokolov'})
    """
    def __init__(self, fname='/data/thesis/data/myvowpal'):
        super(VowpalFormatWriter, self).__init__(fname, lambda x: write_vowpal_v2(self.file_handler, x[0], self.doc_num, x[1]))

    def to_id(self):
        return 'vowpal'


def write_uci(file_handler, doc_vector, doc_num):
    file_handler.writelines(map(lambda x: '{} {} {}\n'.format(doc_num, x[0], x[1]), doc_vector))


def write_vowpal(file_handler, doc_vector, doc_num, class_labels):
    """
    Dumps a doument vector as a single line in the specified target file path in the "Vowpal Wabbit" format.\n
    :param str file_handler: a file object
    :param iterable doc_vector: the representation of a document; an iterable of (token, frequency) tuples; eg [('_builder', 1), ('alpha', 4)]
    :param int doc_num: number to represent document number in queue; eg 1,2,3,4,5 ... D
    :param dict class_labels: keys are class "category" (i.e. 'author') and values are the actual class the document belongs to (i.e. 'Ivan Sokolov')
    """
    file_handler.write('doc{} {} {}\n'.format(
        doc_num,
        # ' '.join(map(lambda x: '{}{}'.format(x[0].encode('utf-8'), freq2string.get(x[1], ':{}'.format(x[1]))), doc_vector)),
        ' '.join(map(lambda x: '{}{}'.format(x[0], freq2string.get(x[1], ':{}'.format(x[1]))), doc_vector)),
        ' '.join(map(lambda x: '|{} {}'.format(x[0], x[1]), class_labels.items()))
    ))
freq2string = {1: ''}

def write_vowpal_v2(file_handler, doc_vector, doc_num, class_labels):
    """
    Dumps a document vector as a single line in the specified target file path in the "Vowpal Wabbit" format.\n
    :param str file_handler: a file object
    :param iterable doc_vector: the representation of a document; an iterable of (token, frequency) tuples; eg [('_builder', 1), ('alpha', 4)]
    :param int doc_num: number to represent document number in queue; eg 1,2,3,4,5 ... D
    :param dict class_labels: keys are class "category" (i.e. 'author') and values are the actual class the document belongs to (i.e. 'Ivan Sokolov')
    """
    file_handler.write('doc{} 1.0 {} |@default_class {}\n'.format(
        doc_num,
        ' '.join(map(lambda x: '|{} {}'.format(x[0], ' '.join([str(_) for _ in modality_features[type(x[1])](x[1])])), class_labels.items())),
        # ' '.join(map(lambda x: '{}{}'.format(x[0].encode('utf-8'), freq2string.get(x[1], ':{}'.format(x[1]))), doc_vector)),
        ' '.join(map(lambda x: '{}{}'.format(x[0], freq2string.get(x[1], ':{}'.format(x[1]))), doc_vector)),
    ))
modality_features = {str: lambda x: [x],
                     list: lambda x: x}
