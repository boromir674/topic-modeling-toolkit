from processor import StateLessProcessor


class StateLessDiskWriter(StateLessProcessor):
    """Ingests one doc vector at a time"""
    def __init__(self, fname, func):
        self.doc_num = 0
        self.fname = fname
        super(StateLessProcessor, self).__init__(func)

    # def __str__(self):
    #     return super(StateLessDiskWriter, self).__str__() + '(' + str(self.fname) + ')'

    def __str__(self):
        return type(self).__name__ + '(' + str(self.fname) + ')'

    def process(self, data):
        _ = super(StateLessProcessor, self).process(data)
        self.doc_num += 1
        return _


class UciFormatWriter(StateLessDiskWriter):
    def __init__(self, fname='/data/thesis/data/myuci'):
        super(UciFormatWriter, self).__init__(fname, lambda x: write_vector(self.fname, map(lambda word_id_weight_tuple: (word_id_weight_tuple[0]+1, word_id_weight_tuple[1]), x), self.doc_num))
        # + 1 to align with the uci format, which states that the minimum word_id = 1; implemented with map-lambda combo on every element of a document vector

    def to_id(self):
        return 'uci'


def write_vector(fname, doc_vector, doc_num):
    with open(fname, 'a') as f:
        f.writelines(map(lambda x: '{} {} {}\n'.format(doc_num, x[0], x[1]), doc_vector))


# class VowpalFormatWriter(StateLessDiskWriter):
#     def __init__(self, fname='/data/thesis/data/myvowpal'):
#         super(VowpalFormatWriter, self).__init__(fname, lambda x: write_vector(self.fname, map(lambda word_id_weight_tuple: (word_id_weight_tuple[0]+1, word_id_weight_tuple[1]), x), self.doc_num))
#         # + 1 to align with the uci format, which states that the minimum word_id = 1; implemented with map-lambda combo on every element of a document vector
#
#     def to_id(self):
#         return 'vowpal'
