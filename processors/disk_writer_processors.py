from processor import StateLessProcessor


class StateLessDiskWriter(StateLessProcessor):
    pass


class UciFormatWriter(StateLessDiskWriter):
    """Ingests one doc vector at a time"""
    def __init__(self, fname='/data/thesis/data/myuci'):
        self.doc_num = 1
        self.fname = fname
        super(StateLessDiskWriter, self).__init__(lambda x: write_vector(self.fname, map(lambda word_id_weight_tuple: (word_id_weight_tuple[0]+1, word_id_weight_tuple[1]), x), self.doc_num))
        # + 1 to align with the uci format, which states that the minimum word_id = 1; implemented with map-lambda combo on every element of a document vector

    def __str__(self):
        return super(StateLessDiskWriter, self).__str__() + '(' + str(self.fname) + ')'

    def to_id(self):
        return 'uci'

    def process(self, data):
        _ = super(StateLessDiskWriter, self).process(data)
        self.doc_num += 1
        return _


def write_vector(fname, doc_vector, doc_num):
    with open(fname, 'a') as f:
        f.writelines(map(lambda x: '{} {} {}\n'.format(doc_num, x[0], x[1]), doc_vector))
