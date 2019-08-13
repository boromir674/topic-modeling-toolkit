from .processor import StateLessProcessor


class GeneratorProcessor(StateLessProcessor):
    def __str__(self):
        return type(self).__name__


def gen_ngrams(word_generator, degree):
    assert degree > 1
    gen_empty = False
    total = 0
    queue = []
    while total < degree:
        queue.append(next(word_generator))
        total += 1
    yield '_'.join(queue)
    total = 1  # total unigrams generated
    while not gen_empty:
        try:
            del queue[0]
            queue.append(next(word_generator))
            yield '_'.join(queue)
            total += 1
        except StopIteration:
            # print 'Generated {} {}-grams'.format(total, degree)
            gen_empty = True


def ngrams_convertion(word_generator, degree):
    if degree == 1:
        return word_generator
    else:
        return (_ for _ in gen_ngrams(word_generator, degree))


def min_length_filter(word_generator, min_length):
    if min_length < 2:
        min_length = 2
    return (w for w in word_generator if len(w) >= min_length)


def max_length_filter(word_generator, max_length):
    if max_length > 50:
        max_length = 50
    return (w for w in word_generator if len(w) <= max_length)


class MinLengthFilter(GeneratorProcessor):
    def __init__(self, min_length):
        self.min_length = min_length
        super(GeneratorProcessor, self).__init__(lambda x: min_length_filter(x, self.min_length))

    def __str__(self):
        return super(GeneratorProcessor, self).__str__() + '(' + str(self.min_length) + ')'

    def to_id(self):
        return 'min_length-{}'.format(self.min_length)


class MaxLengthFilter(GeneratorProcessor):
    def __init__(self, max_length):
        self.max_length = max_length
        super(GeneratorProcessor, self).__init__(lambda x: max_length_filter(x, self.max_length))

    def __str__(self):
        return super(GeneratorProcessor, self).__str__() + '(' + str(self.max_length) + ')'

    def to_id(self):
        return 'max_length-{}'.format(self.max_length)


class WordToNgramGenerator(GeneratorProcessor):
    def __init__(self, degree):
        super(GeneratorProcessor, self).__init__(lambda x: ngrams_convertion(x, degree))
        self.degree = degree

    def __str__(self):
        return super(GeneratorProcessor, self).__str__() + '(' + str(self.degree) + ')'

    def to_id(self):
        return 'ngrams-{}'.format(self.degree)
