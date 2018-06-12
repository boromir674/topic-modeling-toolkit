from abc import ABCMeta, abstractmethod

from ..utils import dump_to_disk


class MetaEvaluator:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__mro__ = [AbstractEvaluator]

    @abstractmethod
    def evaluate(self, data):
        raise NotImplementedError

    def __str__(self):
        return type(self).__name__

    @classmethod
    def __subclasshook__(a_class, C):
        if a_class is AbstractEvaluator:
            if any('evaluate' in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


class AbstractEvaluator(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def evaluate(self, data):
        return self.method(data)


# class SplitEvaluator(AbstractEvaluator):
#     def __init__(self, dataset, method):
#         self._dataset = dataset
#         self.method = method
#         super(SplitEvaluator, self).__init__('{}-evaluator'.format(self._dataset.name))
#
#     def evaluate(self, split):
#         sp = self._dataset[split]
#         pass



MetaEvaluator.register(AbstractEvaluator)


if __name__ == '__main__':
    ae = AbstractEvaluator()
    ass = ArtmScorer()
    gav = Gav()
    # print AbstractEvaluator.__mro__
    # print ArtmScorer.__mro__
    print isinstance(gav, MetaEvaluator)
    print issubclass(Gav, MetaEvaluator)
    print gav.evaluate('taf')
    # print issubclass(ArtmScorer, MetaEvaluator)
    # ass.evaluate()
    # ae.evaluate('gav')
