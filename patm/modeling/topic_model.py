

class TopicModel(object):
    """
    An instance of this class encapsulates the behaviour of a topic model
    """
    def __init__(self, artm_model, a_buffer=None):
        self.model = artm_model
        self._collection_passes = []
        # self.clusters = [Cluster(cl_id, clid2members[cl_id], self._av) for cl_id in range(len(clid2members))]
        self._buffer = a_buffer
        
    def update(self):
        pass
    # def __str__(self):
    #     pre = 2
    #     body, max_lens = self._get_rows(threshold=10, prob_precision=pre)
    #     header = self._get_header(max_lens, pre, [_ for _ in range(len(self))])
    #     return header + body

    # def __iter__(self):
    #     for cl in self.clusters:
    #         yield cl

    # def __len__(self):
    #     return len(self.clusters)

    # def __getitem__(self, item):
    #     return self.clusters[item]

    # def get_cluster_id(self, member_id):
    #     for cl in self:
    #         if member_id in cl.members:
    #             return cl.id


    # def gen_clusters(self, selected):
    #     """
    #     Generates Cluster objects according to the indices in the selected clusters list
    #     :param selected: the indices of the clusters to select
    #     :type selected: list
    #     :return: the generated cluster
    #     :rtype: Cluster
    #     """
    #     for i in selected:
    #         yield self[i]


class TrainSpecs(object):
    def __init__(self, specs):
        self.specs = specs
        assert 'collection_passes' in self.specs

    def __getitem__(self, item):
        return self.specs[item]
