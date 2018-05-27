
class ModelFitListener(object):

    def update(self, *args, **kwargs):

        raise NotImplementedError
        # map_id = '{}_{}_'.format(self.backend, self._dataset_id) + '_'.join(args)
        # self.id2map_obj[map_id] = kwargs['map_object']
        # self.som = kwargs['map_object']
        # self.map_obj2id[self.som] = map_id
