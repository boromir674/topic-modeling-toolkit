import os
import json
from .base_evaluator import ArtmScorer


scorer_type2_reportables = {
    'background-tokens-ratio': ('tokens', 'value'),
    'items-processed': 'value',
    'perplexity': ('class_id_info', 'normalizer', 'raw', 'value'),
    'sparsity-phi': ('total_tokens', 'value', 'zero_tokens'),
    'sparsity-theta': ('total_topics', 'value', 'zero_topics'),
    'theta-snippet': ('document_ids', 'snippet'),
    'topic-mass-phi': ('topic_mass', 'topic_ratio', 'value'),
    'topic-kernel': ('average_coherence', 'average_contrast', 'average_purity', 'average_size', 'coherence', 'contrast', 'purity', 'size', 'tokens'),
    'top-tokens': ('average_coherence', 'coherence', 'num_tokens', 'tokens', 'weights')
}


class ArtmScorerFactory(object):
    def __init__(self, model, scorers_dict):
        self._model = model
        self.scorers = scorers_dict

    def create_scorer(self, name, output_dir):
        assert name in self.scorers.values()
        ars = ArtmScorer(mame, tuple(scorer_type2_reportables[{v: k for k, v in self.scorers.items()}[name]]))
        ars.output_dir = output_dir
        # @addto(ars)
        # def write_to_disk(self):
        #     for attr in self.attributes:
        #         dump_to_disk(os.path.join(output_dir, attr, '.json'), self.evaluate())

        return ars


# def addto(instance):
#     def decorator(f):
#         import types
#         f = types.MethodType(f, instance, instance.__class__)
#         setattr(instance, f.func_name, f)
#         return f
#     return decorator
#
#
# def dump_to_disk(filename, dicti):
#     with open(filename, 'w') as f:
#         json.dump(dicti, f)
