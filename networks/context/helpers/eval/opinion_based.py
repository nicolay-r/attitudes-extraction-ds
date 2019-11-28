import collections

from core.source.synonyms import SynonymsCollection
from networks.context.debug import DebugKeys
from base import EvaluationHelper
from networks.io import NetworkIO


class OpinionBasedEvaluationHelper(EvaluationHelper):

    @staticmethod
    def evaluate_model(data_type, io, indices, synonyms):
        assert(isinstance(io, NetworkIO))
        assert(isinstance(indices, collections.Iterable))
        assert(isinstance(synonyms, SynonymsCollection) and synonyms.IsReadOnly)

        files_to_compare = io.iter_opinion_files_to_compare(data_type=data_type,
                                                            etalon_root=io.get_etalon_root(),
                                                            indices=indices)
        evaluator = Evaluator(synonyms)
        return evaluator.evaluate(files_to_compare_list=list(files_to_compare),
                                  debug=DebugKeys.EvaluateDebug)
