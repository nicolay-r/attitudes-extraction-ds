import random

from core.evaluation.labels import NegativeLabel, PositiveLabel
from networks.cancellation import OperationCancellation
from networks.context.helpers.predict_log import PredictVariables
from networks.context.helpers.relations import ExtractedRelationsCollectionHelper
from networks.context.processing.relations.collection import ExtractedRelationsCollection
from networks.context.processing.relations.relation import ExtractedRelation
from networks.io import DataType
from networks.ranlp.model_context import RaNLPConfTaskModel


class RaNLPConfTaskBaselineRandModel(RaNLPConfTaskModel):

    def fit(self):
        operation_cancel = OperationCancellation()

        if self.callback is not None:
            self.callback.on_epoch_finished(avg_cost=-1.0,
                                            avg_acc=-1.0,
                                            epoch_index=0,
                                            operation_cancel=operation_cancel)

        if self.callback is not None:
            self.callback.on_fit_finished()

    def predict(self, dest_data_type=DataType.Test):
        return self.predict_core(dest_data_type=dest_data_type,
                                 rc_labeling_callback=self.__neg_labeling_callback)

    def __neg_labeling_callback(self, rc, dest_data_type):
        assert(isinstance(rc, ExtractedRelationsCollection))
        assert(isinstance(dest_data_type, unicode))

        rch = self.get_relations_collection_helper(dest_data_type)
        assert(isinstance(rch, ExtractedRelationsCollectionHelper))

        stat, _ = rch.get_statistic()
        neg_prec_bound = stat[2] / 100.0

        for relation in rc:
            assert(isinstance(relation, ExtractedRelation))
            next = random.random()
            label = NegativeLabel() if next <= neg_prec_bound else PositiveLabel()
            rc.apply_label(label=label, relation_id=relation.RelationID)

        return PredictVariables()