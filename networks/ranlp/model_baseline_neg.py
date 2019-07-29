from core.evaluation.labels import NegativeLabel
from networks.cancellation import OperationCancellation
from networks.context.helpers.predict_log import PredictVariables
from networks.context.processing.relations.collection import ExtractedRelationsCollection
from networks.context.processing.relations.relation import ExtractedRelation
from networks.io import DataType
from networks.ranlp.model_context import RaNLPConfTaskModel


class RaNLPConfTaskBaselineNegModel(RaNLPConfTaskModel):

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

        for relation in rc:
            assert(isinstance(relation, ExtractedRelation))
            rc.apply_label(label=NegativeLabel(), relation_id=relation.RelationID)

        return PredictVariables()
