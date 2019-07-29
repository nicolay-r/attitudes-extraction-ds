import numpy as np

from core.evaluation.labels import Label, LabelPair
from core.source.opinion import Opinion
from networks.context.configurations.base import LabelCalculationMode
from networks.context.debug import DebugKeys
from networks.context.helpers.labels.base import LabelsHelper
from networks.context.processing.relations.relation import ExtractedRelation


class PairedLabelsHelper(LabelsHelper):

    @staticmethod
    def get_classes_count():
        return 9

    @staticmethod
    def create_label_from_uint(label_uint):
        assert(label_uint >= 0)
        return LabelPair.from_uint(label_uint)

    @staticmethod
    def create_label_from_relations(relation_labels, label_creation_mode):
        assert(isinstance(relation_labels, list))
        assert(isinstance(label_creation_mode, unicode))

        label = None
        if label_creation_mode == LabelCalculationMode.FIRST_APPEARED:
            label = relation_labels[0]
        if label_creation_mode == LabelCalculationMode.AVERAGE:
            forwards = [l.Forward.to_int() for l in relation_labels]
            backwards = [l.Backward.to_int() for l in relation_labels]
            label = LabelPair(forward=Label.from_int(np.sign(sum(forwards))),
                              backward=Label.from_int(np.sign(sum(backwards))))

        if DebugKeys.PredictLabel:
            print [l.to_int() for l in relation_labels]
            print "Result: {}".format(label.to_int())

        # TODO: Correct label

        return label

    @staticmethod
    def create_label_from_opinions(forward, backward):
        assert(isinstance(forward, Opinion))
        assert(isinstance(backward, Opinion))
        return LabelPair(forward=forward.sentiment, backward=backward.sentiment)

    @staticmethod
    def create_opinions_by_relation_and_label(extracted_relation, label):
        assert(isinstance(extracted_relation, ExtractedRelation))
        assert(isinstance(label, LabelPair))

        forward_opinion = Opinion(value_left=extracted_relation.LeftEntityValue,
                                  value_right=extracted_relation.RightEntityValue,
                                  sentiment=label.Forward)

        backward_opinion = Opinion(value_left=extracted_relation.RightEntityValue,
                                   value_right=extracted_relation.LeftEntityValue,
                                   sentiment=label.Backward)

        return [forward_opinion, backward_opinion]
