import numpy as np

from core.evaluation.labels import Label
from core.source.opinion import Opinion
from networks.context.configurations.base import LabelCalculationMode
from networks.context.helpers.labels.base import LabelsHelper
from networks.context.processing.relations.relation import ExtractedRelation


class SingleLabelsHelper(LabelsHelper):

    @staticmethod
    def get_classes_count():
        return 3

    @staticmethod
    def create_label_from_relations(relation_labels, label_creation_mode):
        assert(isinstance(relation_labels, list))
        assert(isinstance(label_creation_mode, unicode))

        label = None
        if label_creation_mode == LabelCalculationMode.FIRST_APPEARED:
            label = relation_labels[0]
        if label_creation_mode == LabelCalculationMode.AVERAGE:
            forwards = [l.to_int() for l in relation_labels]
            label = Label.from_int(np.sign(sum(forwards)))

        return label

    @staticmethod
    def create_label_from_uint(label_uint):
        assert(label_uint >= 0)
        return Label.from_uint(label_uint)

    @staticmethod
    def create_label_from_opinions(forward, backward):
        assert(isinstance(forward, Opinion))
        return forward.sentiment

    @staticmethod
    def create_opinions_by_relation_and_label(extracted_relation, label):
        assert(isinstance(extracted_relation, ExtractedRelation))
        assert(isinstance(label, Label))

        opinion = Opinion(value_left=extracted_relation.LeftEntityValue,
                          value_right=extracted_relation.RightEntityValue,
                          sentiment=label)

        return [opinion]

