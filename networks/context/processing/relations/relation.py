from core.evaluation.labels import LabelPair, Label
from networks.context.processing.relations.position import ExtractedRelationPosition


class ExtractedRelation:
    """
    Represents a relation which were found in news article
    and composed between two named entities
    (it was found especially by Opinion with predefined label)
    """

    def __init__(self,
                 text_position,
                 relation_id,
                 left_entity_value,
                 right_entity_value,
                 label):
        assert(isinstance(text_position, ExtractedRelationPosition))
        assert(isinstance(relation_id, int) or relation_id is None)
        assert(isinstance(left_entity_value, unicode))
        assert(isinstance(right_entity_value, unicode))
        self.__text_position = text_position
        self.__relation_id = relation_id
        self.__left_entity_value = left_entity_value
        self.__right_entity_value = right_entity_value
        self.__label = None
        self.__set_label(label)

    @property
    def LeftEntityValue(self):
        return self.__left_entity_value

    @property
    def RightEntityValue(self):
        return self.__right_entity_value

    @property
    def TextPosition(self):
        return self.__text_position

    @property
    def Label(self):
        return self.__label

    @property
    def RelationID(self):
        return self.__relation_id

    def __set_label(self, label):
        assert(isinstance(label, LabelPair) or
               isinstance(label, Label))
        self.__label = label

    def set_relation_id(self, relation_id):
        assert(isinstance(relation_id, int))
        self.__relation_id = relation_id

    def set_label(self, label):
        self.__set_label(label)
