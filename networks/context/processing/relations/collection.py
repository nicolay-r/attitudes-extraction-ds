# -*- coding: utf-8 -*-
import cPickle as pickle
import collections

from core.runtime.relations import RelationCollection
from networks.context.processing.relations.position import ExtractedRelationPosition
from networks.context.processing.relations.relation import ExtractedRelation
from networks.context.processing.terms.news_terms import NewsTerms


class ExtractedRelationsCollection:
    """
    Describes text relations with a position precision and forward connection, so
    for each relation we store it's continuation if it originally has.

    Usage:
    It is possible to add relations from news via add_news_relations

    Limitations:
    Not it represents IN-MEMORY implementation.
    Therefore it is supposed not so large amount of relations.
    """

    NO_NEXT_RELATION = None

    def __init__(self):
        # list ExtractedRelations
        self.__relations = []
        # list describes that has i'th relation continuation in text.
        self.__next_relation_id = []
        # provides original label by relation_id
        self.__opinion_labels = []
        # labels defined
        self.__labels_defined = []

    def add_extracted_relations(self,
                                relations,
                                check_relation_is_correct):
        assert(isinstance(relations, collections.Iterable))
        assert(callable(check_relation_is_correct))

        missed = 0
        for index, relation in enumerate(relations):
            assert(isinstance(relation, ExtractedRelation))
            assert(relation.RelationID is None)

            if not check_relation_is_correct(relation):
                missed += 1
                continue

            relation.set_relation_id(len(self.__relations))

            self.__register_relation(relation)

        self.__next_relation_id[-1] = self.NO_NEXT_RELATION
        return missed

    def add_news_relations(self,
                           relations,
                           label,
                           news_terms,
                           news_index,
                           check_relation_is_correct):
        assert(isinstance(relations, RelationCollection))
        assert(isinstance(news_terms, NewsTerms))
        assert(isinstance(news_index, int))
        assert(callable(check_relation_is_correct))

        missed = 0
        for index, relation in enumerate(relations):

            pos_subj = news_terms.get_entity_position(relation.LeftEntityID)
            pos_obj = news_terms.get_entity_position(relation.RightEntityID)
            relation_id = len(self.__relations)

            extracted_relation = ExtractedRelation(
                ExtractedRelationPosition(news_index, pos_subj, pos_obj),
                relation_id,
                relation.LeftEntityValue,
                relation.RightEntityValue,
                label)

            if not check_relation_is_correct(extracted_relation):
                missed += 1
                continue

            self.__register_relation(extracted_relation)

        self.__next_relation_id[-1] = self.NO_NEXT_RELATION
        return missed

    def __register_relation(self, relation):
        assert(isinstance(relation, ExtractedRelation))
        self.__relations.append(relation)
        self.__next_relation_id.append(relation.RelationID + 1)
        self.__opinion_labels.append(relation.Label)
        self.__labels_defined.append(True)

    def check_all_relations_has_labels(self):
        return not (False in self.__labels_defined)

    def check_all_relations_without_labels(self):
        return not (True in self.__labels_defined)

    def apply_label(self, label, relation_id):
        assert(isinstance(relation_id, int))

        if self.__labels_defined[relation_id] is not False:
            assert(self.__relations[relation_id].Label == label)
            return

        self.__relations[relation_id].set_label(label)
        self.__labels_defined[relation_id] = True

    def get_original_label(self, relation_id):
        assert(isinstance(relation_id, int))
        return self.__opinion_labels[relation_id]

    def reset_labels(self):
        for relation in self.__relations:
            relation.set_label(self.__opinion_labels[relation.RelationID])
        self.__labels_defined = [False] * len(self.__labels_defined)

    def save(self, pickle_filepath):
        pickle.dump(self, open(pickle_filepath, 'wb'))

    @classmethod
    def load(cls, pickle_filepath):
        return pickle.load(open(pickle_filepath, 'rb'))

    def __iter__(self):
        for relation in self.__relations:
            yield relation

    def __len__(self):
        return len(self.__relations)

    def __iter_by_linked_relations(self):
        lst = []
        for index, relation in enumerate(self.__relations):
            lst.append(relation)
            if self.__next_relation_id[index] == self.NO_NEXT_RELATION:
                yield lst
                lst = []

    def iter_by_linked_relations(self):
        return self.__iter_by_linked_relations()

    def iter_by_linked_relations_groups(self, group_size):
        assert(isinstance(group_size, int))
        group = []
        for index, linked_relations in enumerate(self.__iter_by_linked_relations()):
            group.append(linked_relations)
            if len(group) == group_size:
                yield group
                group = []
