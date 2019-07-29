from core.evaluation.labels import NeutralLabel
from core.source.opinion import OpinionCollection, Opinion
from core.source.synonyms import SynonymsCollection
from networks.context.helpers.labels.base import LabelsHelper
from networks.context.processing.relations.collection import ExtractedRelationsCollection, ExtractedRelation


class ExtractedRelationsCollectionHelper:

    def __init__(self, collection, labels_helper, name):
        assert(isinstance(collection, ExtractedRelationsCollection))
        assert(isinstance(labels_helper, LabelsHelper))
        assert(isinstance(name, unicode))
        self.__collection = collection
        self.__labels_helper = labels_helper
        self.__name = name

    @property
    def Name(self):
        return self.__name

    @staticmethod
    def __opitional_add_opinion(opinion, collection):
        assert(isinstance(opinion, Opinion))

        if isinstance(opinion.sentiment, NeutralLabel):
            return

        if collection.has_synonymous_opinion(opinion):
            return

        collection.add_opinion(opinion)

    def __get_group_statistic(self):
        statistic = {}
        for group in self.__collection.iter_by_linked_relations():
            key = len(group)
            if key not in statistic:
                statistic[key] = 1
            else:
                statistic[key] += 1
        return statistic

    def save_into_opinion_collections(self,
                                      create_opinion_collection,
                                      create_filepath_by_news_id,
                                      label_calculation_mode):
        assert(callable(create_opinion_collection))
        assert(callable(create_filepath_by_news_id))
        assert(isinstance(label_calculation_mode, unicode))

        for news_ID in self.__iter_unique_news_ids():

            collection = create_opinion_collection()
            assert(isinstance(collection, OpinionCollection))

            self.__fill_opinion_collection(
                opinions=collection,
                news_ID=news_ID,
                label_mode=label_calculation_mode)

            collection.save(filepath=create_filepath_by_news_id(news_ID))

    def debug_labels_statistic(self):
        norm, _ = self.get_statistic()
        total = len(self.__collection)
        print "Extracted relation collection: {}".format(self.__name)
        print "\tTotal: {}".format(total)
        for i, value in enumerate(norm):
            label = self.__labels_helper.create_label_from_uint(i)
            print "\t{}: {:.2f}%".format(label.to_str(), value)

    def debug_unique_relations_statistic(self):
        statistic = self.__get_group_statistic()
        total = sum(list(statistic.itervalues()))
        print "Unique relations statistic: {}".format(self.__name)
        print "\tTotal: {}".format(total)
        for key, value in sorted(statistic.iteritems()):
            print "\t{} -- {} ({:.2f}%)".format(key, value, 100.0 * value / total)
            total += value

    def get_statistic(self):
        stat = [0] * self.__labels_helper.get_classes_count()
        for relation in self.__collection:
            assert(isinstance(relation, ExtractedRelation))
            stat[relation.Label.to_uint()] += 1

        total = sum(stat)
        norm = [100.0 * value / total if total > 0 else 0 for value in stat]
        return norm, stat

    def iter_unique_news_ids(self):
        return self.__iter_unique_news_ids()

    def __fill_opinion_collection(self, opinions, news_ID, label_mode):
        assert(isinstance(opinions, OpinionCollection) and len(opinions) == 0)
        assert(isinstance(news_ID, int))

        for linked_relations in self.__collection.iter_by_linked_relations():

            first_ex_relation = linked_relations[0]
            if first_ex_relation.TextPosition.NewsID != news_ID:
                continue

            label = self.__labels_helper.create_label_from_relations(
                relation_labels=[r.Label for r in linked_relations],
                label_creation_mode=label_mode)

            opinion_list = self.__labels_helper.create_opinions_by_relation_and_label(
                extracted_relation=first_ex_relation,
                label=label)

            for opinion in opinion_list:
                self.__opitional_add_opinion(opinion=opinion,
                                             collection=opinions)

    def __iter_unique_news_ids(self):
        news_ids = {}
        for relation in self.__collection:
            id = relation.TextPosition.NewsID
            if id not in news_ids:
                news_ids[id] = True

        for id in news_ids.iterkeys():
            yield id

