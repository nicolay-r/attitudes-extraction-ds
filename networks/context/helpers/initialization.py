import collections

from core.evaluation.labels import NeutralLabel
from core.runtime.relations import RelationCollection
from core.source.embeddings.base import Embedding
from core.source.embeddings.rusvectores import RusvectoresEmbedding
from core.source.embeddings.static import StaticEmbedding
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import Opinion, OpinionCollection
from core.source.synonyms import SynonymsCollection
from networks.context.configurations.base import CommonModelSettings
from networks.context.debug import DebugKeys
from networks.context.helpers.bags import BagsCollectionHelper
from networks.context.helpers.labels.paired import PairedLabelsHelper
from networks.context.helpers.labels.single import SingleLabelsHelper
from networks.context.helpers.news_terms import NewsTermsHelper
from networks.context.helpers.relations import ExtractedRelationsCollectionHelper
from networks.context.io import RuSentRelNetworkIO
from networks.context.processing import utils
from networks.context.processing.bags.collection import BagsCollection
from networks.context.processing.relations.collection import ExtractedRelationsCollection
from networks.context.processing.sample import Sample
from networks.context.processing.terms.collection import NewsTermsCollection
from networks.context.processing.terms.news_terms import NewsTerms
from networks.io import DataType


class ContextModelInitHelper(object):

    def __init__(self, io, settings):
        assert(isinstance(io, RuSentRelNetworkIO))
        assert(isinstance(settings, CommonModelSettings))

        print "Loading word embedding: {}".format(io.get_word_embedding_filepath())
        word_embedding = RusvectoresEmbedding.from_file(
            filepath=io.get_word_embedding_filepath(),
            binary=True,
            stemmer=settings.Stemmer,
            pos_tagger=settings.PosTagger)
        settings.set_word_embedding(word_embedding)

        self.__synonyms = SynonymsCollection.from_file(filepath=io.get_synonyms_collection_filepath(),
                                                       stemmer=settings.Stemmer,
                                                       is_read_only=True)

        self.__labels_helper = SingleLabelsHelper() if settings.ClassesCount == 3 else PairedLabelsHelper()

        print "Reading train collection ..."
        train_news_terms_collection, train_relations_collection, train_entities, train_relations_missed = \
            self.__read_collection(io=io, data_type=DataType.Train, settings=settings)

        print "Reading test collection ..."
        test_news_terms_collection, test_relations_collection, test_entities, test_relations_missed = \
            self.__read_collection(io=io, data_type=DataType.Test, settings=settings)

        print "Relations rejected (train): {}".format(train_relations_missed)
        print "Relations rejected (test):  {}".format(test_relations_missed)

        static_embedding = StaticEmbedding(settings.WordEmbedding.VectorSize)
        self.__fill_static_embedding_from_ntc(static_embedding=static_embedding,
                                              word_embedding=settings.WordEmbedding,
                                              ntc=train_news_terms_collection)
        self.__fill_static_embedding_from_ntc(static_embedding=static_embedding,
                                              word_embedding=settings.WordEmbedding,
                                              ntc=test_news_terms_collection)
        static_embedding.create_and_add_embedding(word=utils.ENTITY_MASK)

        settings.set_static_embedding(static_embedding)

        settings.set_term_embedding(
            utils.create_term_embedding_matrix(word_embedding=settings.WordEmbedding,
                                               static_embedding=settings.StaticWordEmbedding))

        self.__bags_collection = {
            DataType.Test: self.create_bags_collection(
                relations_collection=test_relations_collection,
                news_terms_collection=test_news_terms_collection,
                data_type=DataType.Test,
                settings=settings),
            DataType.Train: self.create_bags_collection(
                relations_collection=train_relations_collection,
                news_terms_collection=train_news_terms_collection,
                data_type=DataType.Train,
                settings=settings)
        }

        self.__bags_collection_helpers = {
            DataType.Train: BagsCollectionHelper(bags_collection=self.__bags_collection[DataType.Train],
                                                 name=DataType.Train),
            DataType.Test: BagsCollectionHelper(bags_collection=self.__bags_collection[DataType.Test],
                                                name=DataType.Test)
        }

        self.__relations_collections = {
            DataType.Test: test_relations_collection,
            DataType.Train: train_relations_collection
        }

        self.__relations_collection_helpers = {
            DataType.Test: ExtractedRelationsCollectionHelper(test_relations_collection,
                                                              labels_helper=self.__labels_helper,
                                                              name=DataType.Test),
            DataType.Train: ExtractedRelationsCollectionHelper(train_relations_collection,
                                                               labels_helper=self.__labels_helper,
                                                               name=DataType.Train)
        }

        self.__news_terms_collections = {
            DataType.Test: test_news_terms_collection,
            DataType.Train: train_news_terms_collection
        }

        norm, _ = self.__relations_collection_helpers[DataType.Train].get_statistic()

        settings.set_class_weights(norm)

    @property
    def Synonyms(self):
        return self.__synonyms

    @property
    def BagsCollections(self):
        return self.__bags_collection

    @property
    def BagsCollectionHelpers(self):
        return self.__bags_collection_helpers

    @property
    def RelationsCollections(self):
        return self.__relations_collections

    @property
    def RelationsCollectionHelpers(self):
        return self.__relations_collection_helpers

    @property
    def NewsTermsCollections(self):
        return self.__news_terms_collections

    @property
    def LabelsHelper(self):
        return self.__labels_helper

    @staticmethod
    def create_sample_from_relation_and_ntc(relation, ntc, settings):
        assert(isinstance(ntc, NewsTermsCollection))
        assert(relation.TextPosition.LeftEntityPosition.SentenceIndex ==
               relation.TextPosition.RightEntityPosition.SentenceIndex)

        news_terms = ntc.get_by_news_id(relation.TextPosition.NewsID)
        assert(isinstance(news_terms, NewsTerms))
        sentence_index = relation.TextPosition.LeftEntityPosition.SentenceIndex

        return Sample.from_relation(
            relation=relation,
            terms=list(news_terms.iter_sentence_terms(sentence_index)),
            term_index_in_sentence_func=lambda term_index: news_terms.get_term_index_in_sentence(term_index),
            settings=settings)

    @staticmethod
    def create_bags_collection(relations_collection, news_terms_collection, data_type, settings):
        assert(isinstance(relations_collection, ExtractedRelationsCollection))
        assert(isinstance(news_terms_collection, NewsTermsCollection))
        assert(isinstance(settings, CommonModelSettings))

        collection = BagsCollection.from_linked_relations(
            relations_collection,
            data_type=data_type,
            bag_size=settings.BagSize,
            shuffle=True,
            create_empty_sample_func=None,
            create_sample_func=lambda r: ContextModelInitHelper.create_sample_from_relation_and_ntc(
                relation=r, ntc=news_terms_collection, settings=settings))

        return collection

    @staticmethod
    def __fill_static_embedding_from_ntc(static_embedding, word_embedding, ntc):
        assert(isinstance(static_embedding, StaticEmbedding))
        assert(isinstance(word_embedding, Embedding))
        assert(isinstance(ntc, NewsTermsCollection))

        for news_ID in ntc.iter_news_ids():
            for term in ntc.iter_news_terms(news_ID):
                if isinstance(term, unicode) and \
                        term not in word_embedding and \
                        term not in static_embedding:
                    static_embedding.create_and_add_embedding(term)

    @staticmethod
    def __find_or_create_reversed_opinion(opinion, opinion_collections):
        assert(isinstance(opinion, Opinion))
        assert(isinstance(opinion_collections, collections.Iterable))

        reversed_opinion = Opinion(opinion.value_right, opinion.value_left, NeutralLabel())

        for collection in opinion_collections:
            if collection.has_synonymous_opinion(reversed_opinion):
                return collection.get_synonymous_opinion(reversed_opinion)

        return reversed_opinion

    def __read_collection(self, io, data_type, settings):
        assert(isinstance(io, RuSentRelNetworkIO))
        assert(isinstance(data_type, unicode))
        assert(isinstance(settings, CommonModelSettings))

        erc = ExtractedRelationsCollection()
        ntc = NewsTermsCollection()
        entities_list = []
        missed_relations_total = 0
        for news_index in io.get_data_indices(data_type):
            assert(isinstance(news_index, int))

            entity_filepath = io.get_entity_filepath(news_index)
            news_filepath = io.get_news_filepath(news_index)
            opin_filepath = io.get_opinion_input_filepath(news_index)
            neutral_filepath = io.get_neutral_filepath(news_index, data_type)

            entities = EntityCollection.from_file(entity_filepath, settings.Stemmer, self.__synonyms)

            news = News.from_file(news_filepath, entities)

            opinions_collections = [OpinionCollection.from_file(neutral_filepath, self.__synonyms)]
            if data_type == DataType.Train:
                opinions_collections.append(OpinionCollection.from_file(opin_filepath, self.__synonyms))

            news_terms = NewsTerms.create_from_news(news_index, news, keep_tokens=settings.KeepTokens)
            news_terms_helper = NewsTermsHelper(news_terms)

            if DebugKeys.NewsTermsStatisticShow:
                news_terms_helper.debug_statistics()
            if DebugKeys.NewsTermsShow:
                news_terms_helper.debug_show_terms()

            for relations, opinion, opinions in self.__extract_relations(opinions_collections, news, news_terms):
                reversed = ContextModelInitHelper.__find_or_create_reversed_opinion(opinion, opinions_collections)
                missed = erc.add_news_relations(relations=relations,
                                                label=self.__labels_helper.create_label_from_opinions(forward=opinion, backward=reversed),
                                                news_terms=news_terms,
                                                news_index=news_index,
                                                check_relation_is_correct=lambda r: Sample.check_ability_to_create_sample(
                                                    window_size=settings.TermsPerContext,
                                                    relation=r))
                missed_relations_total += missed

            ntc.add_news_terms(news_terms)
            entities_list.append(entities)

        return ntc, erc, entities_list, missed_relations_total

    # TODO: __extract_text_opinions
    @staticmethod
    def __extract_relations(opinion_collections, news, news_terms):
        assert(isinstance(opinion_collections, collections.Iterable))
        assert(isinstance(news, News))
        assert(isinstance(news_terms, NewsTerms))

        def filter_by_distance_in_sentences(relation):
            return abs(news.Helper.get_sentence_index_by_entity(relation.LeftEntity) -
                       news.Helper.get_sentence_index_by_entity(relation.RightEntity))

        for opinions in opinion_collections:
            assert (isinstance(opinions, OpinionCollection))
            for opinion in opinions:

                relations = RelationCollection.from_news_opinion(news, opinion)

                if len(relations) == 0:
                    continue

                relations.apply_filter(lambda relation: filter_by_distance_in_sentences(relation) == 0)

                yield relations, opinion, opinions
