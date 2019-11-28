from core.runtime.parser import ParsedText
from core.source.embeddings.rusvectores import RusvectoresEmbedding
from core.source.embeddings.static import StaticEmbedding
from networks.context.configurations.base import CommonModelSettings
from networks.context.helpers.bags import BagsCollectionHelper
from networks.context.helpers.labels.single import SingleLabelsHelper
from networks.context.helpers.relations import ExtractedRelationsCollectionHelper
from networks.context.processing.bags.collection import BagsCollection
from networks.context.processing.relations.collection import ExtractedRelationsCollection
from networks.context.processing.relations.relation import ExtractedRelation
from networks.context.processing.sample import Sample
from networks.context.processing.utils import create_term_embedding_matrix
from networks.pretraining.helpers.news import NewsProcessingHelper
from networks.pretraining.utils.reader import ContextsReader, ProcessedNews, ProcessedSentence


class ModelInitHelper(object):

    # TODO: Add files list into args.
    def __init__(self,
                 settings,
                 samples_filepaths,
                 word_embedding_filepath,
                 data_type):
        assert(isinstance(settings, CommonModelSettings))
        assert(isinstance(data_type, unicode))
        assert(isinstance(word_embedding_filepath, unicode))

        print "Create init helper: {}".format(data_type)

        self.__settings = settings
        self.__relations_collection = ExtractedRelationsCollection()
        self.__parsed_texts_dict = {}
        self.__data_type = data_type

        # TODO. Others not supported now. Use label_helper to overcome latter.
        self.__labels_helper = SingleLabelsHelper()

        for filepath in samples_filepaths:

            print "Reading filepath: {}".format(filepath)
            relations_rejected = ModelInitHelper.__register_relations_and_texts(
                registered_relations=self.__relations_collection,
                reg_texts=self.__parsed_texts_dict,
                samples_filepath=filepath,
                settings=settings)
            print "Relations rejected ({}): {}".format(data_type, relations_rejected)

        print "Create word embedding ..."
        print "Loading word embedding: {}".format(word_embedding_filepath)
        if settings.WordEmbedding is None:
            word_embedding = RusvectoresEmbedding.from_file(
                filepath=word_embedding_filepath,
                binary=True,
                stemmer=settings.Stemmer,
                pos_tagger=settings.PosTagger)
            settings.set_word_embedding(word_embedding)
        else:
            print "Already initialized. [SKIPPED]"

        print "Create static embedding matrix ..."
        if settings.StaticWordEmbedding is None:
            settings.set_static_embedding(StaticEmbedding(settings.WordEmbedding.VectorSize))
            ModelInitHelper.__fill_static_embedding_from_parsed_texts(
                settings=settings,
                parsed_texts_dict=self.__parsed_texts_dict)
        else:
            print "Already existed. [SKIPPED]"

        print "Create term embedding matrix ..."
        if settings.TermEmbeddingMatrix is None:
            matrix = create_term_embedding_matrix(
                word_embedding=settings.WordEmbedding,
                static_embedding=settings.StaticWordEmbedding)
            settings.set_term_embedding(matrix)
        else:
            print "Already existed. [SKIPPED]"

        print "Create bags ..."
        self.__bags_collection = self.__create_bags_collection(self.__relations_collection)

        print "Create bags helpers ..."
        self.__bags_collection_helper = BagsCollectionHelper(
            bags_collection=self.__bags_collection,
            name=data_type)

        print "Create relations collection helper ..."
        self.__relations_collection_helper = ExtractedRelationsCollectionHelper(
            collection=self.__relations_collection,
            labels_helper=self.__labels_helper,
            name=data_type)

        self.__relations_collection_helper.debug_labels_statistic()
        self.__relations_collection_helper.debug_unique_relations_statistic()
        self.__bags_collection_helper.print_log_statistics()


    @property
    def RelationsCollection(self):
        return self.__relations_collection

    @property
    def ParsedTexts(self):
        return self.__parsed_texts_dict

    @property
    def BagsCollection(self):
        return self.__bags_collection

    @property
    def RelationsCollectionHelper(self):
        return self.__relations_collection_helper

    @property
    def BagsCollectionHelper(self):
        return self.__bags_collection_helper

    @property
    def LabelsHelper(self):
        return self.__labels_helper

    @staticmethod
    def __fill_static_embedding_from_parsed_texts(settings, parsed_texts_dict):
        assert(isinstance(settings, CommonModelSettings))
        assert(isinstance(parsed_texts_dict, dict))
        for parsed_text in parsed_texts_dict.itervalues():
            assert(isinstance(parsed_text, ParsedText))
            for term in parsed_text.iter_raw_terms():
                if isinstance(term, unicode) and \
                        term not in settings.WordEmbedding and \
                        term not in settings.StaticWordEmbedding:
                    settings.StaticWordEmbedding.create_and_add_embedding(term)

    @staticmethod
    def __register_relations_and_texts(registered_relations, reg_texts, samples_filepath, settings):
        assert(isinstance(registered_relations, ExtractedRelationsCollection))
        assert(isinstance(settings, CommonModelSettings))

        total_relations_rejected = 0
        for processed_news in ContextsReader.iter_processed_news(samples_filepath):
            assert(isinstance(processed_news, ProcessedNews))
            for relations, s_inds in NewsProcessingHelper.iter_linked_relations(processed_news):

                rejected_count = registered_relations.add_extracted_relations(
                    relations=relations,
                    check_relation_is_correct=lambda r: Sample.check_ability_to_create_sample(
                        window_size=settings.TermsPerContext,
                        relation=r))

                total_relations_rejected += rejected_count

                for i, relation in enumerate(relations):
                    assert(isinstance(relation, ExtractedRelation))

                    if relation.RelationID is None:
                        continue

                    sentence = processed_news.get_sentence(s_inds[i])
                    assert(isinstance(sentence, ProcessedSentence))
                    reg_texts[relation.RelationID] = sentence.ParsedText

        return total_relations_rejected

    def __create_bags_collection(self, relations_collection):
        assert(isinstance(relations_collection, ExtractedRelationsCollection))
        return BagsCollection.from_linked_relations(
            relations_collection,
            data_type=self.__data_type,
            bag_size=self.__settings.BagSize,
            shuffle=True,
            create_empty_sample_func=None,
            create_sample_func=self.__create_sample_from_relation)

    def __create_sample_from_relation(self, relation):
        assert(isinstance(relation, ExtractedRelation))
        return Sample.from_relation(
            relation=relation,
            terms=list(self.__parsed_texts_dict[relation.RelationID].iter_raw_terms()),       # TODO: Texts Terms, include Entites!!!
            term_index_in_sentence_func=lambda term_index: term_index,                        # TODO: Fix it after Entities.
            settings=self.__settings)
