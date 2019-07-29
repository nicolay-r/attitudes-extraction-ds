from core.runtime.object import TextObject
from core.runtime.ref_opinon import RefOpinion
from core.source.ruattitudes.helpers.news_helper import NewsProcessingHelper
from networks.context.processing.relations.position import ExtractedRelationPosition
from networks.context.processing.relations.relation import ExtractedRelation
from networks.context.processing.terms.position import EntityPosition
from core.source.ruattitudes.reader import ProcessedNews, ProcessedSentence


class CustomProcessingHelper(NewsProcessingHelper):

    @staticmethod
    def iter_linked_relations(processed_news):
        """
        Composes and iters linked relations from processed news
        """
        assert(isinstance(processed_news, ProcessedNews))

        opin_dict = CustomProcessingHelper.build_opinion_dict(processed_news)

        for opin_key, s_inds in opin_dict.iteritems():

            extracted_relations = []
            sentences = []

            for s_ind in s_inds:
                sentence = processed_news.get_sentence(s_ind)
                opinion_ref = sentence.find_ref_opinion_by_key(opin_key)
                relation = CustomProcessingHelper.__create_relation(
                    news_index=processed_news.NewsIndex,
                    processed_sentence=sentence,
                    ref_opinion=opinion_ref)
                extracted_relations.append(relation)
                sentences.append(s_ind)

            yield extracted_relations, sentences

    @staticmethod
    def __create_relation(news_index, processed_sentence, ref_opinion):
        assert(isinstance(news_index, int))
        assert(isinstance(processed_sentence, ProcessedSentence))
        assert(isinstance(ref_opinion, RefOpinion))

        left_text_obj, right_text_obj = processed_sentence.get_objects(ref_opinion)

        assert(isinstance(left_text_obj, TextObject))
        assert(isinstance(right_text_obj, TextObject))

        left_position = EntityPosition(term_index=left_text_obj.Position,
                                       sentence_index=processed_sentence.SentenceIndex)

        right_position = EntityPosition(term_index=right_text_obj.Position,
                                        sentence_index=processed_sentence.SentenceIndex)

        position = ExtractedRelationPosition(news_id=news_index,
                                             left=left_position,
                                             right=right_position)

        relation = ExtractedRelation(text_position=position,
                                     relation_id=None,
                                     left_entity_value=left_text_obj.get_value(),
                                     right_entity_value=right_text_obj.get_value(),
                                     label=ref_opinion.Sentiment)

        return relation
