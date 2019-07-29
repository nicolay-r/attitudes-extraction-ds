from core.runtime.parser import TextParser
from core.source.news import News, Sentence
from networks.context.debug import DebugKeys
from networks.context.processing.terms.position import EntityPosition


class NewsTerms:
    """
    Extracted News lexemes, such as:
        - news words
        - tokens
        - entities (positions).
    """

    def __init__(self, news_ID, terms, entity_positions, sentences_count, sentence_begin_inds):
        assert(isinstance(news_ID, int))
        assert(isinstance(terms, list))
        assert(isinstance(entity_positions, dict))
        assert(isinstance(sentences_count, int))
        assert(isinstance(sentence_begin_inds, list))
        self.__news_ID = news_ID
        self.__terms = terms
        self.__entity_positions = entity_positions
        self.__sentences_count = sentences_count
        self.__sentence_begin_inds = sentence_begin_inds

    @classmethod
    def create_from_news(cls, news_ID, news, keep_tokens):
        assert(isinstance(keep_tokens, bool))
        terms, entity_positions, sentence_begin_inds = cls._extract_terms_and_entity_positions(news, keep_tokens)
        return cls(news_ID, terms, entity_positions, news.sentences_count(), sentence_begin_inds)

    def iter_terms(self):
        for term in self.__terms:
            yield term

    def iter_sentence_terms(self, sentence_index):
        assert(isinstance(sentence_index, int))
        begin = self.__sentence_begin_inds[sentence_index]
        end = len(self.__terms) if sentence_index == self.__sentences_count - 1 \
            else self.__sentence_begin_inds[sentence_index + 1]
        for i in range(begin, end):
            yield self.__terms[i]

    @property
    def RelatedNewsID(self):
        return self.__news_ID

    def get_entity_position(self, entity_ID):
        assert(type(entity_ID) == unicode)      # ID which is a part of *.ann files.
        return self.__entity_positions[entity_ID]

    def get_term_index_in_sentence(self, term_index):
        assert(isinstance(term_index, int))
        begin = 0
        for i, begin_index in enumerate(self.__sentence_begin_inds):
            if begin_index > term_index:
                break
            begin = begin_index

        return term_index - begin

    @staticmethod
    def _extract_terms_and_entity_positions(news, keep_tokens):
        assert(isinstance(news, News))
        assert(isinstance(keep_tokens, bool))

        sentence_begin = []
        terms = []
        entity_positions = {}
        for s_index, sentence in enumerate(news.iter_sentences()):
            assert(isinstance(sentence, Sentence))
            sentence_begin.append(len(terms))
            s_pos = 0
            # TODO: guarantee that entities ordered by e_begin.
            for e_ID, e_begin, e_end in sentence.iter_entities_info():
                # add terms before entity
                if e_begin > s_pos:
                    parsed_text_before = TextParser.parse(sentence.Text[s_pos:e_begin], keep_tokens=keep_tokens)
                    terms.extend(parsed_text_before.iter_raw_terms())
                # add entity position
                entity_positions[e_ID] = EntityPosition(term_index=len(terms), sentence_index=s_index)
                # add entity_text
                terms.append(news.Entities.get_entity_by_id(e_ID))
                s_pos = e_end

            # add text part after last entity of sentence.
            parsed_text_last = TextParser.parse((sentence.Text[s_pos:len(sentence.Text)]), keep_tokens=keep_tokens)
            terms.extend(parsed_text_last.iter_raw_terms())

        return terms, entity_positions, sentence_begin

    def __len__(self):
        return len(self.__terms)

