import numpy as np
import collections
from core.processing.pos.mystem_wrap import POSTagger
from core.source.embeddings.static import StaticEmbedding
from core.source.entity import Entity
from core.source.tokens import Tokens, Token
from core.source.embeddings.base import Embedding
from core.source.embeddings.tokens import TokenEmbeddingVectors
from networks.context.debug import DebugKeys


ENTITY_MASK = u"entity"


def calculate_embedding_indices_for_terms(terms,
                                          term_embedding_matrix,
                                          word_embedding,
                                          static_embedding):
    """
    terms: list
        list that includes words, tokens, entities.
    static_embedding StaticEmbedding
        embedding of missed words

    returns: list int
        list of indices
    """
    # O(N^2) because of search at embedding by word to obtain related
    # index.
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(term_embedding_matrix, np.ndarray))
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(static_embedding, StaticEmbedding))

    indices = []
    embedding_offsets = TermsEmbeddingOffsets(words_count=len(word_embedding.vocab),
                                              static_words_count=len(static_embedding))

    unknown_word_embedding_index = embedding_offsets.get_token_index(
        TokenEmbeddingVectors.get_token_index(Tokens.UNKNOWN_WORD))

    debug_words_found = 0
    debug_words_count = 0
    for i, term in enumerate(terms):
        if isinstance(term, unicode):
            index = unknown_word_embedding_index
            words = word_embedding.Stemmer.lemmatize_to_list(term)
            if len(words) > 0:
                word = words[0]
                if word in word_embedding:
                    index = embedding_offsets.get_word_index(word_embedding.find_index_by_word(word))
                elif word in static_embedding:
                    index = embedding_offsets.get_word_index(static_embedding.find_index_by_word(word=word,
                                                                                                 return_unknown=True))
                debug_words_found += int(word in word_embedding)
                debug_words_count += 1
        elif isinstance(term, Token):
            index = embedding_offsets.get_token_index(TokenEmbeddingVectors.get_token_index(term.get_token_value()))
        elif isinstance(term, Entity):
            index = embedding_offsets.get_word_index(static_embedding.find_index_by_word(word=ENTITY_MASK,
                                                                                         return_unknown=False))
        else:
            raise Exception("Unsuported type {}".format(term))

        indices.append(index)

    if DebugKeys.EmbeddingIndicesPercentWordsFound:
        print "words found: {} ({}%)".format(debug_words_found, 100.0 * debug_words_found / debug_words_count)
        print "words missed: {} ({}%)".format(debug_words_count - debug_words_found,
                                              100.0 * (debug_words_count - debug_words_found) / debug_words_count)

    return indices


def calculate_pos_indices_for_terms(terms, pos_tagger):
    """
    terms: list
    pad_size: int

    returns: list of int
        list of pos indices
    """
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(pos_tagger, POSTagger))

    indices = []

    for index, term in enumerate(terms):
        if isinstance(term, Token):
            pos = pos_tagger.Empty
        elif isinstance(term, unicode):
            pos = pos_tagger.get_term_pos(term)
        else:
            pos = pos_tagger.Unknown

        indices.append(pos_tagger.pos_to_int(pos))

    return indices


def create_term_embedding_matrix(word_embedding, static_embedding):
    """
    Compose complete embedding matrix, which includes:
        - word embeddings
        - token embeddings

    word_embedding: Embedding
        embedding vocabulary for words
    static_embedding: StaticEmbedding
        embedding for missed words of word_vocabulary
    returns: np.ndarray(words_count, embedding_size)
        embedding matrix which includes embedding both for words and
        entities
    """
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(static_embedding, StaticEmbedding))

    embedding_offsets = TermsEmbeddingOffsets(words_count=len(word_embedding.vocab),
                                              static_words_count=len(static_embedding))
    token_embedding = TokenEmbeddingVectors(word_embedding.VectorSize)
    matrix = np.zeros((embedding_offsets.TotalCount, word_embedding.VectorSize))

    # words.
    for word, info in word_embedding.vocab.items():
        index = info.index
        matrix[embedding_offsets.get_word_index(index)] = word_embedding.get_vector_by_index(index)

    # missed words.
    for word, index in static_embedding.iter_word_with_index():
        matrix[embedding_offsets.get_static_word_index(index)] = static_embedding.get_vector_by_index(index)

    # tokens.
    for token_value in token_embedding:
        index = token_embedding.get_token_index(token_value)
        matrix[embedding_offsets.get_token_index(index)] = token_embedding[token_value]

    if DebugKeys.DisplayTermEmbeddingParameters:
        print "Term matrix shape: {}".format(matrix.shape)
        embedding_offsets.debug_print()

    # used as a placeholder
    matrix[0] = 0

    return matrix


class TermsEmbeddingOffsets:
    """
    Describes indices distibution within a further TermsEmbedding.
    """

    def __init__(self,
                 words_count,
                 static_words_count,
                 tokens_count=TokenEmbeddingVectors.count()):
        assert(isinstance(words_count, int))
        assert(isinstance(tokens_count, int))
        self.words_count = words_count
        self.static_words_count = static_words_count
        self.tokens_count = tokens_count

    @property
    def TotalCount(self):
        return self.static_words_count + self.words_count + self.tokens_count

    def get_word_index(self, index):
        return index

    def get_static_word_index(self, index):
        return self.words_count + index

    def get_token_index(self, index):
        return self.words_count + self.static_words_count + index

    def debug_print(self):
        print "Term embedding matrix details ..."
        print "\t\tWords count: {}".format(self.words_count)
        print "\t\tStatic (missed) words count: {}".format(self.static_words_count)
        print "\t\tTokens count: {}".format(self.tokens_count)
