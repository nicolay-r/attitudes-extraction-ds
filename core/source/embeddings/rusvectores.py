from core.source.embeddings.base import Embedding
from gensim.models.word2vec import Word2Vec


class RusvectoresEmbedding(Embedding):

    def __init__(self, w2v_model, stemmer, pos_tagger):
        assert(isinstance(w2v_model, Word2Vec))
        super(RusvectoresEmbedding, self).__init__(w2v_model, stemmer, pos_tagger)

    def __contains__(self, term):
        assert(isinstance(term, unicode))

        item = self.__lemmatize_term_to_rusvectores(term)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__contains__(item)

    def __getitem__(self, term):
        assert(isinstance(term, unicode))

        item = self.__lemmatize_term_to_rusvectores(term)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).__getitem__(item)

    def find_index_by_word(self, word, return_unknown=False):
        assert(isinstance(word, unicode))
        assert(return_unknown is False)

        item = self.__lemmatize_term_to_rusvectores(word)
        if item is None:
            return False
        return super(RusvectoresEmbedding, self).find_index_by_word(item)

    def __lemmatize_term_to_rusvectores(self, term):
        """
        Combine lemmatized 'text' with POS tag (part of speech).
        """
        assert(isinstance(term, unicode))

        term = self._stemmer.lemmatize_to_str(term)
        pos = self._pos_tagger.get_term_pos(term)
        if pos is self._pos_tagger.Unknown:
            return None
        return '_'.join([term, pos])


