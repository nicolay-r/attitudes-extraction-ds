import hashlib

from core.runtime.utils import get_random_vector
from core.source.embeddings.base import Embedding


class StaticEmbedding(Embedding):

    __unknown_word = u"<UNKNOWN>"

    def __init__(self, vector_size):
        assert(isinstance(vector_size, int))
        super(StaticEmbedding, self).__init__()
        self.__word_indices = {}
        self.__vectors = []
        self.__vector_size = vector_size
        self.__create_and_add_embedding(word=self.__unknown_word)

    @property
    def VectorSize(self):
        return self.__vector_size

    def from_file(cls, filepath, binary, stemmer=None, pos_tagger=None):
        raise Exception("Unavailable for this type of embeddings")

    @property
    def vocab(self):
        raise Exception("Not available")

    def __create_and_add_embedding(self, word):
        assert(isinstance(word, unicode))
        assert(word not in self.__word_indices)

        h = hashlib.md5(word.encode('utf-8'))
        seed = int(h.hexdigest(), 16) % 2**32
        vector = get_random_vector(self.__vector_size, int(seed))
        index = len(self.__vectors)

        self.__word_indices[word] = index
        self.__vectors.append(vector)

        return vector

    def create_and_add_embedding(self, word):
        return self.__create_and_add_embedding(word)

    def get_vector_by_index(self, index):
        assert(isinstance(index, int))
        return self.__vectors[index]

    def get_word_by_index(self, index):
        assert(isinstance(index, int))
        assert(isinstance(self.__word_indices, dict))
        for word, i in self.__word_indices.iteritems():
            if i == index:
                return word
        raise Exception("Word has not been found")

    def find_index_by_word(self, word, return_unknown=False):
        assert(isinstance(word, unicode))

        if word in self.__word_indices:
            return self.__word_indices[word]

        if return_unknown:
            return self.__word_indices[self.__unknown_word]

        return None

    def iter_word_with_index(self):
        for word, index in self.__word_indices.iteritems():
            yield word, index

    def __contains__(self, word):
        assert(isinstance(word, unicode))
        return word in self.__word_indices

    def __getitem__(self, word):
        assert(isinstance(word, unicode))
        index = self.__word_indices[word]
        return self.__vectors[index]

    def __len__(self):
        return len(self.__vectors)


