class EntityPosition:

    def __init__(self, term_index, sentence_index):
        self.__term_index = term_index
        self.__sentence_index = sentence_index

    @property
    def TermIndex(self):
        return self.__term_index

    @property
    def SentenceIndex(self):
        return self.__sentence_index

