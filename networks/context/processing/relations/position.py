from networks.context.processing.terms.position import EntityPosition


class ExtractedRelationPosition:
    """
    Represents an article sample by given newsID,
    and [left, right] entities positions
    """

    def __init__(self, news_id, left, right):
        assert(isinstance(news_id, int))    # news index, which is a part of news filename
        assert(isinstance(left, EntityPosition))
        assert(isinstance(right, EntityPosition))
        self.__news_id = news_id
        self.__left = left
        self.__right = right

    @property
    def NewsID(self):
        return self.__news_id

    @property
    def LeftEntityPosition(self):
        return self.__left

    @property
    def RightEntityPosition(self):
        return self.__right
