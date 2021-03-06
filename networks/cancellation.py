class OperationCancellation(object):

    def __init__(self):
        self.__is_cancelled = False

    @property
    def IsCancelled(self):
        return self.__is_cancelled

    def cancel(self):
        self.__is_cancelled = True
