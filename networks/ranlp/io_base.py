from os import path, makedirs
from os.path import join

from networks.io import NetworkIO


class BaseAnswersIO(NetworkIO):

    __template = u"ans{}.csv"

    def __init__(self, answers_root, model_name):
        super(BaseAnswersIO, self).__init__(model_name)
        self.__answers_dir = answers_root
        self.__create_if_not_exist(self.__answers_dir)

    def __create_if_not_exist(self, dir):
        if not path.exists(dir):
            makedirs(dir)

    def get_answer_filepath(self, doc_id):
        assert(isinstance(doc_id, int))
        return join(self.__answers_dir, self.__template.format(doc_id))
