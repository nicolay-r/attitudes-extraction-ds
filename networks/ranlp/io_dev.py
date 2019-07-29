from os import walk, path, makedirs
from os.path import join

import io_utils
from networks.context.helpers.cv import items_to_cv_pairs
from networks.ranlp.io_base import BaseAnswersIO


class RaNLPConfTaskDevIO(BaseAnswersIO):

    __cv_count = 10
    __etalon_root = join(io_utils.get_data_root(), u"ranlp/dev/opinions")
    __answers_root_template = join(io_utils.get_data_root(), u"ranlp/dev/answers/{}")
    __src_file = join(io_utils.get_data_root(), u"ranlp/sources/same.txt")
    __splitted_data_folder = join(io_utils.get_data_root(), u"ranlp/dev/splitted")

    def __init__(self, model_to_pretrain_name):
        super(RaNLPConfTaskDevIO, self).__init__(
            answers_root= self.__answers_root_template.format(model_to_pretrain_name),
            model_name=model_to_pretrain_name)
        self.__cv_index = 0

    @property
    def CVIndex(self):
        return self.__cv_index

    @property
    def SplittedDataFolder(self):
        if not path.exists(self.__splitted_data_folder):
            makedirs(self.__splitted_data_folder)
        return self.__splitted_data_folder

    @property
    def CVCount(self):
        return self.__cv_count

    @property
    def SourceFile(self):
        return self.__src_file

    def inc_cv_index(self):
        self.__cv_index += 1

    def get_train_test_paths(self):
        all_filepaths = sorted(RaNLPConfTaskDevIO.__get_all_subfiles(
            RaNLPConfTaskDevIO.__splitted_data_folder))
        train_test_pairs = list(items_to_cv_pairs(cv=self.__cv_count,
                                                  items_list=all_filepaths,
                                                  shuffle=False))
        return train_test_pairs[self.__cv_index]

    @staticmethod
    def __get_all_subfiles(data_folder):
        filepaths = []
        for root, _, files in walk(data_folder):
            filepaths += map(lambda f: join(root, f), files)
        return sorted(filepaths)

    def get_word_embedding_filepath(self):
        return io_utils.get_rusvectores_news_embedding_filepath()

    def get_etalon_root(self):
        # TODO. Use the same code.
        result = self.__etalon_root
        if not path.exists(result):
            makedirs(result)
        return result
