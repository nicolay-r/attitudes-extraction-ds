from os import walk
from os.path import join
from random import shuffle

import io_utils
from networks.io import NetworkIO


class PretrainingIO(NetworkIO):

    __data_folder = join(io_utils.get_data_root(),
                         u"Pretrain")

    def __init__(self, model_to_pretrain_name):
        super(PretrainingIO, self).__init__(model_to_pretrain_name)

    def get_samples_filepaths(self, limit=3, shuffle_files=True):
        subfiles = self.__get_all_subfiles(self.__data_folder)
        if shuffle_files:
            shuffle(subfiles)
        return subfiles[:limit]

    @staticmethod
    def __get_all_subfiles(data_folder):
        filepaths = []
        for root, _, files in walk(data_folder):
            filepaths += map(lambda f: join(root, f), files)
        return sorted(filepaths)

    def get_word_embedding_filepath(self):
        return io_utils.get_rusvectores_news_embedding_filepath()
