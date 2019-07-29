from os import path, makedirs
from os.path import join

import io_utils
from networks.ranlp.io_base import BaseAnswersIO
from networks.ranlp.io_dev import RaNLPConfTaskDevIO
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO


class RaNLPConfTaskRuSentRelWithDevIO(BaseAnswersIO):

    __etalon_root = join(io_utils.get_data_root(), u"ranlp/rsr_dev/opinions")
    __answers_root_template = join(io_utils.get_data_root(), u"ranlp/rsr_dev/answers/{}")

    def __init__(self, model_to_pretrain_name):
        super(RaNLPConfTaskRuSentRelWithDevIO, self).__init__(
            answers_root=self.__answers_root_template.format(model_to_pretrain_name),
            model_name=model_to_pretrain_name)
        self.__cv_index = 0
        self.__io_rsr = RaNLPConfTaskRuSentRelIO(model_to_pretrain_name)
        self.__io_dev = RaNLPConfTaskDevIO(model_to_pretrain_name)

    @property
    def RuSentRelIO(self):
        return self.__io_rsr

    @property
    def DevIO(self):
        return self.__io_dev

    @property
    def CVIndex(self):
        return self.__cv_index

    @property
    def CVCount(self):
        return self.__io_rsr.CVCount

    @property
    def SourceFile(self):
        return None

    @property
    def SourceDataFolder(self):
        return None

    @property
    def SplittedDataFolder(self):
        return None

    def inc_cv_index(self):
        self.__cv_index += 1
        self.__io_rsr.inc_cv_index()
        self.__io_dev.inc_cv_index()

    def get_train_test_paths(self):
        rsr_train, rsr_test = self.__io_rsr.get_train_test_paths()
        dev_train, dev_test = self.__io_dev.get_train_test_paths()
        return rsr_train + dev_test + dev_train, rsr_test

    def get_word_embedding_filepath(self):
        return io_utils.get_rusvectores_news_embedding_filepath()

    def get_etalon_root(self):
        # TODO. Use the same code. Duplicate
        result = self.__etalon_root
        if not path.exists(result):
            makedirs(result)
        return result

    def iter_test_answers(self):
        pass
