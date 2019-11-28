import collections

import io_utils
from os import makedirs, path
from os.path import join, exists

from core.evaluation.utils import FilesToCompare


class DataType:
    """
    Describes collection types that supportes in
    current implementation, and provides by collections.
    """
    Train = u"train"
    Test = u"test"


class NetworkIO(object):
    """
    Now it includes IO to interact with collection,
    and it is specific towards RuSentiRel collection.
    """

    def __init__(self, model_name):
        assert(isinstance(model_name, unicode))
        self.__model_name = model_name

    @property
    def ModelName(self):
        return self.__model_name

    @staticmethod
    def get_synonyms_collection_filepath():
        return io_utils.get_synonyms_filepath()

    def iter_opinion_files_to_compare(self, data_type, etalon_root, indices):
        model_root = self.__get_model_root(data_type)
        return NetworkIO.__iter_opin_files_to_compare(indices=indices,
                                                      model_root=model_root,
                                                      etalon_root=etalon_root)

    def get_opinion_output_filepath(self, data_type, article_index):
        model_root = self.__get_model_root(data_type)
        return io_utils.get_rusentrel_format_sentiment_opin_filepath(index=article_index,
                                                                     is_etalon=False,
                                                                     root=model_root)

    def get_etalon_root(self):
        raise Exception("Not implemented")

    def get_word_embedding_filepath(self):
        raise Exception("Not impemented")

    def get_model_state_filepath(self):
        return join(self.get_model_states_dir(), u'{}.state'.format(self.__model_name))

    def get_model_states_dir(self):
        result_dir = join(self.__get_model_root(DataType.Train), u'States/')
        if not exists(result_dir):
            makedirs(result_dir)
        return result_dir

    def __get_model_root(self, model_name):
        model_name = self.__decorate_model_name(model_name)
        result = path.join(io_utils.get_eval_root(), model_name)
        if not path.exists(result):
            makedirs(result)
        return result

    def __decorate_model_name(self, data_type):
        if data_type == DataType.Test:
            return self.__model_name
        if data_type == DataType.Train:
            return u'{}_train'.format(self.__model_name)
        raise Exception("Unsupported data_type: {}".format(data_type))

    @staticmethod
    def __iter_opin_files_to_compare(indices, model_root, etalon_root):
        assert(isinstance(indices, collections.Iterable))
        assert(isinstance(model_root, unicode))
        assert(isinstance(etalon_root, unicode))

        for index in indices:
            files_to_compare = FilesToCompare(
                io_utils.get_rusentrel_format_sentiment_opin_filepath(index, is_etalon=False, root=model_root),
                io_utils.get_rusentrel_format_sentiment_opin_filepath(index, is_etalon=True, root=etalon_root),
                index)
            yield files_to_compare
