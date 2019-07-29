import collections

import io_utils
from os import makedirs, path
from os.path import join, exists

from core.evaluation.utils import FilesToCompareUtils


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

    def iter_opinion_files_to_compare(self, data_type, indices):
        return FilesToCompareUtils.get_list_of_comparable_files(
            test_filepath_func=lambda doc_id: self.get_model_doc_opins_filepath(doc_id=doc_id,
                                                                                data_type=data_type),
            etalon_filepath_func=lambda doc_id: self.get_etalon_doc_opins_filepath(doc_id),
            indices=indices)

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

    def get_etalon_doc_opins_filepath(self, doc_id):
        assert(isinstance(doc_id, int))
        return io_utils.get_rusentrel_format_sentiment_opin_filepath(index=doc_id,
                                                                     is_etalon=True,
                                                                     root=self.get_etalon_root())

    def get_model_doc_opins_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))
        return io_utils.get_rusentrel_format_sentiment_opin_filepath(index=doc_id,
                                                                     is_etalon=False,
                                                                     root=self.__get_model_root(data_type))

    def __iter_opin_files_to_compare(self, indices, model_root):
        assert(isinstance(indices, collections.Iterable))
        assert(isinstance(model_root, unicode))

