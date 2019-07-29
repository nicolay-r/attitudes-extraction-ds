from os.path import join

from networks.context.helpers.cv import items_to_cv_pairs
from networks.io import NetworkIO, DataType
import io_utils


class RuSentRelNetworkIO(NetworkIO):
    """
    Represents Input interface for NeuralNetwork context
    Now exploited (treated) as input interface only
    """

    def __init__(self, model_name):
        super(RuSentRelNetworkIO, self).__init__(model_name)

    @staticmethod
    def get_entity_filepath(article_index):
        return io_utils.get_rusentrel_entity_filepath(index=article_index,
                                                      root=io_utils.get_rusentrel_collection_root())

    @staticmethod
    def get_news_filepath(article_index):
        return io_utils.get_rusentrel_news_filepath(index=article_index,
                                                    root=io_utils.get_rusentrel_collection_root())

    def get_word_embedding_filepath(self):
        return io_utils.get_rusvectores_news_embedding_filepath()

    @staticmethod
    def get_neutral_filepath(article_index, data_type):
        if data_type == DataType.Test:
            return io_utils.get_rusentrel_neutral_opin_filepath(
                index=article_index,
                is_train=False,
                root=io_utils.get_rusentrel_collection_root())
        if data_type == DataType.Train:
            return io_utils.get_rusentrel_neutral_opin_filepath(
                index=article_index,
                is_train=True,
                root=io_utils.get_rusentrel_collection_root())

    def get_model_filepath(self):
        return join(self.get_model_states_dir(), u'{}'.format(self.ModelName))

    def get_etalon_root(self):
        return io_utils.get_rusentrel_collection_root()

    def get_data_indices(self, data_type):
        if data_type == DataType.Test:
            return io_utils.get_rusentrel_test_indices()
        if data_type == DataType.Train:
            return io_utils.get_rusentrel_train_indices()
