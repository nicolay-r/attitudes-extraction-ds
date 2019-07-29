from networks.context.model import ContextLevelTensorflowModel
from networks.io import DataType
from networks.pretraining.helpers.initialization import ModelInitHelper
from networks.pretraining.io import PretrainingIO


class TensorflowContextPretrainModel(ContextLevelTensorflowModel):

    def __init__(self, io, network, settings, callback):
        assert(isinstance(io, PretrainingIO))

        self.__train_helper = None

        super(TensorflowContextPretrainModel, self).__init__(
            io=io, network=network, settings=settings, callback=callback)

    def prepare_sources(self):
        self.__train_helper = ModelInitHelper(samples_filepaths=self.IO.get_samples_filepath(),
                                              word_embedding_filepath=self.IO.get_word_embedding_filepath(),
                                              settings=self.Settings,
                                              data_type=DataType.Train)

    def reinit(self):
        self.prepare_sources()

    @property
    def ReadOnlySynonymsCollection(self):
        return self.__init_helper.Synonyms

    def get_bags_collection(self, data_type):
        assert(data_type == DataType.Train)
        return self.__train_helper.BagsCollection

    def get_relations_collection(self, data_type):
        assert(data_type == DataType.Train)
        return self.__train_helper.RelationsCollection

    def get_bags_collection_helper(self, data_type):
        assert(data_type == DataType.Train)
        return self.__train_helper.BagsCollectionHelper

    def get_relations_collection_helper(self, data_type):
        assert(data_type == DataType.Train)
        return self.__train_helper.RelationsCollectionHelper

    def get_labels_helper(self):
        return self.__train_helper.LabelsHelper
