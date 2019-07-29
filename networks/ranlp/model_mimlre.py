import io_utils
from core.source.opinion import OpinionCollection
from core.source.synonyms import SynonymsCollection
from networks.context.configurations.base import LabelCalculationMode
from networks.context.helpers.eval.opinion_based import OpinionBasedEvaluationHelper
from networks.context.helpers.log import display_log
from networks.context.helpers.relations import ExtractedRelationsCollectionHelper
from networks.io import DataType
from networks.mimlre.model import MIMLRETensorflowModel
from networks.pretraining.helpers.initialization import ModelInitHelper
from networks.ranlp.io_dev import RaNLPConfTaskDevIO
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO


class RaNLPConfTaskMIMLREModel(MIMLRETensorflowModel):

    def __init__(self, io, network, settings, callback):
        assert(isinstance(io, RaNLPConfTaskDevIO) or
               isinstance(io, RaNLPConfTaskRuSentRelIO) or
               isinstance(io, RaNLPConfTaskRuSentRelWithDevIO))

        self.__train_helper = None
        self.__test_helper = None
        self.__eval_helper = OpinionBasedEvaluationHelper()
        self.__synonyms = SynonymsCollection.from_file(filepath=io_utils.get_synonyms_filepath(),
                                                       stemmer=settings.Stemmer,
                                                       is_read_only=True)

        super(RaNLPConfTaskMIMLREModel, self).__init__(
            io=io, network=network, settings=settings, callback=callback)

    @property
    def ReadOnlySynonymsCollection(self):
        return self.__synonyms

    def prepare_sources(self):

        train_files, test_files = self.IO.get_train_test_paths()
        print "Train files: ", train_files
        print "Test files: ", test_files

        self.__train_helper = ModelInitHelper(samples_filepaths=train_files,
                                              word_embedding_filepath=self.IO.get_word_embedding_filepath(),
                                              settings=self.Settings,
                                              data_type=DataType.Train)

        self.__test_helper = ModelInitHelper(samples_filepaths=test_files,
                                             word_embedding_filepath=self.IO.get_word_embedding_filepath(),
                                             settings=self.Settings,
                                             data_type=DataType.Test)

        print "Saving train collections ..."
        self.__save_etalon(self.__train_helper.RelationsCollectionHelper)
        print "Saving test collections ..."
        self.__save_etalon(self.__test_helper.RelationsCollectionHelper)

        norm, _ = self.get_relations_collection_helper(DataType.Train).get_statistic()
        self.Settings.set_class_weights(norm)

        keys, values = self.Settings.get_parameters()
        display_log(keys, values)

    def __save_etalon(self, relation_collection_helper):
        assert(isinstance(relation_collection_helper, ExtractedRelationsCollectionHelper))

        relation_collection_helper.save_into_opinion_collections(
            create_opinion_collection=lambda: OpinionCollection(opinions=None,
                                                                synonyms=self.ReadOnlySynonymsCollection),
            create_filepath_by_news_id=lambda news_id: self.IO.get_etalon_doc_opins_filepath(news_id),
            label_calculation_mode=LabelCalculationMode.FIRST_APPEARED)

    def get_bags_collection(self, data_type):
        if data_type == DataType.Train:
            return self.__train_helper.BagsCollection
        elif data_type == DataType.Test:
            return self.__test_helper.BagsCollection

    def get_relations_collection(self, data_type):
        if data_type == DataType.Train:
            return self.__train_helper.RelationsCollection
        elif data_type == DataType.Test:
            return self.__test_helper.RelationsCollection

    def get_relations_collection_helper(self, data_type):
        if data_type == DataType.Train:
            return self.__train_helper.RelationsCollectionHelper
        elif data_type == DataType.Test:
            return self.__test_helper.RelationsCollectionHelper

    def get_bags_collection_helper(self, data_type):
        if data_type == DataType.Train:
            return self.__train_helper.BagsCollectionHelper
        elif data_type == DataType.Test:
            return self.__test_helper.BagsCollectionHelper

    def get_labels_helper(self):
        return self.__train_helper.LabelsHelper

    def get_eval_helper(self):
        return self.__eval_helper

