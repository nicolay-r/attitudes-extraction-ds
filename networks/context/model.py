import numpy as np

from core.source.opinion import OpinionCollection

from networks.callback import Callback
from networks.cancellation import OperationCancellation
from networks.context.helpers.eval.opinion_based import OpinionBasedEvaluationHelper
from networks.context.helpers.initialization import ContextModelInitHelper
from networks.context.helpers.log import display_log
from networks.context.helpers.predict_log import PredictVariables
from networks.context.helpers.relations import ExtractedRelationsCollectionHelper
from networks.context.processing.relations.collection import ExtractedRelationsCollection
from networks.io import DataType
from networks.model import TensorflowModel
from networks.network import NeuralNetwork
from networks.context.configurations.base import CommonModelSettings

from processing.batch import MiniBatch

from debug import DebugKeys


class ContextLevelTensorflowModel(TensorflowModel):

    def __init__(self, io, network, settings, callback):
        assert(isinstance(settings, CommonModelSettings))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(callback, Callback) or callback is None)

        super(ContextLevelTensorflowModel, self).__init__(
            io=io, network=network, callback=callback)

        self.__settings = settings
        self.__init_helper = None
        self.__eval_helper = None
        self.prepare_sources()

    def prepare_sources(self):
        self.__init_helper = self.create_model_init_helper()
        self.__eval_helper = OpinionBasedEvaluationHelper()
        self.__print_statistic()

    @property
    def Settings(self):
        return self.__settings

    @property
    def ReadOnlySynonymsCollection(self):
        return self.__init_helper.Synonyms

    def get_bags_collection(self, data_type):
        return self.__init_helper.BagsCollections[data_type]

    def get_bags_collection_helper(self, data_type):
        return self.__init_helper.BagsCollectionHelpers[data_type]

    def get_relations_collection(self, data_type):
        return self.__init_helper.RelationsCollections[data_type]

    def get_relations_collection_helper(self, data_type):
        return self.__init_helper.RelationsCollectionHelpers[data_type]

    def get_gpu_memory_fraction(self):
        return self.__settings.GPUMemoryFraction

    def get_labels_helper(self):
        return self.__init_helper.LabelsHelper

    def get_eval_helper(self):
        return self.__eval_helper

    def set_optimiser(self):
        self.optimiser = self.Settings.Optimiser.minimize(self.network.Cost)

    def fit(self):
        assert(self.sess is not None)

        operation_cancel = OperationCancellation()
        minibatches = list(self.get_bags_collection(DataType.Train).iter_by_groups(self.Settings.BagsPerMinibatch))
        print "Minibatches passing per epoch count: {}".format(len(minibatches))

        for epoch_index in range(self.Settings.Epochs):

            if operation_cancel.IsCancelled:
                break

            self.get_bags_collection_helper(DataType.Train).print_log_statistics()

            total_cost = 0
            total_acc = 0
            groups_count = 0

            np.random.shuffle(minibatches)

            for bags_group in minibatches:

                minibatch = self.create_batch_by_bags_group(bags_group)
                feed_dict = self.create_feed_dict(minibatch, data_type=DataType.Train)

                var_names, var_tensors = self.network.Variables
                result = self.sess.run([self.optimiser, self.network.Cost, self.network.Accuracy] + var_tensors,
                                       feed_dict=feed_dict)
                cost = result[1]

                if DebugKeys.FitBatchDisplayLog:
                    display_log(var_names, result[3:])

                total_cost += np.mean(cost)
                total_acc += result[2]
                groups_count += 1

            if DebugKeys.FitSaveTensorflowModelState:
                self.save_model(save_path=self.IO.get_model_filepath())

            if self.callback is not None:
                self.callback.on_epoch_finished(avg_cost=total_cost / groups_count,
                                                avg_acc=total_acc / groups_count,
                                                epoch_index=epoch_index,
                                                operation_cancel=operation_cancel)

        if self.callback is not None:
            self.callback.on_fit_finished()

    def predict(self, dest_data_type=DataType.Test):
        eval_result, predict_log = self.predict_core(dest_data_type=dest_data_type,
                                                     rc_labeling_callback=self.__relations_labeling)
        return eval_result, predict_log

    def predict_core(self,
                     dest_data_type,
                     rc_labeling_callback):
        assert(isinstance(dest_data_type, unicode))
        assert(callable(rc_labeling_callback))

        rc = self.get_relations_collection(dest_data_type)
        rch = self.get_relations_collection_helper(dest_data_type)

        assert(isinstance(rc, ExtractedRelationsCollection))
        assert(isinstance(rch, ExtractedRelationsCollectionHelper))

        rc.reset_labels()
        assert(rc.check_all_relations_without_labels())

        predict_log = rc_labeling_callback(rc, dest_data_type)

        assert(rc.check_all_relations_has_labels())

        rch.debug_labels_statistic()

        rch.save_into_opinion_collections(
            create_opinion_collection=lambda: OpinionCollection(opinions=None,
                                                                synonyms=self.ReadOnlySynonymsCollection),
            create_filepath_by_news_id=lambda news_id: self.IO.get_model_doc_opins_filepath(doc_id=news_id,
                                                                                            data_type=dest_data_type),
            label_calculation_mode=self.Settings.RelationLabelCalculationMode)

        eval_result = self.get_eval_helper().evaluate_model(data_type=dest_data_type,
                                                            io=self.IO,
                                                            indices=rch.iter_unique_news_ids(),
                                                            synonyms=self.ReadOnlySynonymsCollection)

        rc.reset_labels()

        return eval_result, predict_log

    def __relations_labeling(self, relations, dest_data_type):
        assert(isinstance(dest_data_type, unicode))
        assert(isinstance(relations, ExtractedRelationsCollection))

        predict_log = PredictVariables()
        var_names, var_tensors = self.network.Variables

        for bags_group in self.get_bags_collection(dest_data_type).iter_by_groups(self.Settings.BagsPerMinibatch):

            minibatch = self.create_batch_by_bags_group(bags_group)
            feed_dict = self.create_feed_dict(minibatch, data_type=dest_data_type)

            result = self.sess.run([self.network.Labels] + var_tensors, feed_dict=feed_dict)
            uint_labels = result[0]

            predict_log.add(names=var_names,
                            results=result[1:],
                            relation_ids=[sample.RelationID for sample in minibatch.iter_by_samples()])

            if DebugKeys.PredictBatchDisplayLog:
                display_log(var_names, result[1:])

            # apply labels
            for bag_index, bag in enumerate(minibatch.iter_by_bags()):
                label = self.get_labels_helper().create_label_from_uint(label_uint=int(uint_labels[bag_index]))
                for sample in bag:
                    if sample.RelationID < 0:
                        continue
                    relations.apply_label(label, sample.RelationID)

        return predict_log

    def create_batch_by_bags_group(self, bags_group):
        return MiniBatch(bags_group)

    def create_model_init_helper(self):
        return ContextModelInitHelper(io=self.IO, settings=self.Settings)

    def create_feed_dict(self, minibatch, data_type):
        assert(isinstance(self.network, NeuralNetwork))
        assert(isinstance(minibatch, MiniBatch))
        assert(isinstance(data_type, unicode))

        input = minibatch.to_network_input()
        if DebugKeys.FeedDictShow:
            MiniBatch.debug_output(input)

        return self.network.create_feed_dict(input, data_type)

    def __print_statistic(self):
        keys, values = self.Settings.get_parameters()
        display_log(keys, values)
        self.get_relations_collection_helper(DataType.Train).debug_labels_statistic()
        self.get_relations_collection_helper(DataType.Train).debug_unique_relations_statistic()
        self.get_relations_collection_helper(DataType.Test).debug_labels_statistic()
        self.get_relations_collection_helper(DataType.Test).debug_unique_relations_statistic()
        self.get_bags_collection_helper(DataType.Train).print_log_statistics()
        self.get_bags_collection_helper(DataType.Test).print_log_statistics()
