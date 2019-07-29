from os import path
from core.evaluation.results.two_class import TwoClassEvalResult
from networks.callback import Callback
from networks.cancellation import OperationCancellation
from networks.context.architectures.base import ContextNetworkVariableNames
from networks.context.helpers.predict_log import PredictVariables
from networks.logger import CSVLogger
from networks.model import TensorflowModel
from networks.io import DataType
from networks.context.debug import DebugKeys
import datetime


class BaseCallback(Callback):

    TrainingLogName = 'stat.csv'
    ParamsTemplateLogName = 'params_{}_{}.csv'
    PredictVerbosePerFileStatistic = True

    def __init__(self,
                 test_on_epochs,
                 log_dir,
                 use_logger=True):
        self.logger = CSVLogger.create(test_on_epochs) if use_logger else None
        self.__model = None
        self.__test_on_epochs = test_on_epochs
        self.__log_dir = log_dir

    @property
    def Model(self):
        return self.__model

    def on_initialized(self, model):
        assert(isinstance(model, TensorflowModel))
        self.__model = model

    def on_epoch_finished(self, avg_cost, avg_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_cost, float))
        assert(isinstance(avg_acc, float))
        assert(isinstance(operation_cancel, OperationCancellation))

        if DebugKeys.FitEpochCompleted:
            print "{}: Epoch: {}: avg. cost: {:.3f}, avg. acc.: {:.3f}".format(
                str(datetime.datetime.now()),
                epoch_index,
                avg_cost,
                avg_acc)

        if epoch_index not in self.__test_on_epochs:
            return

        result_test, log_test = self.__model.predict(dest_data_type=DataType.Test)
        result_train, log_train = self.__model.predict(dest_data_type=DataType.Train)

        if self.logger is not None:
            self.logger.write_evaluation_results(
                current_epoch=epoch_index,
                result_test=result_test,
                result_train=result_train,
                avg_cost=avg_cost,
                avg_acc=avg_acc)

            self.logger.df.to_csv(path.join(self.__log_dir, self.TrainingLogName))

        if self.PredictVerbosePerFileStatistic:
            self.__print_verbose_eval_results(result_test, DataType.Test)
            self.__print_verbose_eval_results(result_train, DataType.Train)

        self.__print_overall_results(result_test, DataType.Test)
        self.__print_overall_results(result_train, DataType.Train)

        assert(isinstance(log_test, PredictVariables))
        assert(isinstance(log_train, PredictVariables))

        if log_test.has_variable(ContextNetworkVariableNames.AttentionWeights):
            self.print_log_values(predict_log=log_test,
                                  data_type=DataType.Test,
                                  epoch_index=epoch_index,
                                  value_key=ContextNetworkVariableNames.AttentionWeights)

    @staticmethod
    def __print_verbose_eval_results(eval_result, data_type):
        assert(isinstance(eval_result, TwoClassEvalResult))

        print "Verbose statistic for {}:".format(data_type)
        for doc_id, result in eval_result.iter_document_results():
            print doc_id, result

    @staticmethod
    def __print_overall_results(eval_result, data_type):
        assert(isinstance(eval_result, TwoClassEvalResult))

        print "Overall statistic for {}:".format(data_type)
        for metric, value in eval_result.Result.iteritems():
            print "\t{}: {}".format(metric, round(value, 2))

    def print_log_values(self, data_type, epoch_index, value_key, predict_log):
        assert(isinstance(predict_log, PredictVariables))

        filepath = path.join(self.__log_dir,
                             self.ParamsTemplateLogName.format(
                                 '{}-{}'.format(value_key, data_type),
                                 epoch_index))

        with open(filepath, 'w') as out:
            for relation_id, value in predict_log.iter_by_value(value_key):
                out.write('{}: {}\n'.format(relation_id, value))
