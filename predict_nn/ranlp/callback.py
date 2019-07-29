import datetime
from core.evaluation.results.base import DocumentCompareTable
from core.evaluation.results.two_class import TwoClassEvalResult
from networks.cancellation import OperationCancellation
from networks.context.debug import DebugKeys
from networks.io import DataType
from networks.ranlp.io_dev import RaNLPConfTaskDevIO
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO
from predict_nn.callback import BaseCallback


class RaNLPTaskCallback(BaseCallback):

    def __init__(self, test_on_epochs, log_dir, io, cv_filepath, cancel_by_cost):
        assert(isinstance(io, RaNLPConfTaskDevIO) or
               isinstance(io, RaNLPConfTaskRuSentRelIO) or
               isinstance(io, RaNLPConfTaskRuSentRelWithDevIO))
        assert(isinstance(cv_filepath, unicode))
        assert(isinstance(cancel_by_cost, bool))

        super(RaNLPTaskCallback, self).__init__(test_on_epochs=test_on_epochs,
                                                log_dir=log_dir,
                                                use_logger=False)
        self.__cv_filepath = cv_filepath
        self.__test_on_epochs = test_on_epochs
        self.__cv_file = None
        self.__io = io
        self.__results = [None] * io.CVCount
        self.__avg_costs = []
        self.__cancel_by_cost = cancel_by_cost

    def on_fit_finished(self):
        self.__cv_file.write("===========\n")
        self.__cv_file.write("Avg. test F-1: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_F1)))
        self.__cv_file.write("-----------\n")
        self.__cv_file.write("Avg. test P_R: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_POS_RECALL)))
        self.__cv_file.write("Avg. test N_R: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_NEG_RECALL)))
        p_r = self.__get_avg_stat(TwoClassEvalResult.C_POS_RECALL)
        n_r = self.__get_avg_stat(TwoClassEvalResult.C_NEG_RECALL)
        self.__cv_file.write("R(P,N):        {}\n".format(round(1.0 * (p_r + n_r) / 2, 2)))
        self.__cv_file.write("-----------\n")
        self.__cv_file.write("Avg. test P_P: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_POS_PREC)))
        self.__cv_file.write("Avg. test N_P: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_NEG_PREC)))
        p_p = self.__get_avg_stat(TwoClassEvalResult.C_POS_PREC)
        n_p = self.__get_avg_stat(TwoClassEvalResult.C_NEG_PREC)
        self.__cv_file.write("P(P,N):        {}\n".format(round(1.0 * (p_p + n_p) / 2, 2)))
        self.__cv_file.write("===========\n")

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

        if epoch_index == 0:
            self.__avg_costs = []

        if epoch_index not in self.__test_on_epochs:
            return

        if avg_acc >= 0.99:
            print "[Cancelling]: avg_acc"
            operation_cancel.cancel()

        cancel_by_cost = False
        if len(self.__avg_costs) > 0 and self.__cancel_by_cost:
            if avg_cost >= self.__avg_costs[-1]:
                print "[Cancelling]: avg_cost is greater than prior value"
                cancel_by_cost = True
                operation_cancel.cancel()

        self.__avg_costs.append(avg_cost)

        result_test, log_test = self.Model.predict(dest_data_type=DataType.Test)
        self.__print_results(eval_result=result_test,
                             epoch_index=epoch_index,
                             data_type=DataType.Test,
                             avg_acc=avg_acc,
                             avg_cost=avg_cost)

        if self.__results[self.__io.CVIndex] is None or not cancel_by_cost:
            self.__results[self.__io.CVIndex] = result_test
            self.__save_answers(result_test)

    def __save_answers(self, result):
        assert(isinstance(result, TwoClassEvalResult))

        for doc_id, cmp_table in result.iter_document_cmp_tables():
            assert(isinstance(cmp_table, DocumentCompareTable))
            cmp_table.save(filepath=self.__io.get_answer_filepath(doc_id))

    def __print_results(self, eval_result, epoch_index, data_type, avg_acc, avg_cost):
        assert(isinstance(eval_result, TwoClassEvalResult))
        assert(isinstance(data_type, unicode))
        assert(isinstance(epoch_index, int))

        self.__cv_file.write("[{}/{}] E:{} R({}): ".format(
            self.__io.CVIndex,
            self.__io.CVCount,
            epoch_index,
            data_type,
            eval_result.Result))

        for meric_name, value in eval_result.Result.iteritems():
            self.__cv_file.write("{}: {}; ".format(meric_name, round(float(value), 2)))

        self.__cv_file.write("avg-cost: {}; ".format(avg_cost))
        self.__cv_file.write("avg-acc: {}; ".format(avg_acc))
        self.__cv_file.write("\n")

    def __print_avg_stat(self, result_metric_name):
        assert(isinstance(result_metric_name, unicode))
        avg_value = self.__get_avg_stat(result_metric_name)
        return round(avg_value, 2)

    def __get_avg_stat(self, result_metric_name):

        avg_f1 = 0.0
        total_count = 0
        for eval_result in self.__results:

            if eval_result is None:
                continue

            assert(isinstance(eval_result, TwoClassEvalResult))

            total_count += 1
            avg_f1 += eval_result.Result[result_metric_name]

        return avg_f1 / total_count

    def __enter__(self):
        self.__cv_file = open(self.__cv_filepath, "w", buffering=0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__cv_file.close()
