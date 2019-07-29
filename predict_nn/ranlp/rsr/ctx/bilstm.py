#!/usr/bin/python
import sys
sys.path.append('../../../../')
from predict_nn.ranlp.rsr.config import TEST_ON_EPOCHS
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO
from networks.ranlp.model_context import RaNLPConfTaskModel
from networks.context.configurations.bi_lstm import BiLSTMConfig
from predict_nn.ranlp.ctx_names import ModelNames
from networks.context.architectures.bi_lstm import BiLSTM
import predict_nn.ranlp.utils as utils


def modify_settings(settings):
    assert(isinstance(settings, BiLSTMConfig))
    settings.modify_hidden_size(128)


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.BiLSTM,
                         create_network=BiLSTM,
                         create_config=BiLSTMConfig,
                         create_model=RaNLPConfTaskModel,
                         create_io=RaNLPConfTaskRuSentRelIO,
                         modify_settings_callback=modify_settings,
                         test_on_epochs=TEST_ON_EPOCHS,
                         cancel_training_by_cost=False)
