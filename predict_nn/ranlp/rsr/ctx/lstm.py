#!/usr/bin/python
import sys
sys.path.append('../../../../')
from predict_nn.ranlp.rsr.config import TEST_ON_EPOCHS
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO
from networks.ranlp.model_context import RaNLPConfTaskModel
from networks.context.configurations.rnn import RNNConfig, CellTypes
from predict_nn.ranlp.ctx_names import ModelNames
from networks.context.architectures.rnn import RNN
import predict_nn.ranlp.utils as utils


def modify_settings(settings):
    assert(isinstance(settings, RNNConfig))
    settings.set_cell_type(CellTypes.LSTM)
    settings.modify_hidden_size(128)


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.RNN,
                         create_network=RNN,
                         create_config=RNNConfig,
                         create_model=RaNLPConfTaskModel,
                         create_io=RaNLPConfTaskRuSentRelIO,
                         modify_settings_callback=modify_settings,
                         test_on_epochs=TEST_ON_EPOCHS,
                         cancel_training_by_cost=False)
