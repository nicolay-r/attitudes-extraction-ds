#!/usr/bin/python
import sys
sys.path.append('../../../../')
from predict_nn.ranlp.rsr_dev.config import TEST_ON_EPOCHS, MI_CONTEXTS_PER_OPINION
from networks.context.architectures.rnn import RNN
from networks.context.configurations.rnn import RNNConfig
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO
from predict_nn.ranlp.mi_names import ModelNames
from networks.mimlre.base import MIMLRE
from networks.ranlp.model_mimlre import RaNLPConfTaskMIMLREModel
from networks.mimlre.configuration.base import MIMLRESettings
import predict_nn.ranlp.utils as utils


def modify_settings(settings):
    assert(isinstance(settings, MIMLRESettings))
    assert(isinstance(settings.ContextSettings, RNNConfig))
    settings.modify_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)
    settings.ContextSettings.modify_hidden_size(128)


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.MI_RNN,
                         create_network=lambda: MIMLRE(context_network=RNN()),
                         create_config=lambda: MIMLRESettings(context_settings=RNNConfig()),
                         create_io=RaNLPConfTaskRuSentRelWithDevIO,
                         create_model=RaNLPConfTaskMIMLREModel,
                         modify_settings_callback=modify_settings,
                         test_on_epochs=TEST_ON_EPOCHS)