#!/usr/bin/python
import sys
sys.path.append('../../../../')
from predict_nn.ranlp.rsr_dev.config import MI_CONTEXTS_PER_OPINION
from predict_nn.ranlp.mi_names import ModelNames
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO
from networks.mimlre.base import MIMLRE
from networks.ranlp.model_mimlre import RaNLPConfTaskMIMLREModel
from networks.context.configurations.cnn import CNNConfig
from networks.mimlre.configuration.base import MIMLRESettings
from networks.context.architectures.cnn import VanillaCNN
import predict_nn.ranlp.utils as utils


def modify_settings(settings):
    assert(isinstance(settings, MIMLRESettings))
    settings.modify_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.MI_CNN,
                         create_network=lambda: MIMLRE(context_network=VanillaCNN()),
                         create_config=lambda: MIMLRESettings(context_settings=CNNConfig()),
                         create_io=RaNLPConfTaskRuSentRelWithDevIO,
                         create_model=RaNLPConfTaskMIMLREModel,
                         modify_settings_callback=modify_settings,
                         test_on_epochs=range(0, 1500, 10),
                         cancel_training_by_cost=False)
