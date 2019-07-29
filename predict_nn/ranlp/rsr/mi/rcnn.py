#!/usr/bin/python
import sys
sys.path.append('../../../../')
from predict_nn.ranlp.mi_names import ModelNames
from predict_nn.ranlp.rsr.config import TEST_ON_EPOCHS, MI_CONTEXTS_PER_OPINION
from networks.context.architectures.rcnn import RCNN
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO
from networks.mimlre.base import MIMLRE
from networks.ranlp.model_mimlre import RaNLPConfTaskMIMLREModel
from networks.context.configurations.rcnn import RCNNConfig
from networks.mimlre.configuration.base import MIMLRESettings
import predict_nn.ranlp.utils as utils


def modify_settings(settings):
    assert(isinstance(settings, MIMLRESettings))
    settings.modify_contexts_per_opinion(MI_CONTEXTS_PER_OPINION)


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.MI_RCNN,
                         create_network=lambda: MIMLRE(context_network=RCNN()),
                         create_config=lambda: MIMLRESettings(context_settings=RCNNConfig()),
                         create_io=RaNLPConfTaskRuSentRelIO,
                         create_model=RaNLPConfTaskMIMLREModel,
                         modify_settings_callback=modify_settings,
                         test_on_epochs=TEST_ON_EPOCHS,
                         cancel_training_by_cost=False)
