#!/usr/bin/python
import sys
sys.path.append('../../../../')
from predict_nn.ranlp.ctx_names import ModelNames
from predict_nn.ranlp.rsr_dev.config import TEST_ON_EPOCHS
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO
from networks.ranlp.model_context import RaNLPConfTaskModel
from networks.context.configurations.rcnn import RCNNConfig
from networks.context.architectures.rcnn import RCNN
import predict_nn.ranlp.utils as utils


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.RCNN,
                         create_network=RCNN,
                         create_config=RCNNConfig,
                         create_io=RaNLPConfTaskRuSentRelWithDevIO,
                         create_model=RaNLPConfTaskModel,
                         test_on_epochs=TEST_ON_EPOCHS)
