#!/usr/bin/python
import sys
sys.path.append('../../../../')
from networks.context.architectures.ian import IAN
from predict_nn.ranlp.rsr_dev.config import TEST_ON_EPOCHS
from networks.context.configurations.ian import IANConfig
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO
from networks.ranlp.model_context import RaNLPConfTaskModel
from predict_nn.ranlp.ctx_names import ModelNames
import predict_nn.ranlp.utils as utils


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.IAN,
                         create_network=IAN,
                         create_config=IANConfig,
                         create_model=RaNLPConfTaskModel,
                         create_io=RaNLPConfTaskRuSentRelWithDevIO,
                         test_on_epochs=TEST_ON_EPOCHS)
