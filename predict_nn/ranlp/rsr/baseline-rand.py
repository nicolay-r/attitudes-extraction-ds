#!/usr/bin/python
import sys
sys.path.append('../../../')
from predict_nn.ranlp.rsr.config import TEST_ON_EPOCHS
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO
from networks.ranlp.model_baseline_rand import RaNLPConfTaskBaselineRandModel
from networks.context.configurations.cnn import CNNConfig
from networks.context.architectures.cnn import VanillaCNN
import predict_nn.ranlp.utils as utils


if __name__ == "__main__":

    utils.run_cv_testing(model_name=u"baseline-rand",
                         create_network=VanillaCNN,
                         create_config=CNNConfig,
                         create_io=RaNLPConfTaskRuSentRelIO,
                         create_model=RaNLPConfTaskBaselineRandModel,
                         test_on_epochs=TEST_ON_EPOCHS)