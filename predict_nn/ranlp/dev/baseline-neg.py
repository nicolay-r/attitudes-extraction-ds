#!/usr/bin/python
import sys
sys.path.append('../../../')
from predict_nn.ranlp.dev.config import TEST_ON_EPOCHS
from networks.ranlp.io_dev import RaNLPConfTaskDevIO
from networks.ranlp.model_baseline_neg import RaNLPConfTaskBaselineNegModel
from networks.context.configurations.cnn import CNNConfig
from networks.context.architectures.cnn import VanillaCNN
import predict_nn.ranlp.utils as utils


if __name__ == "__main__":

    utils.run_cv_testing(model_name=u"baseline-neg",
                         create_network=VanillaCNN,
                         create_config=CNNConfig,
                         create_io=RaNLPConfTaskDevIO,
                         create_model=RaNLPConfTaskBaselineNegModel,
                         test_on_epochs=TEST_ON_EPOCHS)

