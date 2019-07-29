#!/usr/bin/python
import sys
sys.path.append('../../../../')
from predict_nn.ranlp.dev.config import TEST_ON_EPOCHS
from networks.ranlp.io_dev import RaNLPConfTaskDevIO
from networks.ranlp.model_context import RaNLPConfTaskModel
from networks.context.configurations.cnn import CNNConfig
from predict_nn.ranlp.ctx_names import ModelNames
from networks.context.architectures.cnn import VanillaCNN
import predict_nn.ranlp.utils as utils


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.CNN,
                         create_network=VanillaCNN,
                         create_config=CNNConfig,
                         create_io=RaNLPConfTaskDevIO,
                         create_model=RaNLPConfTaskModel,
                         test_on_epochs=TEST_ON_EPOCHS)
