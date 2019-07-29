#!/usr/bin/python
import sys
sys.path.append('../../../../')
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO
from networks.ranlp.model_context import RaNLPConfTaskModel
from networks.context.configurations.cnn import CNNConfig
from predict_nn.ranlp.ctx_names import ModelNames
from networks.context.architectures.cnn import VanillaCNN
import predict_nn.ranlp.utils as utils


if __name__ == "__main__":

    utils.run_cv_testing(model_name=ModelNames.CNN,
                         create_network=VanillaCNN,
                         create_config=CNNConfig,
                         create_io=RaNLPConfTaskRuSentRelWithDevIO,
                         create_model=RaNLPConfTaskModel,
                         test_on_epochs=range(0, 1500, 10),
                         cancel_training_by_cost=False)

