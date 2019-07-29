from networks.context.configurations.cnn import CNNConfig


class PretrainingCNNSettings(CNNConfig):
    """
    Modification of CNN config.
    """

    def __init__(self):
        super(PretrainingCNNSettings, self).__init__()
