import os
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from networks.callback import Callback
from io import NetworkIO, DataType
from networks.network import NeuralNetwork


class TensorflowModel(object):
    """
    Base model class, which provides api for
        - tensorflow model compilation
        - fitting
        - training
        - load/save states during fitting/training
        and more.
    """

    def __init__(self, io, network, callback=None):
        assert(isinstance(io, NetworkIO))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(callback, Callback) or callback is None)
        self.sess = None
        self.__saver = None
        self.optimiser = None
        self.__io = io
        self.network = network
        self.callback = callback

    @property
    def Settings(self):
        """
        Should provide following properties:
        """
        raise Exception("Not Implemented")

    @property
    def IO(self):
        return self.__io

    def get_gpu_memory_fraction(self):
        raise Exception("Not Implemented")

    def __notify_initialized(self):
        if self.callback is not None:
            self.callback.on_initialized(self)

    def load_model(self, save_path):
        assert(isinstance(self.__saver, Saver))
        save_dir = os.path.dirname(save_path)
        self.__saver.restore(sess=self.sess,
                             save_path=tf.train.latest_checkpoint(save_dir))

    def save_model(self, save_path):
        assert(isinstance(self.__saver, Saver))
        self.__saver.save(self.sess,
                          save_path=save_path,
                          write_meta_graph=False)

    def __initialize_session(self):
        """
        Tensorflow session initialization
        """
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.get_gpu_memory_fraction())
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_op)
        self.__saver = tf.train.Saver(max_to_keep=2)
        self.sess = sess

    def dispose_session(self):
        """
        Tensorflow session dispose method
        """
        self.sess.close()

    def run(self, load_model=False):
        self.network.compile(self.Settings, reset_graph=True)
        self.set_optimiser()
        self.__notify_initialized()

        self.__initialize_session()

        if load_model:
            save_path = self.__io.get_model_state_filepath()
            print "Loading model: {}".format(save_path)
            self.load_model(save_path)

        self.fit()
        self.dispose_session()

    def fit(self):
        raise Exception("Not implemented")

    def predict(self, dest_data_type=DataType.Test):
        raise Exception("Not implemented")

    def set_optimiser(self):
        raise Exception("Not implemented")
