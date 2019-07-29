"""
Bi-directional Recurrent Neural Network.
Modified version of Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import tensorflow as tf
from tensorflow.contrib import rnn

from networks.context.architectures.base import BaseContextNeuralNetwork
from networks.context.configurations.bi_lstm import BiLSTMConfig

import utils


class BiLSTM(BaseContextNeuralNetwork):

    @property
    def ContextEmbeddingSize(self):
        return 2 * self.cfg.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.cfg, BiLSTMConfig))

        x = tf.unstack(embedded_terms, axis=1)
        lstm_fw_cell = BiLSTM.__get_cell(self.cfg.HiddenSize)
        lstm_bw_cell = BiLSTM.__get_cell(self.cfg.HiddenSize)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,
                                                     output_keep_prob=self.dropout_keep_prob)

        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                     lstm_bw_cell,
                                                     x,
                                                     dtype=tf.float32)
        return outputs[-1]

    def init_logits_unscaled(self, context_embedding):
        return utils.get_single_layer_logits(context_embedding, self.W, self.b, self.dropout_keep_prob)

    def init_hidden_states(self):
        self.W = tf.Variable(tf.random_normal([self.ContextEmbeddingSize, self.cfg.ClassesCount]))
        self.b = tf.Variable(tf.random_normal([self.cfg.ClassesCount]))

    @staticmethod
    def __get_cell(hidden_size):
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    @property
    def ParametersDictionary(self):
        return {}
