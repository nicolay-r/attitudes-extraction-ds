import tensorflow as tf
from networks.context.architectures.base import BaseContextNeuralNetwork
from networks.context.configurations.rnn import RNNConfig, CellTypes

import utils

# Copyright (c) Joohong Lee
# page: https://github.com/roomylee
# source project: https://github.com/roomylee/rnn-text-classification-tf


class RNN(BaseContextNeuralNetwork):

    @property
    def ContextEmbeddingSize(self):
        return self.cfg.HiddenSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.cfg, RNNConfig))

        with tf.name_scope("rnn"):
            length = tf.cast(utils.length(self.x), tf.int32)
            cell = self.get_cell(self.cfg.HiddenSize, self.cfg.CellType)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=embedded_terms,
                                               sequence_length=length,
                                               dtype=tf.float32)
            h_outputs = self.last_relevant(all_outputs, length)

        return h_outputs

    def init_logits_unscaled(self, context_embedding):

        with tf.name_scope("output"):
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)
            logits = tf.nn.xw_plus_b(context_embedding, self.W, self.b, name="logits")

        return logits, tf.nn.dropout(logits, self.dropout_keep_prob)

    def init_hidden_states(self):
        self.W = tf.get_variable("W",
                                 shape=[self.ContextEmbeddingSize, self.cfg.ClassesCount],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.Variable(tf.constant(0.1, shape=[self.cfg.ClassesCount]),
                             name="b")

    @staticmethod
    def get_cell(hidden_size, cell_type):
        assert(isinstance(cell_type, unicode))
        if cell_type == CellTypes.RNN:
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == CellTypes.LSTM:
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == CellTypes.GRU:
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            Exception("Incorrect cell_type={}".format(cell_type))
            return None

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)

    @property
    def Variables(self):

        params = [("dropout", self.dropout_keep_prob),
                  # ("x", self.x),
                  # ("y", self.y),
                  # ("p_obj", self.p_obj),
                  # ("p_subj", self.p_subj),
                  # ("h_outputs", self.h_outputs),
                  ("weights", self.weights),
                  ("labels", self.Labels)]

        return [list(p) for p in zip(*params)]

    @property
    def ParametersDictionary(self):
        return {}
