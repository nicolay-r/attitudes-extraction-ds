import tensorflow as tf
from networks.context.architectures.base import BaseContextNeuralNetwork
from networks.context.architectures.rnn import RNN
from networks.context.configurations.rcnn import RCNNConfig
import utils

# Copyright (c) Joohong Lee
# page: https://github.com/roomylee


class RCNN(BaseContextNeuralNetwork):

    @property
    def ContextEmbeddingSize(self):
        return self.cfg.HiddenSize + \
               self._get_attention_vector_size(self.cfg)

    def __text_embedding_size(self):
        return self.TermEmbeddingSize + \
               2 * self.cfg.SurroundingOneSideContextEmbeddingSize

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.cfg, RCNNConfig))
        text_length = utils.length(self.x)

        with tf.name_scope("bi-rnn"):
            fw_cell = RNN.get_cell(self.cfg.SurroundingOneSideContextEmbeddingSize, self.cfg.CellType)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = RNN.get_cell(self.cfg.SurroundingOneSideContextEmbeddingSize, self.cfg.CellType)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=embedded_terms,
                                                                                       sequence_length=text_length,
                                                                                       dtype=tf.float32)

        with tf.name_scope("context"):
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]]
            c_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name="context_left")
            c_right = tf.concat([self.output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word-representation"):
            merged = tf.concat([c_left, embedded_terms, c_right], axis=2, name="merged")

        with tf.name_scope("text-representation"):
            y2 = tf.tanh(tf.einsum('aij,jk->aik', merged, self.W1) + self.b1)

        with tf.name_scope("max-pooling"):
            y3 = tf.reduce_max(y2, axis=1)

        if self.cfg.UseAttention:
            y3 = tf.concat([y3, self.init_attention_embedding()], axis=-1)

        return y3

    def init_logits_unscaled(self, context_embedding):

        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            l2_loss += tf.nn.l2_loss(self.W2)
            l2_loss += tf.nn.l2_loss(self.b2)
            logits = tf.nn.xw_plus_b(context_embedding, self.W2, self.b2, name="logits")

        return logits, tf.nn.dropout(logits, self.dropout_keep_prob)

    def init_hidden_states(self):
        assert(isinstance(self.cfg, RCNNConfig))

        self.W1 = tf.Variable(tf.random_uniform([self.__text_embedding_size(), self.cfg.HiddenSize], -1.0, 1.0), name="W1")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.cfg.HiddenSize]), name="b1")

        self.W2 = tf.get_variable("W2",
                                  shape=[self.ContextEmbeddingSize, self.cfg.ClassesCount],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.get_variable("b2",
                                  shape=[self.cfg.ClassesCount],
                                  initializer=tf.contrib.layers.xavier_initializer())
