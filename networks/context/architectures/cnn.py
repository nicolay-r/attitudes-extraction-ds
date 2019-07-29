import tensorflow as tf
from networks.context.architectures.base import BaseContextNeuralNetwork
from networks.context.configurations.cnn import CNNConfig
import utils


class VanillaCNN(BaseContextNeuralNetwork):

    @property
    def ContextEmbeddingSize(self):
        return self.cfg.FiltersCount + \
               self._get_attention_vector_size(self.cfg)

    def init_context_embedding(self, embedded_terms):
        embedded_terms = self.padding(embedded_terms, self.cfg.WindowSize)

        bwc_line = tf.reshape(embedded_terms,
                              [self.cfg.BatchSize,
                               (self.cfg.TermsPerContext + (self.cfg.WindowSize - 1)) * self.TermEmbeddingSize,
                               1])

        bwc_conv = tf.nn.conv1d(bwc_line, self.conv_filter, self.TermEmbeddingSize,
                                "VALID",
                                data_format="NHWC",
                                name="conv")

        bwgc_conv = tf.reshape(bwc_conv, [self.cfg.BatchSize,
                                          1,
                                          self.cfg.TermsPerContext,
                                          self.cfg.FiltersCount])

        # Max Pooling
        bwgc_mpool = tf.nn.max_pool(
                bwgc_conv,
                [1, 1, self.cfg.TermsPerContext, 1],
                [1, 1, self.cfg.TermsPerContext, 1],
                padding='VALID',
                data_format="NHWC")

        bc_mpool = tf.squeeze(bwgc_mpool, axis=[1, 2])

        g = tf.reshape(bc_mpool, [self.cfg.BatchSize, self.cfg.FiltersCount])

        if self.cfg.UseAttention:
            g = tf.concat([g, self.init_attention_embedding()], axis=-1)

        return tf.concat(g, axis=-1)

    def init_logits_unscaled(self, context_embedding):
        return utils.get_two_layer_logits(
            context_embedding,
            self.W, self.bias,
            self.W2, self.bias2,
            self.dropout_keep_prob,
            activations=[tf.tanh, tf.tanh, None])

    def init_hidden_states(self):
        assert(isinstance(self.cfg, CNNConfig))
        self.W = tf.Variable(tf.random_normal([self.ContextEmbeddingSize, self.cfg.HiddenSize]), dtype=tf.float32)
        self.bias = tf.Variable(tf.random_normal([self.cfg.HiddenSize]), dtype=tf.float32)
        self.W2 = tf.Variable(tf.random_normal([self.cfg.HiddenSize, self.cfg.ClassesCount]), dtype=tf.float32)
        self.bias2 = tf.Variable(tf.random_normal([self.cfg.ClassesCount]), dtype=tf.float32)
        self.conv_filter = tf.Variable(tf.random_normal([self.cfg.WindowSize * self.TermEmbeddingSize, 1, self.cfg.FiltersCount]), dtype=tf.float32)

    @staticmethod
    def padding(embedded_data, window_size):
        assert(isinstance(window_size, int) and window_size > 0)

        left_padding = (window_size - 1) / 2
        right_padding = (window_size - 1) - left_padding
        return tf.pad(embedded_data, [[0, 0],
                                      [left_padding, right_padding],
                                      [0, 0]])

    @property
    def ParametersDictionary(self):
        return {}
