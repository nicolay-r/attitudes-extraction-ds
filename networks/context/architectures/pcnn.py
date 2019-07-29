#!/usr/bin/python
import tensorflow as tf
from networks.context.architectures.cnn import VanillaCNN
from networks.context.configurations.cnn import CNNConfig
from networks.context.processing.sample import Sample
import utils


class PiecewiseCNN(VanillaCNN):

    @property
    def ContextEmbeddingSize(self):
        return 3 * self.cfg.FiltersCount + \
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

        # slice all data into 3 parts -- before, inner, and after according to relation
        sliced = tf.TensorArray(dtype=tf.float32, size=self.cfg.BatchSize, infer_shape=False, dynamic_size=True)
        _, _, _, _, _, sliced = tf.while_loop(
                lambda i, *_: tf.less(i, self.cfg.BatchSize),
                self.splitting,
                [0, self.p_subj_ind, self.p_obj_ind, bwc_conv, self.cfg.FiltersCount, sliced])
        sliced = tf.squeeze(sliced.concat())

        # Max Pooling
        bwgc_mpool = tf.nn.max_pool(
                sliced,
                [1, 1, self.cfg.TermsPerContext, 1],
                [1, 1, self.cfg.TermsPerContext, 1],
                padding='VALID',
                data_format="NHWC")

        bwc_mpool = tf.squeeze(bwgc_mpool, [2])
        bcw_mpool = tf.transpose(bwc_mpool, perm=[0, 2, 1])
        g = tf.reshape(bcw_mpool, [self.cfg.BatchSize, 3 * self.cfg.FiltersCount])

        if self.cfg.UseAttention:
            g = tf.concat([g, self.init_attention_embedding()], axis=1)

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
        super(PiecewiseCNN, self).init_hidden_states()
        self.W = tf.Variable(tf.random_normal([self.ContextEmbeddingSize, self.cfg.HiddenSize]), dtype=tf.float32)

    @staticmethod
    def splitting(i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs):
        l_ind = tf.minimum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # left
        r_ind = tf.maximum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # right

        w = tf.Variable(bwc_conv.shape[1], dtype=tf.int32) # total width (words count)

        b_slice_from = [i, 0, 0]
        b_slice_size = tf.concat([[1], l_ind, [channels_count]], 0)
        m_slice_from = tf.concat([[i], l_ind, [0]], 0)
        m_slice_size = tf.concat([[1], r_ind - l_ind, [channels_count]], 0)
        a_slice_from = tf.concat([[i], r_ind, [0]], 0)
        a_slice_size = tf.concat([[1], w-r_ind, [channels_count]], 0)

        bwc_split_b = tf.slice(bwc_conv, b_slice_from, b_slice_size)
        bwc_split_m = tf.slice(bwc_conv, m_slice_from, m_slice_size)
        bwc_split_a = tf.slice(bwc_conv, a_slice_from, a_slice_size)

        pad_b = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w-l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_m = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w-r_ind+l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_a = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([r_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        bwc_split_b = tf.pad(bwc_split_b, pad_b, constant_values=tf.float32.min)
        bwc_split_m = tf.pad(bwc_split_m, pad_m, constant_values=tf.float32.min)
        bwc_split_a = tf.pad(bwc_split_a, pad_a, constant_values=tf.float32.min)

        outputs = outputs.write(i, [[bwc_split_b, bwc_split_m, bwc_split_a]])

        i += 1
        return i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs

    @property
    def ParametersDictionary(self):
        return {}
