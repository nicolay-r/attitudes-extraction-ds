import tensorflow as tf

from networks.context.architectures.utils import get_two_layer_logits
from networks.context.configurations.attention.base import AttentionConfig


class Attention(object):

    def __init__(self, cfg, batch_size, terms_per_context, term_embedding_size):
        assert(isinstance(cfg, AttentionConfig))
        self.cfg = cfg
        self.batch_size = batch_size
        self.terms_per_context = terms_per_context
        self.term_embedding_size = term_embedding_size
        self.__x = None
        self.__entities = None

    @property
    def AttentionEmbeddingSize(self):
        return self.cfg.EntitiesPerContext * self.term_embedding_size

    def set_x(self, x):
        self.__x = x

    def set_entities(self, entities):
        self.__entities = entities

    def init_input(self):
        self.__x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.terms_per_context])
        self.__entities = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.cfg.EntitiesPerContext])

    def init_hidden(self):
        # init hidden
        self.W_we = tf.Variable(tf.random_normal([2 * self.term_embedding_size, self.cfg.HiddenSize]), dtype=tf.float32)
        self.b_we = tf.Variable(tf.random_normal([self.cfg.HiddenSize]), dtype=tf.float32)
        self.W_a = tf.Variable(tf.random_normal([self.cfg.HiddenSize, 1]), dtype=tf.float32)
        self.b_a = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

    def init_body(self, term_embedding):
        # embedded_terms: [batch_size, terms_per_context, embedding_size]
        embedded_terms = tf.nn.embedding_lookup(term_embedding, self.__x)

        with tf.name_scope("attention"):

            def iter_by_entities(entities, handler):
                # entities: [batch_size, entities]

                att_emb_array = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=self.cfg.EntitiesPerContext,
                    infer_shape=False,
                    dynamic_size=True)

                att_weights_array = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=self.cfg.EntitiesPerContext,
                    infer_shape=False,
                    dynamic_size=True)

                _, _, att_emb, att_weights = tf.while_loop(
                    lambda i, *_: tf.less(i, self.cfg.EntitiesPerContext),
                    handler,
                    (0, entities, att_emb_array, att_weights_array))

                return att_emb.stack(), att_weights.stack()

            def process_entity(i, entities, att_embeddings, att_weights):
                # entities: [batch_size, entities_per_context]

                e = tf.gather(entities, [i], axis=1)                       # [batch_size, 1] -- term positions
                e = tf.tile(e, [1, self.terms_per_context])                # [batch_size, terms_per_context]
                e = tf.nn.embedding_lookup(term_embedding, e)              # [batch_size, terms_per_context, embedding_size]

                merged = tf.concat([embedded_terms, e], axis=-1)
                merged = tf.reshape(merged, [self.batch_size * self.terms_per_context, 2 * self.term_embedding_size])

                weights, _ = get_two_layer_logits(merged,
                                                  W1=self.W_we, b1=self.b_we,
                                                  W2=self.W_a, b2=self.b_a,
                                                  dropout_keep_prob=self.cfg.DropoutKeepProb,
                                                  activations=[None,
                                                               lambda tensor: tf.tanh(tensor),
                                                               None])       # [batch_size * terms_per_context, 1]

                original_embedding = tf.reshape(embedded_terms,
                                                [self.batch_size * self.terms_per_context, self.term_embedding_size])

                weighted = tf.multiply(weights, original_embedding)
                weighted = tf.reshape(weighted, [self.batch_size, self.terms_per_context, self.term_embedding_size])
                weighted_sum = tf.reduce_sum(weighted, axis=1)             # [batch_size, embedding_size]
                weighted_sum = tf.nn.softmax(weighted_sum)

                return (i + 1,
                        entities,
                        att_embeddings.write(i, weighted_sum),
                        att_weights.write(i, tf.reshape(weights, [self.batch_size, self.terms_per_context])))

            att_e, att_w = iter_by_entities(self.__entities, process_entity)

            # att_e: [entity_per_context, batch_size, term_embedding_size]
            # att_w: [entity_per_context, batch_size, terms_per_context]

            att_e = tf.transpose(att_e, perm=[1, 0, 2])  # [batch_size, entity_per_context, term_embedding_size]
            att_e = tf.reshape(att_e, [self.batch_size, self.AttentionEmbeddingSize])

            att_w = tf.transpose(att_w, perm=[1, 0, 2])  # [batch_size, entity_per_context, terms_per_context]

        return att_e, att_w
