
import tensorflow as tf
import numpy as np


from model.encoder import Encoder
from model.mlp_decoder import MLPDecoder

from IPython import embed

class EncoderDecoder(tf.keras.Model):
    def __init__(self, vocab_size, mappings, emb_size, model_size, no_of_heads, 
                 hidden_size, mlp_hidden_size):
        super(EncoderDecoder, self).__init__()

        # TODO: add meta data feature
        self.encoder = Encoder(vocab_size, emb_size, model_size, no_of_heads, hidden_size)

        #
        self.mappings = mappings

        self.user_emb = tf.keras.layers.Embedding(len(self.mappings[0]), 20)
        
        # country
        self.country_emb = tf.keras.layers.Embedding(len(self.mappings[1]), 5)

        # session type
        self.session_type = tf.keras.layers.Embedding(len(self.mappings[3]), 5)

        # exercise format
        self.ex_format = tf.keras.layers.Embedding(len(self.mappings[4]), 5)

        # pos
        self.pos_emb = tf.keras.layers.Embedding(len(self.mappings[5]), 5)

        # dependency edge label
        self.dep_edge_label = tf.keras.layers.Embedding(len(self.mappings[6]), 5)

        # TODO: linear model for morphological features
        self.morphological_weights = tf.Variable(tf.random.truncated_normal((1, 1, len(mappings[7])), stddev=0.1))

        self.decoder = MLPDecoder(mlp_hidden_size)

        self.optimizer = tf.keras.optimizers.Adam()
    
    def call(self, x, metadata, mask, training=False):
        hidden_states = self.encoder(x, mask, training)

        # metadata features 
        # TODO: need to verify the correct order the feats
        user_idx = metadata[:, :, 0:1].squeeze(-1)
        countries_idx = metadata[:, :, 1:2].squeeze(-1)
        session_idx = metadata[:, :, 4:5].squeeze(-1)
        format_idx = metadata[:, :, 5:6].squeeze(-1)
        pos_idx = metadata[:, :, 7:8].squeeze(-1)
        dep_idx = metadata[:, :, 8:9].squeeze(-1)
        morph_idx = metadata[:, :, 10:]

        # convert to dense features
        user_feat = self.user_emb(user_idx)
        countries_feat = self.user_emb(countries_idx)
        session_feat = self.session_type(session_idx)
        ex_feat = self.ex_format(format_idx)
        pos_feat = self.pos_emb(pos_idx)
        dep_feat = self.dep_edge_label(dep_idx)

        # 
        morph_feat = morph_idx.astype(np.float32) * self.morphological_weights
        
        meta_feat = tf.concat([user_feat, countries_feat, session_feat, ex_feat, pos_feat, dep_feat, morph_feat], axis=2)

        encoded = tf.concat([hidden_states, meta_feat], axis=2)

        logits = self.decoder(encoded)

        return logits

    def loss_function(self, logits, labels, mask):
        """The loss function computes the loss using cross entropy
        
        Arguments:
            logits {np.array} -- [description]
            labels {np.array} -- the true labels (with padding)
            mask {np.array} -- mask indicating the padding
        
        Returns:
            float -- the batch loss
        """
        new_mask = tf.cast(tf.math.logical_not(tf.math.equal(mask, 1)), dtype=tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) * new_mask

        return tf.reduce_sum(loss)
