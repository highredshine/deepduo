"""The file contains the final encoder-decoder model for training
our transformer architecture
"""
import tensorflow as tf

from model.multi_head_attn import ContextEncoder, positional_encoding
from model.mlp_decoder import MLPDecoder

from IPython import embed

from preprocess import MAX_TOKEN_SIZE

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_size, model_size, no_of_heads, hidden_size):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.model_size = model_size
        self.emb_size = emb_size
        self.no_of_heads = no_of_heads
        self.hidden_size = hidden_size

        # TODO: use glove
        self.embs = tf.keras.layers.Embedding(vocab_size, emb_size, mask_zero=False)
        # self.meta_feat = tf.keras.layers.Embedding(feat_size, 1)
        self.context_encoder = ContextEncoder(model_size, no_of_heads, hidden_size)
        self.pos_encoder = positional_encoding(MAX_TOKEN_SIZE, model_size)

    def call(self, x, mask, training=False):
        # x.shape == (batch_size, max_length_of_seq)
        # meta_x.shape == (batch_size, max_length_of_seq, no_of_features)

        length_of_seq = tf.shape(x)[1]

        # load embeddings
        x = self.embs(x)

        # add positional encoding information (this is not learned) 
        # TODO: verify if this is actually correct
        x *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        x += self.pos_encoder[:, :length_of_seq, :]

        # TODO: add dropout if necessary
        encoding = self.context_encoder(x, training, mask)
        
        return encoding