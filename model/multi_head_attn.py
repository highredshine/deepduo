"""The file contains multi-head attention in tensorflow
"""

import tensorflow as tf
import numpy as np

from IPython import embed

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def scalar_dot_product_attention(q, v, k, mask):
    """The function computes self attention based on the equation in
    Attention is all you need paper. We take in the query, key and value 
    tensors with an mask tensor and compute the new attentive embeddings.
    """
    qk = tf.matmul(q, k, transpose_b=True)
    k_size = tf.cast(k.shape[-1], tf.float32)
    attn_logits = qk / np.sqrt(k_size)

    # adding mask if available (mask is 0/1)
    if mask is not None:
        attn_logits += (mask * -1e9)
    
    attn = tf.nn.softmax(attn_logits, axis=-1)

    new_embs = tf.matmul(attn, v)

    return new_embs


def feedforward(hidden_size, model_size):
    """function to get a sequential mdoel
    
    Arguments:
        hidden_size {int} -- the size of the hidden size
        model_size {int} -- the size of the final layer of the model
    
    Returns:
        tf.keras.Sequential -- sequential point wise feed forward n/w   
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(model_size)
    ])


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, no_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.no_of_heads = no_of_heads
        self.model_size = model_size
        self.depth = model_size // self.no_of_heads

        assert self.model_size % self.no_of_heads == 0

        self.wq = tf.keras.layers.Dense(self.model_size)
        self.wk = tf.keras.layers.Dense(self.model_size)
        self.wv = tf.keras.layers.Dense(self.model_size)

        self.final_layer = tf.keras.layers.Dense(self.model_size)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.no_of_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

#   def call(self, v, k, q, mask):
    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]
        
        query = self.wq(query)  
        key = self.wk(key)  
        value = self.wv(value)
        
        q = self.split_heads(query, batch_size)  
        k = self.split_heads(key, batch_size)  
        v = self.split_heads(value, batch_size)  
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = scalar_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.model_size))  # (batch_size, seq_len_q, d_model)

        output = self.final_layer(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


class ContextEncoder(tf.keras.Model):
    def __init__(self, model_size, no_of_heads, hidden_size, dropout=0.25):
        super(ContextEncoder, self).__init__()
        self.attention = MultiHeadAttention(model_size, no_of_heads)
        self.ffn = feedforward(hidden_size, model_size)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()

        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def call(self, _input, training, mask):
        attention_output = self.attention(_input, _input, _input, mask)
        attention_output = self.dropout_1(attention_output, training=training)
        
        # normalize the layers
        out_1 = self.layer_norm_1(_input + attention_output)

        fnn_output = self.ffn(out_1)
        fnn_output = self.dropout_2(fnn_output, training=training)

        out_2 = self.layer_norm_2(out_1 + fnn_output)

        return out_2
