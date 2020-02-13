"""The file has the mlp decoder
"""
import tensorflow as tf 

from model.multi_head_attn import feedforward

class MLPDecoder(tf.keras.Model):
    def __init__(self, hidden_size):
        super(MLPDecoder, self).__init__()
        self.ffn = feedforward(hidden_size, 2)
    
    def call(self, hidden_states):
        return self.ffn(hidden_states)
