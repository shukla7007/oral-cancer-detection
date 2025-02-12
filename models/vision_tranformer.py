import tensorflow as tf
from tensorflow.keras import layers, models

def build_vit(input_shape=(150, 150, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Flatten the input
    x = layers.Flatten()(inputs)
    
    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    
    # Multi-head attention layer
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=x)
    return model
