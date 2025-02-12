import numpy as np
from tensorflow.keras import layers, models

def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=latent_dim))
    model.add(layers.Reshape((16, 16, 1)))
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model

def build_discriminator(input_shape=(64, 64, 1)):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
