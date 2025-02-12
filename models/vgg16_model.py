from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16_model(input_shape=(150, 150, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    return model
