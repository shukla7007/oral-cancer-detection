import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Cropping2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

# Dataset paths (Update these with the correct paths)
dataset_path_cancer = '/Users/anshulshukla/Desktop/oral cancer detection/images 2/CANCER'
dataset_path_non_cancer = '/Users/anshulshukla/Desktop/oral cancer detection/images 2/NON CANCER'

def load_images_and_masks(dataset_path_cancer, dataset_path_non_cancer, target_size=(150, 150)):
    images = []
    masks = []

    # Load cancer images and create masks (1 for cancer)
    for img_name in os.listdir(dataset_path_cancer):
        img_path = os.path.join(dataset_path_cancer, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize image to the target size
            img = img / 255.0  # Normalize image
            images.append(img)
            masks.append(np.ones((target_size[0], target_size[1], 1)))  # Mask for cancer images is all ones (1)

    # Load non-cancer images and create masks (0 for non-cancer)
    for img_name in os.listdir(dataset_path_non_cancer):
        img_path = os.path.join(dataset_path_non_cancer, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize image to the target size
            img = img / 255.0  # Normalize image
            images.append(img)
            masks.append(np.zeros((target_size[0], target_size[1], 1)))  # Mask for non-cancer images is all zeros (0)

    return np.array(images), np.array(masks)

# Define U-Net model for segmentation with padding='same' to keep the output size consistent
def unet_model(input_size=(150, 150, 3)):
    inputs = Input(shape=input_size)

    # Encoder (Downsampling)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Decoder (Upsampling)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

    # Ensure output size matches input size
    x = Cropping2D(cropping=((1, 1), (1, 1)))(x)  # Crop to get the output size to (150, 150)

    # Output layer (binary segmentation)
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# At the end of the training script, add:
model.save('/Users/anshulshukla/Desktop/oral cancer detection/models/segmentation_model.h5')
