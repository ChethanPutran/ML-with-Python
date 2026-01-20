import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


def encoder(inputs):
    x = layers.Conv2D(filters=64, kernel_size=3,
                      padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3,
                      padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    return x


def bottle_neck(inputs):
    bottle_neck = layers.Conv2D(
        filters=256, kernel_size=3, padding='same', activation='relu')(inputs)
    encoder_viz = layers.Conv2D(
        filters=1, kernel_size=3, padding='same', activation='sigmoid')(bottle_neck)

    return bottle_neck, encoder_viz


def decoder(inputs):
    x = layers.Conv2DTranspose(
        filters=128, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.UpSampling2D(pool_size=2)(x)

    x = layers.Conv2DTranspose(
        filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(pool_size=2)(x)

    x = layers.Conv2DTranspose(
        filters=1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return x


def convolutional_auto_encoder():
    inputs = layers.Input(shape=(28, 28, 1))
    encoder_output = encoder(inputs)
    bottle_neck_output, encoder_visualization = bottle_neck(encoder_output)
    decoder_output = decoder(bottle_neck_output)

    auto_encoder = Model(inputs=inputs, outputs=decoder_output)
    encoder_model = Model(inputs=inputs, outputs=encoder_visualization)
    return auto_encoder, encoder_model
