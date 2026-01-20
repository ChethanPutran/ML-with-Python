import tensorflow as tf
from tensorflow import keras
from keras import layers, backend, optimizers, losses, Model, Input


class Sampling():
    def __call__(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = backend.random_normal(shape=(batch, dim))
        return mu+tf.exp(0.5*sigma)+epsilon


def encoding_layer(inputs, latent_dim):
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2,
                      padding='same', activation='relu', name='encode_conv1')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=2,
                      padding='same', activation='relu', name='encode_conv2')(inputs)
    batch_2 = layers.BatchNormalization()(x)

    x = layers.Flatten(name='encode_flatten')(batch_2)
    x = layers.Dense(20, activation='relu', name='encode_dense')(x)
    x = layers.BatchNormalization()(x)

    mu = layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = layers.Dense(latent_dim, name='latent_sigma')(x)

    return mu, sigma, batch_2.shape


def encoder_model(latent_dim, input_shape):
    inputs = layers.Input(shape=input_shape)
    mu, sigma, conv_shape = encoding_layer(inputs, latent_dim)

    samples = Sampling()(mu, sigma)
    model = Model(inputs, outputs=[mu, sigma, samples])

    return model, conv_shape


def decoding_layer(inputs, conv_shape):
    units = conv_shape[1]*conv_shape[2]*conv_shape[3]

    x = layers.Dense(units, activation='relu', name='decode_dense1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape(
        (conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")(x)

    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                               padding='same', activation='relu', name='decode_convt_1')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                               padding='same', activation='relu', name='decode_convt_2')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1,
                               padding='same', activation='sigmoid', name='decode_final')(x)

    return x


def decoder_model(latent_dim, conv_shape):
    inputs = layers.Input(shape=(latent_dim,))
    outputs = decoder_model(inputs, conv_shape)
    model = Model(inputs, outputs)
    return model


def kl_reconstruction_loss(mu, sigma):
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    return tf.reduce_mean(kl_loss) * - 0.5


def variation_auto_encoders(latent_dim, input_shape):
    encoder, conv_shape = encoder_model(latent_dim, input_shape)
    decoder = decoder_model(latent_dim, conv_shape)

    inputs = Input(shape=input_shape)
    mu, sigma, z = encoder(inputs)
    outputs = decoder(z)

    vae_model = Model(inputs=inputs, outputs=outputs)

    loss = kl_reconstruction_loss(mu, sigma)
    vae_model.add_loss(loss)

    return vae_model


def train(epocs, train_dataset, lr=0.01):
    latent_dim = []
    input_shape = []

    optimizer = optimizers.Adam(epsilon=lr)
    bce_loss = losses.BinaryCrossentropy()

    vae = variation_auto_encoders(latent_dim, input_shape)
    for epoc in range(epocs):
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                flattend_inputs = tf.reshape(x_batch_train, shape=[-1])
                flattend_outputs = tf.reshape(reconstructed, shape=[-1])
                loss = bce_loss(flattend_inputs, flattend_outputs) * 784
                loss += sum(vae.losses)  # Add KLD regularisation loss
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
