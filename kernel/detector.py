import numpy as np
import tensorflow as tf


class DeepFakeDetectorGray(object):
    PROBABILITY_THRESHOLD = 0.5

    def __init__(self, resize_ratio: tuple):
        self.resize_ratio = resize_ratio
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=resize_ratio))
        self.model.add(tf.keras.layers.MaxPooling1D(2))
        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(2))
        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(2))
        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))

        self.model.compile()

    def load_kernel(self, kernel_path: str):
        self.model.load_weights(kernel_path)

    def is_image_real(self, image: np.array) -> bool:
        assert image.shape == (self.resize_ratio, self.resize_ratio)

        input_tensor = image.reshape(1, self.resize_ratio, self.resize_ratio)
        probability = float(self.model(input_tensor))

        return probability >= self.PROBABILITY_THRESHOLD
