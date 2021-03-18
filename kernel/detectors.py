import io
import tensorflow as tf
import face_recognition

from PIL import Image
from kernel.filters import *
from kernel.messages import *


class DeepFakeDetectorBase(object):
    PROBABILITY_THRESHOLD = 0.5
    model, resize_ratio = None, None

    def load_kernel(self, kernel_path: str):
        self.model.load_weights(kernel_path)

    def is_image_real(self, image: np.array) -> bool:
        assert image.shape == self.resize_ratio

        input_tensor = image.reshape(-1, *self.resize_ratio)
        probability = float(self.model(input_tensor))

        return probability >= self.PROBABILITY_THRESHOLD


class DeepFakeDetectorGray(DeepFakeDetectorBase):
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


class DeepFakeDetectorResNet(DeepFakeDetectorBase):
    def __init__(self, resize_ratio: tuple):
        self.resize_ratio = resize_ratio
        input_layer = tf.keras.layers.Input(shape=resize_ratio)
        b1_cnv2d_1 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=2, padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(input_layer)
        b1_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b1_cnv2d_1)

        b1_cnv2d_2 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=2, padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b1_bn_1)
        b1_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b1_cnv2d_2)

        b2_cnv2d_1 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b1_out)
        b2_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b2_cnv2d_1)

        b2_add = tf.keras.layers.Add()([b1_out, b2_bn_1])

        b2_cnv2d_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b2_add)
        b2_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b2_cnv2d_2)

        b3_cnv2d_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b2_out)
        b3_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b3_cnv2d_1)

        b3_add = tf.keras.layers.Add()([b2_out, b3_bn_1])

        b3_cnv2d_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b3_add)
        b3_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b3_cnv2d_2)

        b4_avg_p = tf.keras.layers.GlobalMaxPool1D()(b3_out)

        inter_layer_1 = tf.keras.layers.Dense(16, activation='elu',
                                              kernel_initializer='he_uniform')(b4_avg_p)

        inter_layer_2 = tf.keras.layers.Dense(16, activation='elu',
                                              kernel_initializer='he_uniform')(inter_layer_1)

        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(inter_layer_2)

        self.model = tf.keras.models.Model(input_layer, output_layer)
        self.model.compile()


class FaceScanner(object):
    def __init__(self, resize_ratio: tuple,
                 detector_kernel_path: str,
                 use_gray_filter=True):
        self.resize_ratio, self.use_gray_filter = resize_ratio, use_gray_filter

        self.detector = DeepFakeDetectorResNet(resize_ratio)
        self.detector.load_kernel(detector_kernel_path)

    def validate_image(self, image: io.BytesIO) -> tuple:
        image_array = face_recognition.load_image_file(image)
        positions = face_recognition.face_locations(image_array, model='cnn')

        if len(positions) == 0:
            return False, NO_FACES_DETECTED
        elif len(positions) == 1:
            face_position = positions[0]
            left, upper, right, lower = face_position[3], face_position[0], face_position[1], face_position[2]

            image_resized = np.array(Image.fromarray(
                image_array).crop((left, upper, right, lower)).resize(self.resize_ratio))

            is_image_real = self.detector.is_image_real(rgb2gray(image_resized / 255.0)) \
                if self.use_gray_filter \
                else self.detector.is_image_real(image_resized / 255.0)

            if is_image_real:
                return True, VALID_IMAGE
            else:
                return False, FAKE_DETECTED

        else:
            return False, MULTIPLE_FACES_DETECTED
