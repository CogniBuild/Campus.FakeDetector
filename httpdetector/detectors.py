import io
import tensorflow as tf
import face_recognition

from PIL import Image
from .filters import *
from .settings import *


class DeepFakeDetectorBase(object):
    PROBABILITY_THRESHOLD = 0.5
    model, resize_ratio = None, None

    def __init__(self, probability_threshold: float):
        self.PROBABILITY_THRESHOLD = probability_threshold

    def load_kernel(self, kernel_path: str):
        self.model.load_weights(kernel_path)

    def is_image_real(self, image: np.array) -> bool:
        assert image.shape == self.resize_ratio

        input_tensor = image.reshape(-1, *self.resize_ratio)
        probability = float(self.model(input_tensor))

        return probability >= self.PROBABILITY_THRESHOLD


class DeepFakeDetectorResNet2D(DeepFakeDetectorBase):
    def __init__(self, resize_ratio: tuple, probability_threshold: float):
        super().__init__(probability_threshold)
        self.resize_ratio = resize_ratio

        input_layer = tf.keras.layers.Input(shape=self.resize_ratio)

        b1_cnv2d_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(input_layer)
        b1_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b1_cnv2d_1)

        b1_cnv2d_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b1_bn_1)
        b1_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b1_cnv2d_2)

        b2_cnv2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b1_out)
        b2_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b2_cnv2d_1)

        b2_add = tf.keras.layers.Add()([b1_out, b2_bn_1])

        b2_cnv2d_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b2_add)
        b2_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b2_cnv2d_2)

        b3_cnv2d_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b2_out)
        b3_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b3_cnv2d_1)

        b3_add = tf.keras.layers.Add()([b2_out, b3_bn_1])

        b3_cnv2d_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal', activation='elu')(b3_add)
        b3_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b3_cnv2d_2)

        b4_avg_p = tf.keras.layers.GlobalMaxPooling2D()(b3_out)

        inter_layer_1 = tf.keras.layers.Dense(512, activation='elu',
                                              kernel_initializer='he_uniform')(b4_avg_p)
        dropout_layer_1 = tf.keras.layers.Dropout(0.2)(inter_layer_1)

        inter_layer_2 = tf.keras.layers.Dense(512, activation='elu',
                                              kernel_initializer='he_uniform')(dropout_layer_1)
        dropout_layer_2 = tf.keras.layers.Dropout(0.2)(inter_layer_2)

        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_layer_2)

        self.model = tf.keras.models.Model(input_layer, output_layer)
        self.model.compile()


class FaceScanner(object):
    def __init__(self, resize_ratio: tuple, detector_kernel_path: str,
                 use_gray_filter=False, use_crop=False, cropping_offset=0,
                 probability_threshold=0.5):
        self.resize_ratio, self.use_gray_filter = resize_ratio, use_gray_filter
        self.cropping_offset, self.use_crop = cropping_offset, use_crop

        self.detector = DeepFakeDetectorResNet2D(resize_ratio, probability_threshold)
        self.detector.load_kernel(detector_kernel_path)

    def validate_image(self, image: io.BytesIO) -> tuple:
        image_array = face_recognition.load_image_file(image)
        positions = face_recognition.face_locations(image_array, model='hog')

        if len(positions) == 0:
            return "No faces detected on the photo", {
                "no-faces": 1,
                "multiple-faces": 0,
                "deep-fake": 0
            }
        elif len(positions) == 1:
            face_position = positions[0]
            left, upper, right, lower = face_position[3] - self.cropping_offset, \
                face_position[0] - self.cropping_offset, \
                face_position[1] + self.cropping_offset, \
                face_position[2] + self.cropping_offset

            image_resized = np.array(Image.fromarray(
                image_array).crop((left, upper, right, lower)).resize((self.resize_ratio[0], self.resize_ratio[1])))

            if self.use_crop:
                is_image_real = self.detector.is_image_real(rgb2gray(image_resized / 255.0)) \
                    if self.use_gray_filter \
                    else self.detector.is_image_real(image_resized / 255.0)
            else:
                image_array = np.array(Image.fromarray(
                    image_array).resize((self.resize_ratio[0], self.resize_ratio[1])))

                is_image_real = self.detector.is_image_real(rgb2gray(image_array / 255.0)) \
                    if self.use_gray_filter \
                    else self.detector.is_image_real(image_array / 255.0)

            if is_image_real:
                return "Photo passed all preliminary checks", {
                    "no-faces": 0,
                    "multiple-faces": 0,
                    "deep-fake": 0
                }
            else:
                return "Detected face on photo is fake", {
                    "no-faces": 0,
                    "multiple-faces": 0,
                    "deep-fake": 1
                }

        else:
            return "Multiple faces detected on the photo", {
                "no-faces": 0,
                "multiple-faces": 1,
                "deep-fake": 0
            }


face_scanner = FaceScanner(RESIZE_RATIO, DETECTOR_KERNEL_FILE_PATH, PROBABILITY_THRESHOLD)
