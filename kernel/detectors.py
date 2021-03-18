import cv2
import base64
import tensorflow as tf

from kernel.filters import *
from kernel.messages import *


def to_base64(image: np.array) -> bytes:
    _, buffer_img = cv2.imencode('.png', image)
    return base64.b64encode(buffer_img)


class DeepFakeDetectorBase(object):
    PROBABILITY_THRESHOLD = 0.5
    model, resize_ratio = None, None

    def load_kernel(self, kernel_path: str):
        self.model.load_weights(kernel_path)

    def is_image_real(self, image: np.array) -> bool:
        assert image.shape == self.resize_ratio

        input_tensor = image.reshape(1, *self.resize_ratio)
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

        b1_cnv2d_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal')(input_layer)
        b1_relu_1 = tf.keras.layers.ReLU()(b1_cnv2d_1)
        b1_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b1_relu_1)

        b1_cnv2d_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal')(b1_bn_1)
        b1_relu_2 = tf.keras.layers.ReLU()(b1_cnv2d_2)
        b1_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b1_relu_2)

        b2_cnv2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                            use_bias=False, kernel_initializer='normal')(b1_out)
        b2_relu_1 = tf.keras.layers.ReLU()(b2_cnv2d_1)
        b2_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b2_relu_1)

        b2_add = tf.keras.layers.Add()([b1_out, b2_bn_1])

        b2_cnv2d_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal')(b2_add)
        b2_relu_2 = tf.keras.layers.ReLU()(b2_cnv2d_2)
        b2_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b2_relu_2)

        b3_cnv2d_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                            use_bias=False, kernel_initializer='normal')(b2_out)
        b3_relu_1 = tf.keras.layers.ReLU()(b3_cnv2d_1)
        b3_bn_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b3_relu_1)

        b3_add = tf.keras.layers.Add()([b2_out, b3_bn_1])

        b3_cnv2d_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                            use_bias=False, kernel_initializer='normal')(b3_add)
        b3_relu_2 = tf.keras.layers.ReLU()(b3_cnv2d_2)
        b3_out = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(b3_relu_2)

        b4_avg_p = tf.keras.layers.GlobalAveragePooling2D()(b3_out)

        inter_layer_1 = tf.keras.layers.Dense(16, activation='softmax',
                                              kernel_initializer='he_uniform')(b4_avg_p)

        inter_layer_2 = tf.keras.layers.Dense(16, activation='softmax',
                                              kernel_initializer='he_uniform')(inter_layer_1)

        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(inter_layer_2)

        self.model = tf.keras.models.Model(input_layer, output_layer)
        self.model.compile()


class FaceScanner(object):
    # Red color
    INCORRECT_COLOR = (0, 0, 255)

    # Green color
    CORRECT_COLOR = (0, 255, 0)

    def __init__(self, resize_ratio: tuple,
                 scanner_kernel_path: str,
                 detector_kernel_path: str,
                 use_gray_filter=False,
                 use_crop=False, crop_offset=0,
                 scale_factor=1.1, min_neighbors=5,
                 min_size=(64, 64), border_thickness=3):
        self.resize_ratio = resize_ratio
        self.scanner = cv2.CascadeClassifier(scanner_kernel_path)
        self.scale_factor, self.min_neighbors, self.min_size = scale_factor, min_neighbors, min_size
        self.border_thickness, self.use_crop, self.crop_offset = border_thickness, use_crop, crop_offset
        self.use_gray_filter = use_gray_filter

        self.detector = DeepFakeDetectorResNet(resize_ratio)
        self.detector.load_kernel(detector_kernel_path)

    def validate_image(self, image: np.array) -> tuple:
        cv_image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        cv_image_grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.scanner.detectMultiScale(cv_image_grayscale,
                                              scaleFactor=self.scale_factor,
                                              minNeighbors=self.min_neighbors,
                                              minSize=self.min_size)

        if len(faces) == 0:
            height, width, _ = cv_image.shape
            cv2.rectangle(cv_image, (0, 0), (width, height), self.INCORRECT_COLOR, self.border_thickness)

            return False, NO_FACES_DETECTED, to_base64(cv_image)
        elif len(faces) == 1:
            x, y, width, height = faces[0]

            cv_image_cropped = cv_image[x - self.crop_offset:x + width + self.crop_offset,
                                        y - self.crop_offset:y + height + self.crop_offset]

            if self.use_crop:
                cv_image_resized = cv2.resize(cv_image_cropped, (self.resize_ratio[0], self.resize_ratio[1]))
            else:
                cv_image_resized = cv2.resize(cv_image, (self.resize_ratio[0], self.resize_ratio[1]))

            is_image_real = self.detector.is_image_real(rgb2gray(cv_image_resized)) \
                if self.use_gray_filter \
                else self.detector.is_image_real(cv_image_resized)

            if is_image_real:
                return True, VALID_IMAGE, to_base64(cv_image_cropped) if self.use_crop else to_base64(cv_image)
            else:
                cv2.rectangle(cv_image, (x, y), (x + width, y + height), self.INCORRECT_COLOR, self.border_thickness)
                return False, FAKE_DETECTED, to_base64(cv_image)

        else:
            for (x, y, width, height) in faces:
                cv2.rectangle(cv_image, (x, y), (x + width, y + height), self.INCORRECT_COLOR, self.border_thickness)

            return False, MULTIPLE_FACES_DETECTED, to_base64(cv_image)
