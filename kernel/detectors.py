import cv2
import tensorflow as tf

from kernel.filters import *
from kernel.messages import *


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
        assert image.shape == self.resize_ratio

        input_tensor = image.reshape(1, *self.resize_ratio)
        probability = float(self.model(input_tensor))

        return probability >= self.PROBABILITY_THRESHOLD


class FaceScanner(object):
    # Red color
    INCORRECT_COLOR = (0, 0, 255)

    # Green color
    CORRECT_COLOR = (0, 255, 0)

    def __init__(self, resize_ratio: tuple,
                 scanner_kernel_path: str,
                 detector_kernel_path: str,
                 use_crop=False, crop_offset=0,
                 scale_factor=1.1, min_neighbors=5,
                 min_size=(10, 10), border_thickness=3):
        self.resize_ratio = resize_ratio
        self.scanner = cv2.CascadeClassifier(scanner_kernel_path)
        self.scale_factor, self.min_neighbors, self.min_size = scale_factor, min_neighbors, min_size
        self.border_thickness, self.use_crop, self.crop_offset = border_thickness, use_crop, crop_offset

        self.detector = DeepFakeDetectorGray(resize_ratio)
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

            return False, NO_FACES_DETECTED, cv_image
        elif len(faces) == 1:
            x, y, width, height = faces[0]

            cv_image_cropped = cv_image[x - self.crop_offset:x + width + self.crop_offset,
                                        y - self.crop_offset:y + height + self.crop_offset]

            if self.use_crop:
                cv_image_resized = cv2.resize(cv_image_cropped, self.resize_ratio)
            else:
                cv_image_resized = cv2.resize(cv_image, self.resize_ratio)

            if self.detector.is_image_real(rgb2gray(cv_image_resized)):
                return True, VALID_IMAGE, cv_image_cropped if self.use_crop else cv_image
            else:
                cv2.rectangle(cv_image, (x, y), (x + width, y + height), self.INCORRECT_COLOR, self.border_thickness)
                return False, FAKE_DETECTED, cv_image

        else:
            for (x, y, width, height) in faces:
                cv2.rectangle(cv_image, (x, y), (x + width, y + height), self.INCORRECT_COLOR, self.border_thickness)

            return False, MULTIPLE_FACES_DETECTED, cv_image
