import os

RESIZE_RATIO = (128, 128, 3)
DETECTOR_KERNEL_FILE_PATH = os.getenv('ENV_DETECTOR_KERNEL_FILE_PATH')
PROBABILITY_THRESHOLD = os.getenv('ENV_PROBABILITY_THRESHOLD')