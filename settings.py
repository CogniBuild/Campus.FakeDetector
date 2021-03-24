import os


# Web application settings
PRODUCTION_MODE = os.getenv('ENV_PRODUCTION_MODE') == 'True'
APPLICATION_SECRET = os.getenv('ENV_APPLICATION_SECRET')
PROTOCOL_SECRET = os.getenv('ENV_PROTOCOL_SECRET')

# Deep learning classifier settings
RESIZE_RATIO = (128, 128, 3)
DETECTOR_KERNEL_FILE_PATH = os.getenv('ENV_DETECTOR_KERNEL_FILE_PATH')
