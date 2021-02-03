import os


# Web application settings
PRODUCTION_MODE = os.getenv('ENV_PRODUCTION_MODE') == 'True'
APPLICATION_SECRET = os.getenv('ENV_APPLICATION_SECRET')
APPLICATION_URL = os.getenv('ENV_APPLICATION_URL')
PROTOCOL_SECRET = os.getenv('ENV_PROTOCOL_SECRET')

# Deep learning classifier settings
RESIZE_RATIO = int(os.getenv('ENV_RESIZE_RATIO'))
KERNEL_FILE_PATH = os.getenv('ENV_KERNEL_FILE_PATH')
