import os


PRODUCTION_MODE = os.getenv('PRODUCTION_MODE') == 'True'

APPLICATION_SECRET = os.getenv('APPLICATION_SECRET')
APPLICATION_URL = os.getenv('APPLICATION_URL')

PROTOCOL_SECRET = os.getenv('PROTOCOL_SECRET')
