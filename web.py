import numpy as np

from flask import Flask, request, make_response
from settings import *
from kernel.detectors import FaceScanner


app = Flask(__name__)
app.config['SECRET_KEY'] = APPLICATION_SECRET
face_scanner = FaceScanner(RESIZE_RATIO,
                           SCANNER_KERNEL_FILE_PATH,
                           DETECTOR_KERNEL_FILE_PATH)


@app.route('/', methods=['GET'])
def index():
    return 'DEEP FAKE DETECTOR IS ALIVE'


@app.route('/api/detection', methods=['POST'])
def fake_photo_detection():
    if 'PROTOCOL_SECRET' not in request.headers or request.headers['PROTOCOL_SECRET'] != PROTOCOL_SECRET:
        return make_response("Protocol secret is missing or incorrect"), 401
    elif 'photo' not in request.files:
        return make_response("Profile photo is missing in the request body"), 400
    else:
        response, _ = face_scanner.validate_image(np.frombuffer(request.files['photo'].read(), np.uint8))
        return response, 200


if __name__ == '__main__':
    app.debug = not PRODUCTION_MODE
    app.logger.info('Application started in %s mode', "production" if PRODUCTION_MODE else "development")
    app.run()
