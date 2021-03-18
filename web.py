import numpy as np

from flask import Flask, request, jsonify
from settings import *
from kernel import FaceScanner


app = Flask(__name__)
app.config['SECRET_KEY'] = APPLICATION_SECRET
face_scanner = FaceScanner(RESIZE_RATIO, DETECTOR_KERNEL_FILE_PATH)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def is_extension_correct(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return 'DEEP FAKE DETECTOR IS ALIVE'


@app.route('/api/detection', methods=['POST'])
def fake_photo_detection():
    if 'PROTOCOL_SECRET' not in request.headers or request.headers['PROTOCOL_SECRET'] != PROTOCOL_SECRET:
        return jsonify({"Message": "Protocol secret is missing or incorrect"}), 401
    elif 'photo' not in request.files:
        return jsonify({"Message": "Profile photo is missing in the request body"}), 400
    else:
        file = request.files['photo']
        if is_extension_correct(file.filename):
            is_valid, response = face_scanner.validate_image(file)
            return jsonify({"Message": response, "IsValid": is_valid}), 200
        else:
            return jsonify({"Message": "Allowed file extensions are *.png, *.jpg, *.jpeg"}), 400


if __name__ == '__main__':
    app.debug = not PRODUCTION_MODE
    app.logger.info('Application started in %s mode', "production" if PRODUCTION_MODE else "development")
    app.run()
