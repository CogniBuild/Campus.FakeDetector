import json
import logging
import azure.functions as func

from .detectors import face_scanner


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def is_extension_valid(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def main(req: func.HttpRequest) -> func.HttpResponse:
    image_file = req.files.get("file")

    logging.info(f"HTTP deep fake detector requested with file: {image_file}.")

    if image_file is None or image_file.filename == "":
        logging.info("Request finished with 'Profile photo is missing in the request body' error.")

        return func.HttpResponse(
            json.dumps({
                "success": 0,
                "message": "Profile photo is missing in the request body",
                "checks": {
                    "missing-photo": 1,
                    "wrong-extension": 0,
                    "no-faces": 0,
                    "multiple-faces": 0,
                    "deep-fake": 0
                }
            }),
            status_code=400
        )

    if not is_extension_valid(image_file.filename):
        logging.info("Request finished with 'Allowed file extensions are *.png, *.jpg, *.jpeg' error.")

        return func.HttpResponse(
            json.dumps({
                "success": 0,
                "message": "Allowed file extensions are *.png, *.jpg, *.jpeg",
                "checks": {
                    "missing-photo": 0,
                    "wrong-extension": 1,
                    "no-faces": 0,
                    "multiple-faces": 0,
                    "deep-fake": 0
                }
            }),
            status_code=400
        )

    response, status_codes = face_scanner.validate_image(image_file)

    if 1 in status_codes.values():
        return func.HttpResponse(
            json.dumps({
                "success": 0,
                "message": response,
                "checks": {
                    "missing-photo": 0,
                    "wrong-extension": 0,
                    **status_codes
                }
            }),
            status_code=400
        )
    else:
        return func.HttpResponse(
            json.dumps({
                "success": 1,
                "message": response,
                "checks": {
                    "missing-photo": 0,
                    "wrong-extension": 0,
                    **status_codes
                }
            }),
            status_code=200
        )
