import json
import logging
import tensorflow as tf
import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('HTTP health-check requested.')

    try:
        tf_engine = tf.__version__
    except Exception as ex:
        tf_engine = "unavailable"
        logging.error(ex)

    return func.HttpResponse(
        json.dumps({ "application": "available", "computational_engine": tf_engine }),
        status_code=200
    )
