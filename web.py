from flask import Flask
from settings import *


app = Flask(__name__)
app.secret_key = APPLICATION_SECRET


@app.route('/', methods=['GET'])
def index():
    return "DEEP FAKE DETECTOR IS ALIVE"


if __name__ == '__main__':
    app.debug = not PRODUCTION_MODE
    app.run()
