import io
import logging
from threading import Condition
import threading
import traceback
import datetime as dt
from flask import Flask, render_template, Response
from pathlib import Path

from authlib.client import OAuth2Session
import google.oauth2.credentials
import googleapiclient.discovery

from apis.stream import stream_api, frame_available
from apis.google_auth import google_api

logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__)
app.register_blueprint(stream_api)
app.register_blueprint(google_api)
app.secret_key = 'aspdoihasdfoihasdfasdf345adf'

output_condition = Condition()
output_frame = None

class WebServer(object):
    def __init__(self, port):
        self.port = port

    def start(self):
        address = ('', self.port)

        thread = threading.Thread(target = self.__start_flask)
        thread.daemon = True
        thread.start()

    def __start_flask(self):
        app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)

    def frame_available(self, frame):
        frame_available(frame)
