import json

from flask import render_template, Blueprint, redirect, url_for, Response
import apis.google_auth

import time
import logging
from threading import Condition
import threading
from flask import Flask, render_template, Response
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)

stream_api = Blueprint('stream_api', __name__, template_folder='templates')

output_condition = Condition()
output_frame = None

def frame_available(frame):
    global output_condition
    global output_frame
    with output_condition:
        output_frame = frame
        output_condition.notify_all()

class StreamingHandler(object):
    def getOutputFrame(self):
        global output_condition
        global output_frame
        while True:
            with output_condition:
                output_condition.wait()
                frame = output_frame
                yield (b'--FRAME\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

PAGE="""\
<html>
<head>
<title>RosieVision</title>
<script type="text/javascript">
function toggleSize(image) {
 if(image.style.width == "100%") {
   image.style.width = "49%"
 } else {
   image.style.width = "100%"
 }
}
</script>
</head>
<body>
<h1>RosieVision MJPEG Stream</h1>
<img src="/stream.mjpg" width="49%" onclick="toggleSize(this)"/>
<img src="/stream2.mjpg" width="49%" onclick="toggleSize(this)"/>
<a href="/restart">Restart Server</a>
</body>
</html>
"""

RESTART="""\
<html>
<head>
<title>RosieVision Restarting</title>
<meta http-equiv="refresh" content="10;/" />
</head>
<body>
<h1>Restarting server...</h1>
This page will automatically redirect
</body>
</html>
"""


streamingHandler = StreamingHandler()

@stream_api.route('/')
@stream_api.route('/index.html')
def index():
    if apis.google_auth.is_logged_in():
        return PAGE, 200

    return redirect(url_for('google_auth.login'))

@stream_api.route('/restart')
def restart():
    if apis.google_auth.is_logged_in():
        thread = threading.Thread(target = __restart)
        thread.start()
        return RESTART, 200

    return redirect(url_for('google_auth.login'))

def __restart():
    time.sleep(1)
    Path("restart.now").touch()

@stream_api.route('/stream.mjpg')
def stream():
    if apis.google_auth.is_logged_in():
        logging.info("Stream request starting.")
        return Response(streamingHandler.getOutputFrame(), mimetype='multipart/x-mixed-replace; boundary=FRAME')

    return redirect(url_for('google_auth.login'))
