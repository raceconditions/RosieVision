import traceback
import logging
import threading
from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler,SimpleHTTPRequestHandler,HTTPServer
from threading import Condition
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)

output_condition = Condition()
output_frame = None

class WebServer(object):
    def __init__(self, port):
        self.port = port

    def start(self):
        address = ('', self.port)
        server = ThreadedHTTPServer(address, StreamingHandler)
        thread = threading.Thread(target = server.serve_forever)
        thread.daemon = True
        thread.start()

    def frame_available(self, frame):
        global output_condition
        global output_frame
        with output_condition:
            logging.info("got frame")
            output_frame = frame
            output_condition.notify_all()

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
<img src="http://rv-cortex:8081/stream.mjpg" width="49%" onclick="toggleSize(this)"/>
<img src="http://rv-cortex:8082/stream.mjpg" width="49%" onclick="toggleSize(this)"/>
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

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/restart':
            self.send_response(201)
            self.end_headers()
            self.wfile.write(RESTART.encode('utf-8'))
            Path("restart.now").touch()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            logging.info("Stream request starting.")
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = self.getOutputFrame()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                traceback.print_exc()
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

    def getOutputFrame(self):
        global output_condition
        global output_frame
        with output_condition:
            output_condition.wait()
            frame = output_frame
            return frame

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

