import sys
import time
import os
import subprocess
import signal
import logging

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("apscheduler").setLevel(logging.ERROR)

server_a_cmd = ['python3', 'server.py', 'http://io:8081/stream.mjpg', '--web-port', '8081', '--use-tpu'] # > /media/images/log_a.txt 2>&1' &
server_b_cmd = ['python3', 'server.py', 'http://rv-1:8080/stream/video.mjpeg', '--web-port', '8082'] # > /media/images/log_b.txt 2>&1' &

server_a = subprocess.Popen(server_a_cmd)
server_b = subprocess.Popen(server_b_cmd)

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
  
    def exit_gracefully(self, signum, frame):
        logging.info("Received signal " + signal.Signals(signum).name + " from container, cleaning up...")
        os.kill(server_a.pid, signal.SIGTERM)
        os.kill(server_b.pid, signal.SIGTERM)
        os.kill(server_a.pid, signal.SIGTERM)
        os.kill(server_b.pid, signal.SIGTERM)
        scheduler.shutdown()

class FileWatcher:
    def __init__(self, src_path):
        self.__src_path = src_path
        self.__event_handler = FilesEventHandler()
        self.__event_observer = Observer()

    def run(self):
        self.start()
        logging.info("Starting FileWatcher with source path %s", src_path)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def start(self):
        self.__schedule()
        self.__event_observer.start()

    def stop(self):
        self.__event_observer.stop()
        self.__event_observer.join()

    def __schedule(self):
        self.__event_observer.schedule(
            self.__event_handler,
            self.__src_path,
            recursive=True
        )

class FilesEventHandler(PatternMatchingEventHandler):
    FILE_PATTERN = ["*.now"]

    def __init__(self):
        super().__init__(patterns=self.FILE_PATTERN)

    def on_any_event(self, event):
        self.process(event)

    def process(self, event):
        filename, ext = os.path.splitext(os.path.basename(event.src_path))
        logging.info("Received %s event", filename)
        if filename == 'restart':
            restart_process()

def restart_process():
    global server_a, server_b, server_a_cmd, server_b_cmd
    os.kill(server_a.pid, signal.SIGTERM)
    os.kill(server_b.pid, signal.SIGTERM)
    time.sleep(1)
    os.kill(server_a.pid, signal.SIGTERM)
    os.kill(server_b.pid, signal.SIGTERM)
    time.sleep(2)
    server_a = subprocess.Popen(server_a_cmd)
    server_b = subprocess.Popen(server_b_cmd)

def check_process():
    global server_a, server_b, server_a_cmd, server_b_cmd
    server_a_result = server_a.poll()
    if server_a_result is not None:
        logging.info("server a process exited with error code ", server_a_result)
        server_a.wait()
        server_a = subprocess.Popen(server_a_cmd)
  
    server_b_result = server_b.poll()
    if server_b_result is not None:
        logging.info("server b process exited with error code ", server_b_result)
        server_b.wait()
        server_b = subprocess.Popen(server_b_cmd)

if __name__ == "__main__":
    src_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    scheduler.add_job(check_process, 'interval', seconds=15)
    scheduler.start()
    killer = GracefulKiller()
    FileWatcher(src_path).run()
    logging.info("Graceful shutdown complete, exiting.")

