cam.py:

from flask import Flask, Response
import subprocess
import cv2
import numpy as np

app = Flask(_name_)

@app.route('/')
def index():
    return '''
        <html>
            <head>
                <title>Raspberry Pi Camera Stream</title>
            </head>
            <body>
                <h1>Raspberry Pi Camera Stream</h1>
                <img src="/video" width="640" height="480">
            </body>
        </html>
    '''

@app.route('/video')
def video():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video():
    cam_cmd = [
        "libcamera-vid", "-t", "0", "--width", "640", "--height", "480",
        "--codec", "yuv420", "--nopreview", "--framerate", "30", "-o", "-"
    ]
    
    ffmpeg_cmd = [
        "ffmpeg", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "640x480",
        "-framerate", "30", "-i", "-", "-f", "image2pipe", "-vcodec", "mjpeg", "-"
    ]
    
    with subprocess.Popen(cam_cmd, stdout=subprocess.PIPE) as libcam:
        with subprocess.Popen(ffmpeg_cmd, stdin=libcam.stdout, stdout=subprocess.PIPE) as ffmpeg:
            libcam.stdout.close()
            while True:
                # Read one complete JPEG frame
                jpg = b''
                while True:
                    byte = ffmpeg.stdout.read(1)
                    if not byte:
                        return
                    jpg += byte
                    if jpg[-2:] == b'\xff\xd9':
                        break

                # Yield MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)