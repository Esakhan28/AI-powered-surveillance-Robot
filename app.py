import cv2
import dlib
import numpy as np
import pickle
import os
import time
from flask import Flask, render_template, Response, request, redirect, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import base64
import threading

app = Flask(__name__)

# === Email Setup ===
EMAIL_SENDER = "kesa4716@gmail.com"
EMAIL_PASSWORD = "fbrc nibl kqok kzus"  # App password
EMAIL_RECEIVER = "mohammedesakhan.j2022ai-ml@sece.ac.in"

# === Load dlib Models ===
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# === Load Known Faces ===
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

def save_known_faces():
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

# === Face Utilities ===
def get_face_embedding(image, face_rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = sp(gray, face_rect)
    return np.array(facerec.compute_face_descriptor(image, shape))

def recognize_face(embedding, threshold=0.6):
    best_match = "Unknown"
    best_distance = float("inf")
    for name, embeddings in known_faces.items():
        for known_embedding in embeddings:
            dist = np.linalg.norm(embedding - known_embedding)
            if dist < best_distance:
                best_distance = dist
                best_match = name
    return best_match if best_distance < threshold else "Unknown"

def send_email_alert(face_image):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = "⚠️ Unknown Face Detected"
        
        # Text part
        text = MIMEText("An unknown face was detected. Please visit the app to label it.", "plain")
        msg.attach(text)
        
        # Image part
        _, img_buffer = cv2.imencode('.jpg', face_image)
        img_data = img_buffer.tobytes()
        image = MIMEImage(img_data, name="unknown_face.jpg")
        msg.attach(image)
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Alert email with image sent.")
    except Exception as e:
        print(f"Email error: {e}")

# === Globals ===
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise Exception("Webcam not accessible")

unknown_face = None
current_embedding = None
last_alert_time = 0
alert_cooldown = 60  # seconds
unknown_face_img = None
unknown_face_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_unknown_face")
def check_unknown_face():
    global unknown_face_img
    with unknown_face_lock:
        if unknown_face_img is not None:
            _, buffer = cv2.imencode('.jpg', unknown_face_img)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                "unknown_face_detected": True, 
                "image": img_str,
                "timestamp": time.time()
            })
    return jsonify({"unknown_face_detected": False})

@app.route("/label_unknown", methods=["POST"])
def label_unknown():
    global unknown_face, current_embedding, unknown_face_img
    data = request.get_json()
    name = data.get("name")
    
    with unknown_face_lock:
        if name and current_embedding is not None:
            known_faces.setdefault(name, []).append(current_embedding)
            save_known_faces()
            print(f"Saved new face: {name}")
            unknown_face = None
            current_embedding = None
            unknown_face_img = None
            return jsonify({"success": True})
    return jsonify({"success": False, "error": "No face to label"})

def gen_frames():
    global unknown_face, current_embedding, last_alert_time, unknown_face_img
    while True:
        success, frame = capture.read()
        if not success:
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()

            embedding = get_face_embedding(frame, face)
            name = recognize_face(embedding)

            if name == "Unknown":
                with unknown_face_lock:
                    if unknown_face is None:
                        unknown_face = frame[y1:y2, x1:x2].copy()
                        current_embedding = embedding
                        unknown_face_img = frame[y1:y2, x1:x2].copy()
                        
                        if time.time() - last_alert_time > alert_cooldown:
                            send_email_alert(unknown_face_img)
                            last_alert_time = time.time()

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        ret, buffer = cv2.imencode(".jpg", display_frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/get_known_faces")
def get_known_faces():
    return jsonify({"known_faces": list(known_faces.keys())})

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)