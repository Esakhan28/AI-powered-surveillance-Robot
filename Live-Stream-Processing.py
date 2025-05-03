from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import numpy as np
import requests
from ultralytics import YOLO
import pandas as pd
from threading import Lock, Thread
import os
import time
from collections import deque
import dlib
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import base64

app = Flask(__name__)
lock = Lock()
face_lock = Lock()

# Configuration
PI_IP = '192.168.137.19'
STREAM_URL = f'http://{PI_IP}:5000/video'
TARGET_FPS = 30
MIN_CONFIDENCE = 0.75
ACTIVITY_THRESHOLD = 0.5
ACTIVITY_HISTORY = 10

# Email Configuration
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # Use app password for Gmail
EMAIL_RECEIVER = "recipient_email@example.com"
ALERT_COOLDOWN = 60  # seconds between alerts

# Face Recognition
FACE_RECOGNITION_THRESHOLD = 0.6
KNOWN_FACES_PATH = 'known_faces.pkl'

print("Loading YOLO model...")
activity_model = YOLO("yolo11s-pose.pt")
print("YOLO model loaded successfully")

print("Loading face recognition models...")
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
print("Face recognition models loaded successfully")

# Load known faces
try:
    with open(KNOWN_FACES_PATH, "rb") as f:
        known_faces = pickle.load(f)
except (FileNotFoundError, EOFError):
    known_faces = {}

output_dirs = {
    'full_frames': 'saved_images/full_frames',
    'cropped_persons': 'saved_images/cropped_persons',
    'data': 'saved_data',
    'known_faces': 'known_faces_data'
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Global variables
current_frame = None
processed_frame = None
frame_counter = 0
person_counter = 0
all_data = []
processing_active = True
last_save_time = time.time()
frame_times = deque(maxlen=30)
activity_history = []
current_mode = "activity"  # "activity" or "face_recognition"
unknown_face_img = None
current_embedding = None
last_alert_time = 0
prompt_for_name = False

activity_params = {
    'walking': {'leg_angle_range': (15, 60), 'arm_swing_threshold': 0.1},
    'running': {'leg_angle_range': (45, 90), 'arm_swing_threshold': 0.2},
    'sitting': {'torso_angle_range': (80, 110)},
    'standing': {'keypoints_variance_threshold': 0.01}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Surveillance System</title>
    <style>
        :root {
            --primary-color: #4a6fa5;
            --secondary-color: #166088;
            --accent-color: #4fc3f7;
            --background-color: #f5f7fa;
            --card-color: #ffffff;
            --text-color: #333333;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --danger-color: #f44336;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 20px 0;
            text-align: center;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        h1 {
            margin: 0;
            font-size: 2.2em;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background-color: var(--card-color);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .video-card {
            grid-column: 1 / -1;
            text-align: center;
        }
        
        .video-container {
            position: relative;
            margin: 0 auto;
            width: 640px;
            height: 480px;
            border: 2px solid var(--secondary-color);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            background-color: var(--card-color);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--secondary-color);
            margin: 5px 0;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
        }
        
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: var(--primary-color);
            color: white;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        button:hover {
            background-color: var(--secondary-color);
            transform: translateY(-2px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .btn-success {
            background-color: var(--success-color);
        }
        
        .btn-warning {
            background-color: var(--warning-color);
        }
        
        .btn-danger {
            background-color: var(--danger-color);
        }
        
        .activity-log {
            max-height: 200px;
            overflow-y: auto;
            padding: 10px;
            background-color: var(--card-color);
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .activity-item {
            padding: 8px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }
        
        .face-recognition-ui {
            display: none;
            margin-top: 20px;
            text-align: center;
        }
        
        .unknown-face-container {
            margin: 20px 0;
        }
        
        .unknown-face-img {
            max-width: 200px;
            border: 2px solid var(--warning-color);
            border-radius: 5px;
        }
        
        #nameInput {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
        }
        
        .mode-indicator {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            background-color: var(--accent-color);
            color: white;
            font-size: 0.9em;
            margin-left: 10px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .video-container {
                width: 100%;
                height: auto;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Smart Surveillance System</h1>
        </div>
    </header>
    
    <div class="container">
        <div class="dashboard">
            <div class="card video-card">
                <h2>Live Feed <span class="mode-indicator" id="modeIndicator">Activity Detection</span></h2>
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" alt="Live Video Feed">
                </div>
                <div class="controls">
                    <button id="toggleProcessing">{{ 'Pause Processing' if processing_active else 'Resume Processing' }}</button>
                    <button id="switchMode">Switch to Face Recognition</button>
                </div>
            </div>
            
            <div class="card">
                <h2>System Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Status</div>
                        <div class="stat-value" id="statusValue">{{ 'Active' if processing_active else 'Paused' }}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">FPS</div>
                        <div class="stat-value" id="fpsValue">{{ '%.1f'|format(current_fps) }}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Frames Processed</div>
                        <div class="stat-value" id="frameValue">{{ frame_counter }}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Persons Detected</div>
                        <div class="stat-value" id="personValue">{{ person_counter }}</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Current Activities</h2>
                <div class="activity-log" id="activityLog">
                    {% for activity in current_activities %}
                    <div class="activity-item">
                        <span>{{ activity }}</span>
                        <span>{{ time.strftime('%H:%M:%S') }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="card face-recognition-ui" id="faceRecognitionUI">
                <h2>Face Recognition</h2>
                <div id="unknownFaceAlert" class="unknown-face-container" style="display: none;">
                    <h3>Unknown Face Detected!</h3>
                    <img id="unknownFaceImg" class="unknown-face-img" src="" alt="Unknown Face">
                    <div>
                        <input type="text" id="nameInput" placeholder="Enter name">
                        <button id="saveFaceBtn" class="btn-success">Save Face</button>
                    </div>
                </div>
                <div id="knownFacesList">
                    <h3>Known Faces</h3>
                    <div id="facesContainer"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Toggle processing
        document.getElementById('toggleProcessing').addEventListener('click', function() {
            fetch(this.textContent === 'Pause Processing' ? '/process_off' : '/process_on')
                .then(response => response.text())
                .then(data => {
                    this.textContent = this.textContent === 'Pause Processing' ? 'Resume Processing' : 'Pause Processing';
                    document.getElementById('statusValue').textContent = 
                        this.textContent === 'Pause Processing' ? 'Active' : 'Paused';
                });
        });
        
        // Switch between modes
        document.getElementById('switchMode').addEventListener('click', function() {
            fetch('/switch_mode')
                .then(response => response.json())
                .then(data => {
                    if (data.mode === 'face_recognition') {
                        this.textContent = 'Switch to Activity Detection';
                        document.getElementById('modeIndicator').textContent = 'Face Recognition';
                        document.getElementById('faceRecognitionUI').style.display = 'block';
                        loadKnownFaces();
                        checkForNamePrompt();
                    } else {
                        this.textContent = 'Switch to Face Recognition';
                        document.getElementById('modeIndicator').textContent = 'Activity Detection';
                        document.getElementById('faceRecognitionUI').style.display = 'none';
                    }
                });
        });
        
        // Check for name prompt
        function checkForNamePrompt() {
            fetch('/check_prompt_name')
                .then(response => response.json())
                .then(data => {
                    if (data.prompt) {
                        document.getElementById('unknownFaceAlert').style.display = 'block';
                        document.getElementById('unknownFaceImg').src = 'data:image/jpeg;base64,' + data.image;
                    }
                });
            
            setTimeout(checkForNamePrompt, 3000);
        }
        
        // Save face button
        document.getElementById('saveFaceBtn').addEventListener('click', function() {
            const name = document.getElementById('nameInput').value.trim();
            if (name) {
                fetch('/label_unknown', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({name: name})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Face saved successfully!');
                        document.getElementById('unknownFaceAlert').style.display = 'none';
                        document.getElementById('nameInput').value = '';
                        loadKnownFaces();
                    } else {
                        alert('Error saving face: ' + data.error);
                    }
                });
            } else {
                alert('Please enter a name');
            }
        });
        
        // Load known faces
        function loadKnownFaces() {
            fetch('/get_known_faces')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('facesContainer');
                    container.innerHTML = '';
                    
                    if (data.known_faces.length > 0) {
                        data.known_faces.forEach(face => {
                            const faceDiv = document.createElement('div');
                            faceDiv.className = 'activity-item';
                            faceDiv.textContent = face;
                            container.appendChild(faceDiv);
                        });
                    } else {
                        container.innerHTML = '<p>No known faces yet</p>';
                    }
                });
        }
        
        // Update stats periodically
        function updateStats() {
            fetch('/get_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fpsValue').textContent = data.current_fps.toFixed(1);
                    document.getElementById('frameValue').textContent = data.frame_count;
                    document.getElementById('personValue').textContent = data.person_count;
                    
                    const activityLog = document.getElementById('activityLog');
                    activityLog.innerHTML = '';
                    data.current_activities.forEach(activity => {
                        const item = document.createElement('div');
                        item.className = 'activity-item';
                        item.innerHTML = `<span>${activity}</span><span>${new Date().toLocaleTimeString()}</span>`;
                        activityLog.appendChild(item);
                    });
                });
            
            setTimeout(updateStats, 2000);
        }
        
        updateStats();
    </script>
</body>
</html>
"""

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def detect_activity(keypoints):
    activities = []
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14

    try:
        left_leg_angle = calculate_angle(
            keypoints[LEFT_HIP],
            keypoints[LEFT_KNEE],
            [keypoints[LEFT_KNEE][0], keypoints[LEFT_KNEE][1] - 0.1]
        )
        right_leg_angle = calculate_angle(
            keypoints[RIGHT_HIP],
            keypoints[RIGHT_KNEE],
            [keypoints[RIGHT_KNEE][0], keypoints[RIGHT_KNEE][1] - 0.1]
        )
        torso_angle = calculate_angle(
            keypoints[LEFT_SHOULDER],
            keypoints[LEFT_HIP],
            keypoints[RIGHT_HIP]
        )

        if (activity_params['walking']['leg_angle_range'][0] < left_leg_angle < activity_params['walking']['leg_angle_range'][1] or
            activity_params['walking']['leg_angle_range'][0] < right_leg_angle < activity_params['walking']['leg_angle_range'][1]):
            activities.append("Walking")
        if (activity_params['running']['leg_angle_range'][0] < left_leg_angle < activity_params['running']['leg_angle_range'][1] or
            activity_params['running']['leg_angle_range'][0] < right_leg_angle < activity_params['running']['leg_angle_range'][1]):
            activities.append("Running")
        if (activity_params['sitting']['torso_angle_range'][0] < torso_angle < activity_params['sitting']['torso_angle_range'][1]):
            activities.append("Sitting")
        if len(activities) == 0:
            activities.append("Standing")
    except Exception as e:
        print(f"Activity detection error: {e}")
        activities.append("Unknown")

    return activities

def get_face_embedding(image, face_rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = shape_predictor(gray, face_rect)
    return np.array(face_recognizer.compute_face_descriptor(image, shape))

def recognize_face(embedding):
    best_match = "Unknown"
    best_distance = float("inf")
    for name, embeddings in known_faces.items():
        for known_embedding in embeddings:
            dist = np.linalg.norm(embedding - known_embedding)
            if dist < best_distance:
                best_distance = dist
                best_match = name
    return best_match if best_distance < FACE_RECOGNITION_THRESHOLD else "Unknown"

def send_email_alert(face_image):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = "⚠️ Unknown Face Detected"
        
        text = MIMEText("An unknown face was detected. Please visit the app to label it.", "plain")
        msg.attach(text)
        
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

def save_known_faces():
    with open(KNOWN_FACES_PATH, "wb") as f:
        pickle.dump(known_faces, f)

def process_stream():
    global current_frame, processed_frame, frame_counter, person_counter, all_data
    global last_save_time, frame_times, activity_history, current_mode
    global unknown_face_img, current_embedding, last_alert_time, prompt_for_name

    session = requests.Session()
    stream = session.get(STREAM_URL, stream=True)
    bytes_data = bytes()

    try:
        last_frame_time = time.time()
        frame_interval = 1.0 / TARGET_FPS
        last_annotated = None
        current_activities = []
        last_detection_results = None
        frame_skip_counter = 0
        SKIP_FRAMES = 30  # Process every 30 frames (once per second at 30 FPS)

        while True:
            start_time = time.time()
            bytes_data += stream.raw.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                with lock:
                    current_frame = frame.copy()
                    frame_counter += 1
                    frame_skip_counter += 1

                    if processing_active:
                        current_time = time.time()
                        elapsed = current_time - last_frame_time

                        if elapsed >= frame_interval:
                            last_frame_time = current_time
                            process_start = time.time()

                            if current_mode == "activity":
                                if frame_skip_counter >= SKIP_FRAMES or last_detection_results is None:
                                    frame_skip_counter = 0
                                    results = activity_model(frame, verbose=False)
                                    last_detection_results = results
                                else:
                                    results = last_detection_results
                                
                                annotated_frame = results[0].plot()
                                current_activities = []

                                for r in results:
                                    bound_box = r.boxes.xyxy
                                    conf = r.boxes.conf.tolist()
                                    keypoints = r.keypoints.xyn.tolist()

                                    for index, box in enumerate(bound_box):
                                        if conf[index] > MIN_CONFIDENCE:
                                            x1, y1, x2, y2 = box.tolist()
                                            cropped_person = frame[int(y1):int(y2), int(x1):int(x2)]
                                            activities = detect_activity(keypoints[index])
                                            current_activities.extend(activities)
                                            person_path = os.path.join(output_dirs['cropped_persons'], f'person_{person_counter}.jpg')
                                            cv2.imwrite(person_path, cropped_person)
                                            data = {
                                                'frame_num': frame_counter,
                                                'image_name': f'person_{person_counter}.jpg',
                                                'timestamp': current_time,
                                                'activities': ', '.join(activities)
                                            }
                                            for j in range(len(keypoints[index])):
                                                data[f'x{j}'] = keypoints[index][j][0]
                                                data[f'y{j}'] = keypoints[index][j][1]
                                            all_data.append(data)
                                            person_counter += 1

                                activity_history.append(current_activities)
                                if len(activity_history) > ACTIVITY_HISTORY:
                                    activity_history.pop(0)

                                if current_activities:
                                    activity_text = f"Activities: {', '.join(set(current_activities))}"
                                    cv2.putText(annotated_frame, activity_text, (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                                last_annotated = annotated_frame

                            elif current_mode == "face_recognition":
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                faces = face_detector(gray)
                                annotated_frame = frame.copy()
                                
                                for face in faces:
                                    x1, y1 = face.left(), face.top()
                                    x2, y2 = face.right(), face.bottom()

                                    embedding = get_face_embedding(frame, face)
                                    name = recognize_face(embedding)

                                    if name == "Unknown":
                                        with face_lock:
                                            if unknown_face_img is None:
                                                unknown_face_img = frame[y1:y2, x1:x2].copy()
                                                current_embedding = embedding
                                                prompt_for_name = True
                                                
                                                if time.time() - last_alert_time > ALERT_COOLDOWN:
                                                    send_email_alert(unknown_face_img)
                                                    last_alert_time = time.time()
                                    
                                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(annotated_frame, name, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                                last_annotated = annotated_frame

                            if last_annotated is not None:
                                processed_frame = last_annotated.copy()

                            processing_time = (time.time() - process_start) * 1000
                            frame_times.append(processing_time)

                            if time.time() - last_save_time >= 60:
                                timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                                frame_path = os.path.join(output_dirs['full_frames'], f'frame_{timestamp_str}.jpg')
                                cv2.imwrite(frame_path, frame)
                                save_data()
                                last_save_time = time.time()

                    processing_duration = time.time() - start_time
                    sleep_time = max(0, frame_interval - processing_duration)
                    time.sleep(sleep_time)

    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        save_data()
        stream.close()
        session.close()

def save_data():
    global all_data
    if not all_data:
        return
    csv_path = os.path.join(output_dirs['data'], 'activity_data.csv')
    df = pd.DataFrame(all_data)
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
    all_data = []
    print(f"Saved {len(df)} data points to {csv_path}")

def generate_frames():
    global current_frame, processed_frame
    while True:
        with lock:
            if processed_frame is not None:
                frame = processed_frame
            elif current_frame is not None:
                frame = current_frame
            else:
                continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    current_fps = 0
    avg_proc_time = 0
    if len(frame_times) > 0:
        current_fps = min(TARGET_FPS, 1000 / (sum(frame_times) / len(frame_times))) if frame_times else 0
        avg_proc_time = sum(frame_times) / len(frame_times)
    
    current_activities = []
    if activity_history:
        flat_activities = [item for sublist in activity_history for item in sublist]
        if flat_activities:
            from collections import Counter
            common_activities = Counter(flat_activities).most_common(3)
            current_activities = [f"{act} ({count}x)" for act, count in common_activities]
    
    return render_template_string(HTML_TEMPLATE,
                                processing_active=processing_active,
                                current_fps=current_fps,
                                frame_count=frame_counter,
                                person_count=person_counter,
                                current_activities=current_activities)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_on')
def process_on():
    global processing_active
    processing_active = True
    return "Processing started", 200

@app.route('/process_off')
def process_off():
    global processing_active
    processing_active = False
    return "Processing paused", 200

@app.route('/switch_mode')
def switch_mode():
    global current_mode
    current_mode = "face_recognition" if current_mode == "activity" else "activity"
    return jsonify({"mode": current_mode})

@app.route("/check_prompt_name")
def check_prompt_name():
    global prompt_for_name, unknown_face_img
    with face_lock:
        if prompt_for_name and unknown_face_img is not None:
            _, buffer = cv2.imencode('.jpg', unknown_face_img)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                "prompt": True, 
                "image": img_str
            })
    return jsonify({"prompt": False})

@app.route("/label_unknown", methods=["POST"])
def label_unknown():
    global unknown_face_img, current_embedding, prompt_for_name, known_faces
    data = request.get_json()
    name = data.get("name")
    
    with face_lock:
        if name and current_embedding is not None:
            known_faces.setdefault(name, []).append(current_embedding)
            save_known_faces()
            unknown_face_img = None
            current_embedding = None
            prompt_for_name = False
            return jsonify({"success": True})
    return jsonify({"success": False, "error": "No face to label"})

@app.route("/get_known_faces")
def get_known_faces():
    return jsonify({"known_faces": list(known_faces.keys())})

@app.route("/get_stats")
def get_stats():
    current_fps = 0
    if len(frame_times) > 0:
        current_fps = min(TARGET_FPS, 1000 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    
    current_activities = []
    if activity_history:
        flat_activities = [item for sublist in activity_history for item in sublist]
        if flat_activities:
            from collections import Counter
            common_activities = Counter(flat_activities).most_common(3)
            current_activities = [f"{act}" for act, count in common_activities]
    
    return jsonify({
        "current_fps": current_fps,
        "frame_count": frame_counter,
        "person_count": person_counter,
        "current_activities": current_activities
    })

if __name__ == '__main__':
    Thread(target=process_stream, daemon=True).start()
    app.run(host='0.0.0.0', port=8000, threaded=True)