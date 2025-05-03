import cv2
import os
import dlib
import numpy as np
import pickle

# Load dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load or initialize known faces
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)  # {name: [embedding1, embedding2, ...]}
else:
    known_faces = {}

def save_known_faces():
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

def get_face_embedding(image, face_rect=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = [face_rect] if face_rect else detector(gray)
    if not faces:
        return None
    for face in faces:
        landmarks = sp(gray, face)
        embedding = facerec.compute_face_descriptor(image, landmarks)
        return np.array(embedding)
    return None

def recognize_face(embedding, threshold=0.6):
    if not known_faces:
        return "Unknown"
    
    best_match = "Unknown"
    best_distance = float("inf")

    for name, embeddings in known_faces.items():
        for known_embedding in embeddings:
            dist = np.linalg.norm(embedding - known_embedding)
            if dist < best_distance:
                best_distance = dist
                best_match = name

    return best_match if best_distance < threshold else "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    print("🎥 Starting face recognition... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, y1 = max(0, face.left()), max(0, face.top())
            x2, y2 = min(frame.shape[1], face.right()), min(frame.shape[0], face.bottom())
            face_img = frame[y1:y2, x1:x2]

            embedding = get_face_embedding(frame, face)
            if embedding is None:
                continue

            name = recognize_face(embedding)

            if name == "Unknown":
                cv2.imshow("Unknown Face - Press any key", face_img)
                print("👤 Unknown face detected. Please enter name:")
                input_name = input("Enter name: ").strip()
                cv2.destroyWindow("Unknown Face - Press any key")

                if input_name:
                    # If the name is not in the known_faces, initialize it as a list
                    if input_name not in known_faces:
                        known_faces[input_name] = []

                    # Ensure the embeddings are stored in a list (even if a single embedding exists)
                    if isinstance(known_faces[input_name], np.ndarray):
                        known_faces[input_name] = [known_faces[input_name]]
                    
                    # Append the new embedding to the list of that person
                    known_faces[input_name].append(embedding)
                    save_known_faces()
                    print(f"✅ Saved new face for: {input_name}")
                    name = input_name

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
