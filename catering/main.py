import cv2
import face_recognition
import numpy as np
import os
import sqlite3
import pickle
from datetime import datetime

# --- CONFIGURATION ---
DB_FILE = "hall_visitors.db"
IMAGE_DIR = "visitors_images"
TOLERANCE = 0.5

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encoding BLOB NOT NULL,
            image_path TEXT,
            first_seen TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def save_visitor_to_db(encoding, image_path):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    binary_encoding = pickle.dumps(encoding)
    cursor.execute("INSERT INTO visitors (encoding, image_path, first_seen) VALUES (?, ?, ?)",
                   (binary_encoding, image_path, datetime.now()))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def load_known_faces():
    known_encodings = []
    known_ids = []
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, encoding FROM visitors")
    rows = cursor.fetchall()
    for row in rows:
        try:
            known_encodings.append(pickle.loads(row[1]))
            known_ids.append(row[0])
        except:
            continue
    conn.close()
    return known_encodings, known_ids


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    init_db()
    known_face_encodings, known_face_ids = load_known_faces()
    unique_visitor_count = len(known_face_encodings)
    print(f"System Loaded. {unique_visitor_count} visitors known.")

    # --- ARDUCAM SETUP ---
    print("Searching for Arducam (Index 1)...")

    # We try Index 1 first (Standard for external USB cameras)
    video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # If Index 1 fails, fall back to laptop camera (Index 0)
    if not video_capture.isOpened():
        print("Warning: Arducam (Index 1) not found. Trying Index 2...")
        video_capture = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    if not video_capture.isOpened():
        print("Error: Could not find Arducam. Reverting to Laptop Webcam (Index 0).")
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not video_capture.isOpened():
        print("CRITICAL ERROR: No cameras found.")
        exit()

    print("Camera active! Starting detection...")

    while True:
        # 1. READ FRAME
        ret, frame = video_capture.read()
        if not ret or frame is None:
            continue

        # 2. PREPARE IMAGE
        # Resize to 1/4 size for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR (OpenCV format) to RGB (Face Recognition format)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Force 8-bit format (Fixes "Unsupported Image" error)
        rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)

        face_locations = []
        face_encodings = []

        # 3. DETECT FACES
        try:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        except Exception as e:
            # Arducams can sometimes drop a frame, just ignore it
            pass

        # 4. DRAW RESULTS
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Unknown"
            is_new_person = True

            # Identify
            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < TOLERANCE:
                    name = f"ID: {known_face_ids[best_match_index]}"
                    is_new_person = False

            # Register New Person
            if is_new_person:
                top, right, bottom, left = face_location
                top *= 4;
                right *= 4;
                bottom *= 4;
                left *= 4

                # Safety checks
                h, w, _ = frame.shape
                top = max(0, top);
                left = max(0, left)
                bottom = min(h, bottom);
                right = min(w, right)

                face_image = frame[top:bottom, left:right]

                if face_image.size > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_filename = f"{IMAGE_DIR}/temp_{timestamp}.jpg"
                    cv2.imwrite(temp_filename, face_image)

                    new_id = save_visitor_to_db(face_encoding, temp_filename)

                    final_filename = f"{IMAGE_DIR}/visitor_{new_id}_{timestamp}.jpg"
                    if os.path.exists(temp_filename):
                        os.rename(temp_filename, final_filename)

                    conn = sqlite3.connect(DB_FILE)
                    conn.execute("UPDATE visitors SET image_path = ? WHERE id = ?", (final_filename, new_id))
                    conn.commit()
                    conn.close()

                    known_face_encodings.append(face_encoding)
                    known_face_ids.append(new_id)
                    unique_visitor_count += 1
                    name = f"NEW: {new_id}"
                    print(f"New Visitor Registered: ID {new_id}")

            # Draw Box
            top, right, bottom, left = face_location
            top *= 4;
            right *= 4;
            bottom *= 4;
            left *= 4
            color = (0, 255, 0) if not is_new_person else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # 5. SHOW VIDEO
        cv2.putText(frame, f"Total Visitors: {unique_visitor_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        cv2.imshow('Hall Monitor (Arducam)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()