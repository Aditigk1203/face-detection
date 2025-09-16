import cv2
import face_recognition
import os
import numpy as np

# Path to dataset folder
DATASET_DIR = "face_database"

known_encodings = []
known_names = []

print("Loading known faces...")

# Loop through each person in dataset
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    # Loop through images of that person
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        # Load and encode image
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            print(f"⚠️ No face found in {img_path}, skipping...")
            continue

        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_encodings.append(face_encoding)
        known_names.append(person_name)

print(f"✅ Loaded {len(known_names)} face encodings.")

# Start webcam
video_capture = cv2.VideoCapture(0)

print("Starting live camera. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Match each detected face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Scale back face locations to full frame size
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box + name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Show video
    cv2.imshow('Live Face Recognition', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
