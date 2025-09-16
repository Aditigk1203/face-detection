import cv2
import face_recognition
import pickle
import time

ENCODINGS_FILE = "encodings.pickle"

print("[INFO] Loading encodings...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("[INFO] Starting video stream...")
video_capture = cv2.VideoCapture(0)

process_this_frame = True
fps_start = time.time()
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Detect faces & compute encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for i, b in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    name = known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # FPS counter
    frame_count += 1
    if frame_count >= 10:
        fps = frame_count / (time.time() - fps_start)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        frame_count = 0
        fps_start = time.time()

    cv2.imshow("Face Recognition", frame)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
