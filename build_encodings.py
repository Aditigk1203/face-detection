import os
import cv2
import face_recognition
import pickle

DATASET_DIR = "face_database"
ENCODINGS_FILE = "encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] Starting to process dataset...")

# Loop over each person’s folder
for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"[INFO] Processing {person_name}...")
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not read {image_path}, skipping...")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")  # use "cnn" if you want slower but more accurate
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"[⚠️] No face found in {image_path}, skipping...")

# Save encodings
print(f"[INFO] Saving encodings to {ENCODINGS_FILE}...")
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("[✅] Encoding generation complete!")
