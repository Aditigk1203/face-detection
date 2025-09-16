import cv2
import os

# Path to store faces
DATASET_PATH = "face_database"

# Create dataset folder if not exists
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Load Haarcascade face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

person_name = input("Enter person's name: ").strip()
person_folder = os.path.join(DATASET_PATH, person_name)

if not os.path.exists(person_folder):
    os.makedirs(person_folder)

count = 0
print("Capturing faces... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Draw rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Crop and save face
        face_img = frame[y:y+h, x:x+w]
        face_filename = os.path.join(person_folder, f"face_{count}.jpg")
        cv2.imwrite(face_filename, face_img)
        count += 1

    cv2.imshow("Face Capture", frame)

    # Quit if q pressed or enough samples collected
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Saved {count} face images for {person_name} in {person_folder}")
