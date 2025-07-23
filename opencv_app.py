import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/face_mask_detector.h5")

# Load face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face, (150, 150))  # or 224 if your model uses 224x224
            normalized = face_resized / 255.0
            reshaped = np.reshape(normalized, (1, 150, 150, 3))
            result = model.predict(reshaped)[0][0]

            label = "Mask" if result < 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except:
            pass

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()