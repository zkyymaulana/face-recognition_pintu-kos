import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pickle
import time

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained YOLO model and label encoder
class YOLO(nn.Module):
    def __init__(self, num_classes):
        super(YOLO, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 20 * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# baca gambar dan label
with open('trained_model.pkl', 'rb') as file:
    model_state_dict = pickle.load(file)
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# mencocokkan live cam dengan model
num_classes = len(label_encoder.classes_)
model = YOLO(num_classes=num_classes)
model.load_state_dict(model_state_dict)
model.eval()

# Transform for the input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Function to get the current time in seconds
def get_time():
    return time.time()

# Initialize variables for tracking recognized faces
recognized_faces = {}
recognition_threshold = 0.90
time_threshold = 10  # 10 seconds

# Open the webcam
cap = cv2.VideoCapture(0)

# Flag to stop the camera when the door is open
door_open = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_tensor = transform(face).unsqueeze(0)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            predicted_label = label_encoder.inverse_transform(predicted.numpy())[0]
            confidence = confidence.item()

        # Check if the face is recognized or unknown
        if confidence >= recognition_threshold:
            label = f"{predicted_label}: {confidence:.2f}"
            if predicted_label not in recognized_faces:
                recognized_faces[predicted_label] = get_time()
            elif get_time() - recognized_faces[predicted_label] >= time_threshold:
                label = "PINTU TERBUKA"
                door_open = True
        else:
            label = "UNKNOWN!"

        # Draw bounding box and label
        color = (0, 255, 0) if label == "PINTU TERBUKA" else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Live Face Recognition', frame)

    # Break the loop if the door is open or on 'q' key press
    if door_open:
        cv2.imshow('Live Face Recognition', frame)
        cv2.waitKey(3000)  # Wait for 3 seconds
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
