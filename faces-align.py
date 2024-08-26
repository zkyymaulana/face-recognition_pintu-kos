import cv2
import os

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def align_faces(input_dir, output_dir, face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                output_subdir = os.path.dirname(output_path)
                create_output_dir(output_subdir)

                img = cv2.imread(input_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w]
                    resized_face = cv2.resize(face, (160, 160))
                    aligned_face_path = os.path.join(output_subdir, file)
                    cv2.imwrite(aligned_face_path, resized_face)
                    print(f"Aligned face saved: {aligned_face_path}")
                    break  # Assuming one face per image for simplicity

if __name__ == '__main__':
    input_dir = 'images'  # Path to the dataset folder
    output_dir = 'dataset'  # Path to the output folder
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Path to the Haar cascade XML file

    align_faces(input_dir, output_dir, face_cascade_path)
