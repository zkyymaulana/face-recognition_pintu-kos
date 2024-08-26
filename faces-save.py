import cv2
import os
import time
import tkinter as tk
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root, output_dir, subfolder_name, max_images=100, frame_width=400, frame_height=300, delay=0.5):
        self.root = root
        self.output_dir = output_dir
        self.subfolder_name = subfolder_name
        self.max_images = max_images
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.delay = delay
        self.count = 0
        self.capturing = False

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.create_output_dir()

        self.panel = tk.Label(root)
        self.panel.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()
        self.start_capturing()

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            self.panel.imgtk = image
            self.panel.configure(image=image)
        self.root.after(10, self.update_frame)

    def start_capturing(self):
        self.capturing = True
        self.capture_image()

    def capture_image(self):
        if self.capturing and self.count < self.max_images:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                image_path = os.path.join(self.output_dir, f'{self.subfolder_name}_{self.count + 1:03d}.jpg')
                cv2.imwrite(image_path, frame)
                print(f"Image saved: {image_path}")
                self.count += 1

            if self.count >= self.max_images:
                self.capturing = False
                print("Capture complete")
            else:
                self.root.after(int(self.delay * 1000), self.capture_image)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    subfolder_name = input("Masukkan nama subfolder: ")

    base_directory = 'images'
    output_directory = os.path.join(base_directory, subfolder_name)

    root = tk.Tk()
    app = CameraApp(root, output_directory, subfolder_name)
    root.mainloop()
