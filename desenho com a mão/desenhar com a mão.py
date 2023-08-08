import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk

class NeonDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neon Drawing App")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.drawing = False
        self.hand_open_time = 0
        self.open_duration_threshold = 3
        self.prev_x, self.prev_y = None, None
        self.neon_color = (0, 255, 0)
        self.neon_window = None

        ret, frame = self.cap.read()
        if ret:
            self.camera_height, self.camera_width, _ = frame.shape
            self.camera_frame = tk.Frame(self.root, width=self.camera_width, height=self.camera_height)
            self.camera_frame.pack(side="left", padx=10, pady=10)
            self.neon_frame = tk.Frame(self.root, width=self.camera_width, height=self.camera_height)
            self.neon_frame.pack(side="right", padx=10, pady=10)

        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()
        
        self.neon_label = tk.Label(self.neon_frame)
        self.neon_label.pack()
        
        self.update_frames()
        self.run()

    def update_frames(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Inverte horizontalmente
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                    x, y = int(landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), \
                           int(landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                    thumb_x, thumb_y = int(landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]), \
                                       int(landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                    finger_distance = np.sqrt((x - thumb_x)**2 + (y - thumb_y)**2)

                    if finger_distance < 50:
                        self.drawing = True
                        self.hand_open_time = time.time()
                    else:
                        self.drawing = False

                    if len(results.multi_hand_landmarks) > 1:
                        second_hand_landmarks = results.multi_hand_landmarks[1]
                        second_x, second_y = int(second_hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), \
                                             int(second_hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                        hand_distance = np.sqrt((x - second_x)**2 + (y - second_y)**2)

                        if hand_distance < 50:
                            self.drawing = False

                    if self.drawing and self.prev_x is not None and self.prev_y is not None:
                        cv2.line(frame, (self.prev_x, self.prev_y), (x, y), self.neon_color, 2)
                        cv2.line(self.neon_window, (self.prev_x, self.prev_y), (x, y), self.neon_color, 2)

                    self.prev_x, self.prev_y = x, y

            if not self.drawing and time.time() - self.hand_open_time >= self.open_duration_threshold:
                self.neon_window = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

            self.camera_img = Image.fromarray(frame_rgb)
            self.camera_img = ImageTk.PhotoImage(image=self.camera_img)
            self.camera_label.configure(image=self.camera_img)
            self.camera_label.image = self.camera_img

            neon_img = Image.fromarray(self.neon_window)
            neon_img = ImageTk.PhotoImage(image=neon_img.resize((frame.shape[1], frame.shape[0])))
            self.neon_label.configure(image=neon_img)
            self.neon_label.image = neon_img
            
        self.root.after(10, self.update_frames)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = NeonDrawingApp(root)
