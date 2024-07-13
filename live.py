import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from facenet_pytorch import InceptionResnetV1, MTCNN as FaceNetMTCNN
import tkinter as tk
from tkinter import messagebox
import threading
import time
import queue

class LiveEmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Emotion Detection")

        self.queue = queue.Queue()
        self.model = load_model('fer2013_custom_model.h5')
        self.face_net_detector = FaceNetMTCNN()
        self.face_net_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (255, 0, 0),
            'Happy': (0, 255, 255), 'Sad': (255, 255, 0), 'Surprise': (255, 0, 255), 'Neutral': (255, 255, 255)
        }
        self.selected_emotions = []
        self.cameras = []
        self.selected_cameras = []
        self.show_all_bounding_boxes = tk.BooleanVar(value=True)
        self.enable_alerts = tk.BooleanVar(value=True)
        self.alert_duration = tk.DoubleVar(value=2.0)  # Duration in seconds
        self.alert_cooldown = tk.DoubleVar(value=10.0)  # Cooldown in seconds
        self.last_alert_time = 0
        self.emotion_timers = {}
        self.init_gui()

    def init_gui(self):
        tk.Button(self.root, text="Search Cameras", command=self.search_cameras).grid(row=0, column=0)
        tk.Button(self.root, text="Select Emotions", command=self.select_emotions).grid(row=1, column=0)
        tk.Button(self.root, text="What to Detect", command=self.what_to_detect).grid(row=2, column=0)
        tk.Button(self.root, text="Alert Settings", command=self.alert_settings).grid(row=3, column=0)
        tk.Button(self.root, text="Start Detection", command=self.start_detection).grid(row=4, column=0)

        self.camera_frame = tk.Frame(self.root)
        self.camera_frame.grid(row=5, column=0, columnspan=3)

        self.root.after(100, self.process_queue)

    def process_queue(self):
        try:
            task = self.queue.get_nowait()
            task()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def search_cameras(self):
        def check_camera(index):
            cap = cv2.VideoCapture(index)
            if cap is None or not cap.isOpened():
                return False
            cap.release()
            return True

        def refresh_cameras():
            for widget in camera_selection_window.winfo_children():
                widget.destroy()
            populate_camera_list()

        def populate_camera_list():
            self.cameras = [i for i in range(10) if check_camera(i)]

            tk.Label(camera_selection_window, text="Select cameras to use:").grid(row=0, column=0, columnspan=2)

            self.camera_vars = []
            for i in self.cameras:
                var = tk.BooleanVar(value=i in self.selected_cameras)
                self.camera_vars.append(var)
                tk.Checkbutton(camera_selection_window, text=f"Camera {i}", variable=var).grid(row=i + 1, column=0, sticky='w')

            tk.Button(camera_selection_window, text="Refresh", command=refresh_cameras).grid(row=len(self.cameras) + 1, column=0, columnspan=2)
            tk.Button(camera_selection_window, text="Save", command=save_selections).grid(row=len(self.cameras) + 2, column=0, columnspan=2)

        def save_selections():
            self.selected_cameras = [i for i, var in zip(self.cameras, self.camera_vars) if var.get()]
            camera_selection_window.destroy()

        camera_selection_window = tk.Toplevel(self.root)
        camera_selection_window.title("Select Cameras")

        populate_camera_list()

    def select_emotions(self):
        emotion_selection_window = tk.Toplevel(self.root)
        emotion_selection_window.title("Select Emotions")

        tk.Label(emotion_selection_window, text="Select emotions to highlight:").grid(row=0, column=0, columnspan=2)

        check_vars = [tk.BooleanVar(value=(self.emotion_label[i], self.emotion_colors[self.emotion_label[i]]) in self.selected_emotions) for i in range(len(self.emotion_label))]
        for i, emotion in enumerate(self.emotion_label):
            tk.Checkbutton(emotion_selection_window, text=emotion, variable=check_vars[i]).grid(row=i + 1, column=0, sticky='w')

        def save_selections():
            self.selected_emotions = [(self.emotion_label[i], self.emotion_colors[self.emotion_label[i]]) for i, var in enumerate(check_vars) if var.get()]
            emotion_selection_window.destroy()

        tk.Button(emotion_selection_window, text="Save", command=save_selections).grid(row=len(self.emotion_label) + 1, column=0, columnspan=2)

    def what_to_detect(self):
        detection_selection_window = tk.Toplevel(self.root)
        detection_selection_window.title("What to Detect")

        tk.Label(detection_selection_window, text="Select detection mode:").grid(row=0, column=0, columnspan=2)

        tk.Checkbutton(detection_selection_window, text="Show all bounding boxes", variable=self.show_all_bounding_boxes).grid(row=1, column=0, sticky='w')

        def save_selections():
            detection_selection_window.destroy()

        tk.Button(detection_selection_window, text="Save", command=save_selections).grid(row=2, column=0, columnspan=2)

    def alert_settings(self):
        alert_settings_window = tk.Toplevel(self.root)
        alert_settings_window.title("Alert Settings")

        tk.Label(alert_settings_window, text="Enable Alerts").grid(row=0, column=0, columnspan=2)
        tk.Checkbutton(alert_settings_window, text="Enable", variable=self.enable_alerts).grid(row=1, column=0, sticky='w')

        tk.Label(alert_settings_window, text="Duration to Display Emotion (seconds)").grid(row=2, column=0, columnspan=2)
        tk.Entry(alert_settings_window, textvariable=self.alert_duration).grid(row=3, column=0, columnspan=2)

        tk.Label(alert_settings_window, text="Alert Cooldown (seconds)").grid(row=4, column=0, columnspan=2)
        tk.Entry(alert_settings_window, textvariable=self.alert_cooldown).grid(row=5, column=0, columnspan=2)

        def save_selections():
            alert_settings_window.destroy()

        tk.Button(alert_settings_window, text="Save", command=save_selections).grid(row=6, column=0, columnspan=2)

    def start_detection(self):
        if not self.selected_cameras:
            messagebox.showerror("Input Error", "Please select at least one camera.")
            return

        self.stop_event = threading.Event()
        self.threads = []
        for cam in self.selected_cameras:
            thread = threading.Thread(target=self.detect_emotions, args=(cam,))
            thread.start()
            self.threads.append(thread)

        self.stop_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection)
        self.stop_button.grid(row=6, column=0)

    def stop_detection(self):
        self.stop_event.set()
        threading.Thread(target=self.wait_for_threads_to_finish).start()

    def wait_for_threads_to_finish(self):
        for thread in self.threads:
            thread.join()

        self.queue.put(lambda: self.stop_button.grid_forget())
        self.queue.put(lambda: tk.Button(self.root, text="Start Detection", command=self.start_detection).grid(row=4, column=0))
        self.queue.put(lambda: messagebox.showinfo("Detection Stopped", "Emotion detection has been stopped."))

    def detect_emotions(self, cam_index):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Failed to open camera {cam_index}")
            return

        window_name = f"Camera {cam_index}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Enable window resizing

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            print(f"Detected faces: {faces}")

            for (x, y, x2, y2) in faces:
                face = frame[y:y2, x:x2]
                if face.size == 0:
                    continue  # Skip if the face region is empty
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                normalized_face = resized_face / 255.0
                reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
                emotion_prediction = self.model.predict(reshaped_face)
                max_index = np.argmax(emotion_prediction[0])
                emotion_label = self.emotion_label[max_index]
                emotion_color = self.emotion_colors[emotion_label]

                # Draw bounding box for all detected faces
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), 2)
                if emotion_label in [e[0] for e in self.selected_emotions]:
                    cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

                    if self.enable_alerts.get():
                        self.handle_alert(emotion_label, (x, y, x2, y2))

            self.update_camera_feed(window_name, frame)

        cap.release()
        cv2.destroyWindow(window_name)

    def handle_alert(self, emotion_label, face_box):
        current_time = time.time()
        if (current_time - self.last_alert_time) < self.alert_cooldown.get():
            return

        if emotion_label not in self.emotion_timers:
            self.emotion_timers[emotion_label] = current_time
        else:
            if (current_time - self.emotion_timers[emotion_label]) >= self.alert_duration.get():
                self.last_alert_time = current_time
                self.emotion_timers.pop(emotion_label, None)
                self.queue.put(lambda: threading.Thread(target=self.alert_user, args=(emotion_label,)).start())

    def detect_faces(self, frame):
        faces = []

        boxes, _ = self.face_net_detector.detect(frame)
        if boxes is not None:
            for (x1, y1, x2, y2) in boxes:
                faces.append([int(x1), int(y1), int(x2), int(y2)])

        return self.non_max_suppression(np.array(faces), overlapThresh=0.3)

    def update_camera_feed(self, window_name, frame):
        # Get the current window size
        window_size = cv2.getWindowImageRect(window_name)[2:]

        # Resize frame to fit the window while maintaining aspect ratio
        height, width, _ = frame.shape
        aspect_ratio = width / height
        window_width, window_height = window_size

        if window_width / aspect_ratio <= window_height:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)
        else:
            new_width = int(window_height * aspect_ratio)
            new_height = window_height

        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a blank image with the window size and place the resized frame on it
        blank_image = np.zeros((window_height, window_width, 3), np.uint8)
        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2
        blank_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

        cv2.imshow(window_name, blank_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_event.set()

    def non_max_suppression(self, boxes, overlapThresh=0.3):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, y2[i] - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    def alert_user(self, emotion):
        messagebox.showinfo("Emotion Detected", f"{emotion} detected")

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveEmotionDetectionApp(root)
    root.mainloop()
