import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import dlib
from facenet_pytorch import InceptionResnetV1, MTCNN as FaceNetMTCNN
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# Load pre-trained emotion detection model
model = load_model('fer2013_custom_model.h5')

# Create Dlib detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load OpenCV Haar Cascade face detector
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize FaceNet model for secondary face verification
face_net_detector = FaceNetMTCNN()
face_net_model = InceptionResnetV1(pretrained='vggface2').eval()

# Emotion labels
emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion to color mapping
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 255, 0),    # Green
    'Fear': (255, 0, 0),       # Blue
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 255, 0),      # Cyan
    'Surprise': (255, 0, 255), # Magenta
    'Neutral': (255, 255, 255) # White
}

# Tkinter GUI setup
root = tk.Tk()
root.title("Emotion Detection in Video")

selected_emotions = []

# Function to browse for video file
def browse_file():
    video_path.set(filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")]))

# Function to select output directory
def select_output_directory():
    output_directory.set(filedialog.askdirectory())

# Function to select emotions
def select_emotions():
    emotion_selection_window = tk.Toplevel(root)
    emotion_selection_window.title("Select Emotions")
    
    tk.Label(emotion_selection_window, text="Select emotions to highlight:").grid(row=0, column=0, columnspan=2)
    
    check_vars = [tk.BooleanVar(value=(emotion_label[i], emotion_colors[emotion_label[i]]) in selected_emotions) for i in range(len(emotion_label))]
    for i, emotion in enumerate(emotion_label):
        tk.Checkbutton(emotion_selection_window, text=emotion, variable=check_vars[i]).grid(row=i+1, column=0, sticky='w')
    
    def save_selections():
        global selected_emotions
        selected_emotions = [(emotion_label[i], emotion_colors[emotion_label[i]]) for i, var in enumerate(check_vars) if var.get()]
        emotion_selection_window.destroy()
    
    tk.Button(emotion_selection_window, text="Save", command=save_selections).grid(row=len(emotion_label)+1, column=0, columnspan=2)

# Function to start the detection process
def start_detection():
    video = video_path.get()
    output_dir = output_directory.get()
    start = start_time.get()
    end = end_time.get()
    output_name = output_filename.get()

    if not video or not output_dir or not start or not end or not output_name:
        messagebox.showerror("Input Error", "Please fill all fields.")
        return

    try:
        start = float(start)
        end = float(end)
    except ValueError:
        messagebox.showerror("Input Error", "Start and End times must be numeric.")
        return

    output_path = os.path.join(output_dir, output_name + '.mp4')

    progress_window = tk.Toplevel(root)
    progress_window.title("Processing Video")
    tk.Label(progress_window, text="Processing...").grid(row=0, column=0)
    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
    progress_bar.grid(row=1, column=0)
    progress_window.update_idletasks()

    def process_video_thread():
        process_video(video, output_path, start, end, progress_bar)
        progress_window.destroy()

    threading.Thread(target=process_video_thread).start()

def nms(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def process_video(video_path, output_path, start_time, end_time, progress_bar):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    overall_emotion_count = np.zeros(len(emotion_label), dtype=int)
    overall_total_faces = 0

    trackers = []
    face_ids = 0
    known_faces = {}
    face_last_seen = {}

    detection_interval = 5
    frame_count = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    progress_bar['maximum'] = end_frame - start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > end_frame:
            break

        frame_count += 1

        new_trackers = []
        for tracker in trackers:
            success, bbox = tracker['tracker'].update(frame)
            if success:
                tracker['bbox'] = bbox
                new_trackers.append(tracker)

        if frame_count % detection_interval == 0:
            faces = []
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces_dlib = detector(gray_frame)
            detected_faces_haar = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for face in detected_faces_dlib:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                faces.append((x, y, x + w, y + h))
            
            for (x, y, w, h) in detected_faces_haar:
                faces.append((x, y, x + w, y + h))

            faces = np.array(faces)
            nms_faces = nms(faces, 0.3)

            verified_faces = []
            for (x1, y1, x2, y2) in nms_faces:
                face_roi = frame[y1:y2, x1:x2]
                if verify_face(face_roi):
                    landmarks = shape_predictor(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB), dlib.rectangle(0, 0, x2-x1, y2-y1))
                    if landmarks:
                        verified_faces.append((x1, y1, x2-x1, y2-y1))
            
            faces = verified_faces
            print(f"Detected {len(faces)} faces at frame {frame_count}")
        else:
            faces = []

        for (x, y, w, h) in faces:
            new_face = True
            for tracker in new_trackers:
                (tx, ty, tw, th) = [int(v) for v in tracker['bbox']]
                if (x > tx - 0.5 * w and y > ty - 0.5 * h and x + w < tx + 1.5 * tw and y + h < ty + 1.5 * th):
                    new_face = False
                    break
            if new_face:
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, (x, y, w, h))
                new_trackers.append({'tracker': tracker, 'bbox': (x, y, w, h), 'id': face_ids})
                face_last_seen[face_ids] = frame_count
                face_ids += 1

        trackers = new_trackers

        for tracker in trackers:
            (x, y, w, h) = [int(v) for v in tracker['bbox']]
            face_roi = frame[y:y+h, x:x+w]

            gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face_roi, (48, 48))
            resized_face = resized_face / 255.0
            resized_face = np.expand_dims(resized_face, axis=-1)

            predictions = model.predict(np.expand_dims(resized_face, axis=0))
            dominant_emotion_index = np.argmax(predictions)
            dominant_emotion = emotion_label[dominant_emotion_index]

            if tracker['id'] not in known_faces or (frame_count - face_last_seen[tracker['id']]) > detection_interval:
                known_faces[tracker['id']] = dominant_emotion_index
                face_last_seen[tracker['id']] = frame_count

            rect_color = (255, 0, 0)
            for emotion, color in selected_emotions:
                if dominant_emotion == emotion:
                    rect_color = color
                    break

            cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)

            overall_emotion_count[dominant_emotion_index] += 1

        overall_total_faces += len(faces)

        emotion_percentages = (overall_emotion_count / overall_total_faces) * 100 if overall_total_faces > 0 else np.zeros(len(emotion_label), dtype=int)

        out.write(frame)

        progress_bar['value'] = current_frame - start_frame
        root.update_idletasks()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Overall Emotion Counts:")
    for i, emotion in enumerate(emotion_label):
        print(f'{emotion}: {overall_emotion_count[i]}')

    print(f'Total Unique Faces Detected: {len(known_faces)}')

def verify_face(face_img):
    try:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_tensor = face_net_detector(face_img)
        if face_tensor is not None:
            return True
    except:
        pass
    return False

video_path = tk.StringVar()
output_directory = tk.StringVar()
start_time = tk.StringVar()
end_time = tk.StringVar()
output_filename = tk.StringVar()

tk.Label(root, text="Video File:").grid(row=0, column=0)
tk.Entry(root, textvariable=video_path, width=50).grid(row=0, column=1)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2)

tk.Label(root, text="Output Directory:").grid(row=1, column=0)
tk.Entry(root, textvariable=output_directory, width=50).grid(row=1, column=1)
tk.Button(root, text="Browse", command=select_output_directory).grid(row=1, column=2)

tk.Label(root, text="Start Time (s):").grid(row=2, column=0)
tk.Entry(root, textvariable=start_time).grid(row=2, column=1)

tk.Label(root, text="End Time (s):").grid(row=3, column=0)
tk.Entry(root, textvariable=end_time).grid(row=3, column=1)

tk.Label(root, text="Output Filename:").grid(row=4, column=0)
tk.Entry(root, textvariable=output_filename).grid(row=4, column=1)

tk.Button(root, text="Select Emotions", command=select_emotions).grid(row=5, column=0)
tk.Button(root, text="Start Detection", command=start_detection).grid(row=6, column=0)

if __name__ == "__main__":
    root.mainloop()
