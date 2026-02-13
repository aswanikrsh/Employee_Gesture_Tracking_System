
#!/usr/bin/env python3
"""
Employee Gesture Tracking System (EMPLOYEE-ONLY + CSV Logging)
- Detect only employees from images in employees/<Name>/ folders
- Logs detected employees and gestures into employee_log.csv
- No employee ID shown
- Gesture tracking and smoothing remain
- Graceful exit on Ctrl+C
"""

import os
import cv2
import time
import math
import threading
import queue
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ============================================================
# GLOBAL SETTINGS
# ============================================================
FRAME_DOWNSCALE = 640
FACE_SIZE = 128
MAX_FPS_LOSS_TOLERANCE = 0.7
UNKNOWN_THRESHOLD = 0.32
IGNORE_UNKNOWN = True  # ignore non-employee faces

stop_event = threading.Event()
frame_q = queue.Queue(maxsize=12)
identity_cache = {}
next_track_id = 1
cache_lock = threading.Lock()

EMP_DIR = "employees"
EMP_EMBED_CACHE = os.path.join(EMP_DIR, "embeddings_cache")
os.makedirs(EMP_EMBED_CACHE, exist_ok=True)

CSV_FILE = "employee_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Camera", "Employee", "Gesture"])

# ============================================================
# LOAD EMPLOYEE EMBEDDINGS FROM FOLDER STRUCTURE
# ============================================================
def load_employee_embeddings():
    embeddings = []
    names = []

    print("Loading employee embeddings...")
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(FACE_SIZE, FACE_SIZE))

    for emp_name in os.listdir(EMP_DIR):
        emp_path = os.path.join(EMP_DIR, emp_name)
        if not os.path.isdir(emp_path):
            continue

        cache_file = os.path.join(EMP_EMBED_CACHE, f"{emp_name}.npy")
        if os.path.exists(cache_file):
            emb = np.load(cache_file)
            embeddings.append(emb)
            names.append(emp_name)
            continue

        for file in os.listdir(emp_path):
            if file.lower().endswith((".jpg",".jpeg",".png")):
                img_path = os.path.join(emp_path, file)
                img = cv2.imread(img_path)
                faces = face_app.get(img)
                if faces:
                    emb = faces[0].embedding
                    emb = emb / (np.linalg.norm(emb)+1e-6)
                    embeddings.append(emb)
                    names.append(emp_name)
                    np.save(cache_file, emb)
                    print(f"Saved embedding for {emp_name}")
                    break
        else:
            print(f"No face detected for {emp_name}, skipping...")

    print(f"Total employees loaded: {len(names)}")
    return embeddings, names

EMP_EMBS, EMP_NAMES = load_employee_embeddings()

def match_employee(embedding):
    best_score = -1
    best_name = "Unknown"
    for emp_emb, emp_name in zip(EMP_EMBS, EMP_NAMES):
        score = np.dot(embedding, emp_emb)
        if score > best_score:
            best_score = score
            best_name = emp_name
    return best_name if best_score >= UNKNOWN_THRESHOLD else "Unknown"

# ============================================================
# TRACKING
# ============================================================
def assign_track_id(xyxy):
    global next_track_id
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1+x2)/2, (y1+y2)/2
    best_id = None
    best_dist = 9999

    with cache_lock:
        for tid, data in identity_cache.items():
            bx1, by1, bx2, by2 = data["bbox"]
            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
            dist = math.dist((cx, cy), (bcx, bcy))
            if dist < best_dist and dist < 120:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1
            identity_cache[best_id] = {"name":"Detecting...", "last_seen":time.time(),"bbox":xyxy}

        identity_cache[best_id]["bbox"] = xyxy
        identity_cache[best_id]["last_seen"] = time.time()

    return best_id

def clean_cache():
    now = time.time()
    with cache_lock:
        for tid in list(identity_cache.keys()):
            if now - identity_cache[tid]["last_seen"] > MAX_FPS_LOSS_TOLERANCE:
                del identity_cache[tid]

# ============================================================
# RTSP READER THREAD
# ============================================================
class RTSPReader(threading.Thread):
    def __init__(self, url, cam_name):
        super().__init__(daemon=True)
        self.url = url
        self.cam_name = cam_name
        self.cap = cv2.VideoCapture(url)
        self.running = True
        self.start()

    def run(self):
        while self.running and not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.2)
                continue
            if not frame_q.full():
                frame_q.put((self.cam_name, frame))
        self.cap.release()

    def stop(self):
        self.running = False

# ============================================================
# INFERENCE WORKER
# ============================================================
class InferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.yolo = YOLO("yolov8n.pt")
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0, det_size=(FACE_SIZE, FACE_SIZE))
        self.start()

    def run(self):
        while not stop_event.is_set():
            try:
                cam_name, frame = frame_q.get(timeout=1)
            except queue.Empty:
                continue

            small = cv2.resize(frame,(FRAME_DOWNSCALE,FRAME_DOWNSCALE))
            results = self.yolo(small,verbose=False)[0]

            for box in results.boxes:
                if int(box.cls[0]) != 0:  # person only
                    continue

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                scale_x, scale_y = frame.shape[1]/FRAME_DOWNSCALE, frame.shape[0]/FRAME_DOWNSCALE
                X1,Y1,X2,Y2 = int(x1*scale_x),int(y1*scale_y),int(x2*scale_x),int(y2*scale_y)

                crop = frame[Y1:Y2,X1:X2]
                faces = self.face_app.get(crop)

                employee_name = "Unknown"
                if faces:
                    face = faces[0]
                    emb = face.embedding / (np.linalg.norm(face.embedding)+1e-6)
                    employee_name = match_employee(emb)

                if employee_name == "Unknown":
                    continue  # skip non-employees

                track_id = assign_track_id([X1,Y1,X2,Y2])
                identity_cache[track_id]["name"] = employee_name

                # Draw
                cv2.rectangle(frame,(X1,Y1),(X2,Y2),(0,255,0),2)
                cv2.putText(frame,employee_name,(X1,Y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                # Gesture classification
                torso_angle = abs((Y2-Y1)/(X2-X1))
                gesture = "Perfect" if torso_angle<1.2 else "Lazy"
                cv2.putText(frame,gesture,(X1,Y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)

                # ================= CSV Logging =================
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, cam_name, employee_name, gesture])

            clean_cache()
            cv2.imshow(cam_name,frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
            frame_q.task_done()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    CAMERA_URLS = {
        #   'Cam1': 'rtsp://admin:Admin%40123@192.168.1.200:554/Streaming/Channels/101',
        # 'Cam2': 'rtsp://admin:Admin%25123@192.168.29.200:554/Streaming/Channels/201',
        'Cam3': 'rtsp://admin:Admin%40123@192.168.1.200:554/Streaming/Channels/301',
        # 'Cam4': 'rtsp://admin:Admin%25123@192.168.29.200:554/Streaming/Channels/401',
    }

    # Start RTSP readers
    readers = {}
    for cam,url in CAMERA_URLS.items():
        readers[cam] = RTSPReader(url,cam)
        print(f"Started {cam}")

    # Start inference worker
    worker = InferenceWorker()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
            # Optional: clean up any stuck frames
            while not frame_q.empty():
                frame_q.get_nowait()
                frame_q.task_done()
    except KeyboardInterrupt:
        stop_event.set()  # Graceful shutdown

    # Stop all readers
    for r in readers.values():
        r.stop()

    # Wait for remaining frames to finish
    frame_q.join()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("System shutdown complete.")
