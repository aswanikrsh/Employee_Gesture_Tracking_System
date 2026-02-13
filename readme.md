# 🧑‍💼 Employee Gesture Tracking System
## Employee-Only Face Recognition + Gesture Classification + CSV Logging

---

## 📌 Project Overview

The **Employee Gesture Tracking System** is a real-time AI-powered monitoring system that:

- Detects only registered employees
- Performs face recognition using embeddings
- Tracks employees across frames
- Classifies posture as **Perfect** or **Lazy**
- Logs results into a CSV file
- Supports RTSP CCTV camera streams
- Gracefully shuts down using `Ctrl + C`

Unknown persons are automatically ignored.

---

## 🚀 Key Features

- YOLOv8 person detection
- InsightFace face recognition
- Employee-only filtering
- Real-time tracking with identity cache
- Posture-based gesture classification
- Automatic CSV logging
- Multi-camera support
- Threaded architecture for smooth performance
- Graceful shutdown system

---

## 🧠 Technologies Used

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- InsightFace
- NumPy
- Threading & Queue
- CSV file handling

---

## 📂 Project Structure

```
Employee-Gesture-Tracking-System/
│
├── employees/
│   ├── John/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── Mary/
│   │   └── img1.jpg
│   └── embeddings_cache/
│
├── employee_log.csv
├── main.py
└── README.md
```

---

## 🧑‍🤝‍🧑 Employee Registration

To register employees:

1. Open the `employees/` folder.
2. Create a folder with the employee name.
3. Add at least one clear face image inside the folder.

Example:

```
employees/
   ├── John/
   │     ├── john1.jpg
   ├── Mary/
   │     ├── mary1.jpg
```

On first run:
- Face embeddings will be generated automatically.
- Embeddings are stored inside `employees/embeddings_cache/`.

---

## ⚙️ Installation Guide

### Step 1: Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / Mac**
```bash
source venv/bin/activate
```

---

### Step 2: Install Required Libraries

```bash
pip install ultralytics opencv-python insightface numpy onnxruntime
```

If using GPU:

```bash
pip install onnxruntime-gpu
```

---

## ▶️ Running the System

Edit camera URLs inside the Python file:

```python
CAMERA_URLS = {
    "Camera1": "rtsp://username:password@ip:port/stream"
}
```

Run the system:

```bash
python main.py
```

Press:
- `Q` to quit
- `Ctrl + C` for graceful shutdown

---

## 📝 System Workflow

### 1️⃣ Person Detection
YOLOv8 detects only the **person** class.

### 2️⃣ Face Recognition
InsightFace extracts face embeddings and compares them with registered employees.

### 3️⃣ Employee Filtering
- If similarity score ≥ threshold → Employee name displayed
- If not → Person ignored

### 4️⃣ Gesture Classification
Posture is calculated using bounding box ratio:

```
Perfect → Upright posture
Lazy → Leaning posture
```

### 5️⃣ CSV Logging

Every detection is saved in:

```
employee_log.csv
```

Format:

```
Timestamp, Camera, Employee, Gesture
```

Example:

```
2026-02-13 10:25:30, Camera1, John, Perfect
```

---

## 🔧 Configuration Parameters

Inside the code:

```python
FRAME_DOWNSCALE = 640
UNKNOWN_THRESHOLD = 0.32
IGNORE_UNKNOWN = True
MAX_FPS_LOSS_TOLERANCE = 0.7
```

You can modify:

- Detection resolution
- Face similarity threshold
- Frame loss tolerance
- Unknown filtering behavior

---

## 🛑 Graceful Shutdown

The system safely shuts down when:

- `Q` key is pressed
- `Ctrl + C` is used

The system will:
- Stop camera threads
- Release RTSP streams
- Close OpenCV windows
- Finish remaining queued frames

---

## 📊 Output

- Real-time video with:
  - Green bounding box
  - Employee name
  - Gesture label
- Automatic CSV log generation

---

## 🔮 Future Improvements

- Phone usage detection
- Head-turn detection
- Excel report generation
- Database integration (MySQL / Firebase)
- Web dashboard (Flask / React)
- Real-time alert system

---

## 👨‍💻 Author

**Aswanikrishna**  
Diploma in Artificial Intelligence & Data Engineering  
Skilled in Python, Machine Learning, and Computer Vision  

---

## 📜 License

This project is for educational and research purposes only.
