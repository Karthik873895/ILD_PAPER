import cv2
import joblib
import numpy as np
from pathlib import Path
from collections import OrderedDict
from feature_extraction import extract_features

# Configuration Parameters (adjust these as needed)
MAX_DISAPPEARED = 5       # Frames to keep a face without detection (higher = more persistent boxes)
DISTANCE_THRESHOLD = 50   # Pixel distance to consider the same face (lower = more sensitive tracking)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

class FaceTracker:
    def __init__(self, max_disappeared=5, distance_threshold=50):
        self.faces = OrderedDict()  # {id: (rect, disappeared_count)}
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def update(self, current_rects):
        # Mark all existing faces as potentially disappeared
        for face_id in list(self.faces.keys()):
            rect, count = self.faces[face_id]
            self.faces[face_id] = (rect, count + 1)

        # Update with current detections
        for rect in current_rects:
            x, y, w, h = rect
            current_center = np.array([x + w/2, y + h/2])
            
            # Find closest existing face
            matched_id = None
            min_distance = float('inf')
            
            for face_id, (existing_rect, _) in self.faces.items():
                ex, ey, ew, eh = existing_rect
                existing_center = np.array([ex + ew/2, ey + eh/2])
                distance = np.linalg.norm(current_center - existing_center)
                
                if distance < self.distance_threshold and distance < min_distance:
                    min_distance = distance
                    matched_id = face_id

            if matched_id is not None:
                # Update existing face
                self.faces[matched_id] = (rect, 0)
            else:
                # Add new face
                self.faces[self.next_id] = (rect, 0)
                self.next_id += 1

        # Remove faces that disappeared for too long
        self.faces = OrderedDict(
            (fid, data) for fid, data in self.faces.items() 
            if data[1] <= self.max_disappeared
        )
        
        return self.faces

# Load models
models_dir = Path(__file__).parent.parent / "models"
svm = joblib.load(models_dir / "ild_svm_model.pkl")
scaler = joblib.load(models_dir / "scaler.pkl") 
pca = joblib.load(models_dir / "pca.pkl")

# Initialize tracker with our parameters
tracker = FaceTracker(
    max_disappeared=MAX_DISAPPEARED,
    distance_threshold=DISTANCE_THRESHOLD
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    active_faces = tracker.update(faces)
    
    for face_id, ((x,y,w,h), _) in active_faces.items():
        # Draw the same "Person" label for all faces
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, "Person", (x,y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()