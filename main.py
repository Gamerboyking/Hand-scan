from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
import io
from PIL import Image

app = FastAPI()

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

@app.post("/detect")
async def detect_hand(file: UploadFile = File(...)):
    # Image read karein
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    img_np = np.array(img)
    
    # Hand landmarks detect karein
    results = hands.process(img_np)
    
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Sirf Index Finger Tip (Point 8) bhej rahe hain speed ke liye
            index_finger = hand_landmarks.landmark[8]
            landmarks.append({"x": index_finger.x, "y": index_finger.y})
            
    return {"landmarks": landmarks}

@app.get("/")
def home():
    return {"status": "Server is running!"}
  
