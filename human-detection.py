import cv2
import numpy as np
import torch
import time
import random
import string

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]

# Load the video or webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

def get_human_box(frame, x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Crop the image based on the bounding box
    cropped_image = frame_resized[y1:y2, x1:x2]
    random_filename=create_random_filename()
    filename=f"human_{random_filename}" 
    cv2.imwrite(f"./images/faces/{filename}", cropped_image)

def create_random_filename():
    characters = string.ascii_letters + string.digits   
    random_string = ''.join(random.choice(characters) for _ in range(8))
    epoch_time = int(time.time())

    return f"{random_string}_{epoch_time}.jpg" 


while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing (optional)
    frame_resized = cv2.resize(frame, (680, 360))

    # Perform detection
    results = model(frame_resized)

    # Draw bounding boxes
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_resized, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            random_filename=create_random_filename()
            filename = f"snapshot_{random_filename}"
            cv2.imwrite(f"./images/snapshots/{filename}", frame_resized)
            get_human_box(frame_resized, int(x1), int(y1), int(x2), int(y2))

    cv2.imshow('Human Detection', frame_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



