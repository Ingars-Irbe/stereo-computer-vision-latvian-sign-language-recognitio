import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import time

# Ielādē kalibrācijas parametrus
calib = np.load('stereo_calib.npz')
mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
R, T = calib['R'], calib['T']

# Izveido projekcijas matricas
P1 = np.hstack((mtx_l, np.zeros((3, 1))))  # P1 = K [I | 0]
P2 = np.hstack((mtx_r @ R, mtx_r @ T))     # P2 = K [R | T]

# Inicializē MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializē kameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)

# Sagatavo CSV failu
csv_filename = "hand_points.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, "w") as f:
        header = "timestamp," + ",".join([f"x{i},y{i},z{i}" for i in range(21)]) + "\n"
        f.write(header)

while cap_left.isOpened() and cap_right.isOpened():
    ret_l, frame_left = cap_left.read()
    ret_r, frame_right = cap_right.read()
    if not ret_l or not ret_r:
        break

    # Konvertē attēlus uz RGB (MediaPipe prasa RGB)
    frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    
    # Apstrādā attēlus ar MediaPipe Hands
    results_left = hands.process(frame_left_rgb)
    results_right = hands.process(frame_right_rgb)
    
    if results_left.multi_hand_landmarks and results_right.multi_hand_landmarks:
        hand_left = results_left.multi_hand_landmarks[0]
        hand_right = results_right.multi_hand_landmarks[0]

        h_left, w_left, _ = frame_left.shape
        h_right, w_right, _ = frame_right.shape
        
        points_left, points_right = [], []

        # Apstrādā visus 21 rokas punktus
        for i in range(21):  
            lm_left = hand_left.landmark[i]
            lm_right = hand_right.landmark[i]
            
            x_left, y_left = int(lm_left.x * w_left), int(lm_left.y * h_left)
            x_right, y_right = int(lm_right.x * w_right), int(lm_right.y * h_right)

            points_left.append([x_left, y_left])
            points_right.append([x_right, y_right])

        # Pārvērš punktus par NumPy masīviem (2xN formātā)
        points_left = np.array(points_left, dtype=np.float32).T  # (2, N)
        points_right = np.array(points_right, dtype=np.float32).T  # (2, N)

        # Triangulācija: 2D -> 3D pārveide
        points_4d_hom = cv2.triangulatePoints(P1, P2, points_left, points_right)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Pārvērš homogēnās koordinātas uz 3D

        # Iegūst 3D punktus kā rindu sarakstu
        timestamp = time.time()
        row = [timestamp] + points_3d.flatten().tolist()

        # Saglabā CSV failā
        with open(csv_filename, "a") as f:
            f.write(",".join(map(str, row)) + "\n")

        print("Saglabāti punkti:", row)

    # Parāda attēlus
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
