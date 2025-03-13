import cv2
import numpy as np
import mediapipe as mp
import os
import csv
import re
import time

# Iestatām direktorijus
video_dir = "recorded_videos"
output_dir = "gesture_samples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Inicializē MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Atrodam visus video failus
video_files = [f for f in os.listdir(video_dir) if f.endswith(".avi")]

# Regulārais izteiciens, lai izvilktu žesta marķējumu un timestamp no faila nosaukuma
pattern = re.compile(r"gesture_(\d)_([a-z]+)_(\d{8}-\d{6})\.avi")

for video_file in video_files:
    match = pattern.match(video_file)
    if not match:
        print(f"❌ Neizdevās parsēt faila nosaukumu: {video_file}")
        continue

    gesture_label, camera_side, timestamp = match.groups()
    other_side = "right" if camera_side == "left" else "left"

    video_other = f"gesture_{gesture_label}_{other_side}_{timestamp}.avi"
    calib_file = f"gesture_{gesture_label}_calibration_{timestamp}.npz"

    video_path = os.path.join(video_dir, video_file)
    video_other_path = os.path.join(video_dir, video_other)
    calib_path = os.path.join(video_dir, calib_file)

    if not os.path.exists(video_other_path) or not os.path.exists(calib_path):
        print(f"❌ Trūkst otra video vai kalibrācijas faila: {video_other}, {calib_file}")
        continue

    # Ielādē kalibrācijas datus
    calib = np.load(calib_path)
    mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
    mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
    R, T = calib['R'], calib['T']

    # Pārliecināmies, ka T ir 3x1 vektors
    T = T.reshape(3, 1) if T.shape == (3,) else T

    # Izveido projekcijas matricas
    P1 = np.hstack((mtx_l, np.zeros((3, 1))))  # P1 = K [I | 0]
    P2 = np.hstack((mtx_r @ R, mtx_r @ T))     # P2 = K [R | T]

    # Atver video failus
    cap_left = cv2.VideoCapture(video_path if "left" in video_file else video_other_path)
    cap_right = cv2.VideoCapture(video_other_path if "left" in video_file else video_path)

    # Sagatavo CSV faila nosaukumu
    csv_filename = os.path.join(output_dir, f"gesture_{gesture_label}_{timestamp}.csv")

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["gesture_id", "timestamp"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
        writer.writerow(header)

        while cap_left.isOpened() and cap_right.isOpened():
            ret_l, frame_left = cap_left.read()
            ret_r, frame_right = cap_right.read()

            if not ret_l or not ret_r:
                break

            # Konvertē uz RGB
            frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

            # Apstrādā ar MediaPipe
            results_left = hands.process(frame_left_rgb)
            results_right = hands.process(frame_right_rgb)

            if results_left.multi_hand_landmarks and results_right.multi_hand_landmarks:
                hand_left = results_left.multi_hand_landmarks[0]
                hand_right = results_right.multi_hand_landmarks[0]

                h_left, w_left, _ = frame_left.shape
                h_right, w_right, _ = frame_right.shape

                points_left, points_right = [], []

                for i in range(21):
                    lm_left = hand_left.landmark[i]
                    lm_right = hand_right.landmark[i]

                    x_left, y_left = int(lm_left.x * w_left), int(lm_left.y * h_left)
                    x_right, y_right = int(lm_right.x * w_right), int(lm_right.y * h_right)

                    points_left.append([x_left, y_left])
                    points_right.append([x_right, y_right])

                # Pārvērš punktus par NumPy masīviem
                points_left = np.array(points_left, dtype=np.float32).T
                points_right = np.array(points_right, dtype=np.float32).T

                # Triangulācija: 2D -> 3D pārveide
                points_4d_hom = cv2.triangulatePoints(P1, P2, points_left, points_right)
                points_3d = points_4d_hom[:3] / points_4d_hom[3]

                # Sagatavo ierakstu
                frame_timestamp = time.time()
                row = [gesture_label, frame_timestamp] + points_3d.flatten().tolist()
                writer.writerow(row)

    cap_left.release()
    cap_right.release()

    print(f"✅ Saglabāts: {csv_filename}")

cv2.destroyAllWindows()
print("📝 Roku koordinātas veiksmīgi izvilktas no visiem video.")
