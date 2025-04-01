import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# MediaPipe atpazīšanas risinājumi
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands  

# Mape ar video failiem
video_folder = "recorded_videos"
file_folder = "2d_gesture_recording"
if not os.path.exists(file_folder):
    os.makedirs(file_folder)
video_files = [f for f in os.listdir(video_folder) if f.endswith(".avi")]

# Inicializē MediaPipe Holistic modeli
with mp_holistic.Holistic(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9) as holistic:

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        # CSV faila nosaukums (tas pats kā video, bet ar .csv)
        csv_filename = os.path.join(file_folder, video_file.replace(".avi", ".csv"))

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Sagatavo kolonnu nosaukumus: time, gesture, 42 roku punkti, 15 pozu punkti
            columns = ["time", "gesture"]
            for i in range(57):  # Labā roka (21 punkti)
                columns.extend([f"x{i}", f"y{i}"])

            
            writer.writerow(columns)

            frame_number = 0
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Video beidzies

                # Laika atzīme
                timestamp = round(time.time() - start_time, 3)

                # Pārveido attēlu no BGR uz RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Iegūst attēla izmērus
                h, w, _ = frame.shape

                # Noklusējuma vērtība (ja punkts nav atpazīts)
                missing_value = -1

                # Sagatavo rindu ar sākuma vērtībām
                row = [timestamp, video_file]  

                # Labās rokas punkti
                if results.right_hand_landmarks:
                    for lm in results.right_hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                else:
                    row.extend([missing_value, missing_value] * 21)

                # Kreisās rokas punkti
                if results.left_hand_landmarks:
                    for lm in results.left_hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                else:
                    row.extend([missing_value, missing_value] * 21)

                # Pose punkti (tikai 0-14)
                if results.pose_landmarks:
                    for i in range(15):
                        lm = results.pose_landmarks.landmark[i]
                        row.extend([lm.x, lm.y])
                else:
                    row.extend([missing_value, missing_value] * 15)

                # Ieraksta rindu CSV failā
                writer.writerow(row)

                frame_number += 1

        cap.release()

print("Datu apstrāde pabeigta! CSV faili ir saglabāti mapē 'recorded_videos'.")
