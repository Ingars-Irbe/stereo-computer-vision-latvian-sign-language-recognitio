import cv2
import numpy as np
import mediapipe as mp
import os
import time
import csv

# Ielādē kalibrācijas parametrus
calib = np.load('stereo_calib.npz')
mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
R, T = calib['R'], calib['T']

# Pārliecināmies, ka T ir 3x1 vektors
T = T.reshape(3, 1) if T.shape == (3,) else T

# Izveido projekcijas matricas
P1 = np.hstack((mtx_l, np.zeros((3, 1))))  # P1 = K [I | 0]
P2 = np.hstack((mtx_r @ R, mtx_r @ T))     # P2 = K [R | T]

# Inicializē MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializē kameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)

# Izveido direktoriju, kur tiks saglabāti žestu paraugi
output_dir = "gesture_samples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Definē žestu sarakstu (piemēram, divi žesti)
gestures = ["Gesture 1", "Gesture 2"]
# Varat piešķirt arī numerisku id, ja nepieciešams (piem., { "Gesture 1": 1, "Gesture 2": 2 })
gesture_index = 0  # Pašreizējais žests
sample_count = 0   # Cik reizes pašreizējais žests ir ierakstīts

# Ierakstīšanas parametri
recording = False
record_duration = 5  # Ieraksta ilgums sekundēs
start_time = 0
gesture_rows = []  # Saraksts, kurā katrs elements ir rindiņa ar [gesture_id, timestamp, x1, y1, z1, ..., x21, y21, z21]

print("Nospiediet 's', lai sāktu ierakstīt žestu 5 sekundes, 'q' lai pārtrauktu.")

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
    
    # Attēlo statusu uz kreisās kameras attēla
    status_text = f"Gesture: {gestures[gesture_index]}  Samples: {sample_count}/10"
    if recording:
        status_text += "  [Recording]"
    cv2.putText(frame_left, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if recording:
        current_time = time.time()
        if current_time - start_time <= record_duration:
            if results_left.multi_hand_landmarks and results_right.multi_hand_landmarks:
                hand_left = results_left.multi_hand_landmarks[0]
                hand_right = results_right.multi_hand_landmarks[0]
    
                h_left, w_left, _ = frame_left.shape
                h_right, w_right, _ = frame_right.shape
                
                points_left, points_right = [], []
    
                # Iterē caur visiem 21 rokas punktiem
                for i in range(21):
                    lm_left = hand_left.landmark[i]
                    lm_right = hand_right.landmark[i]
                    
                    x_left, y_left = int(lm_left.x * w_left), int(lm_left.y * h_left)
                    x_right, y_right = int(lm_right.x * w_right), int(lm_right.y * h_right)
    
                    points_left.append([x_left, y_left])
                    points_right.append([x_right, y_right])
    
                # Pārvērš punktus par NumPy masīviem (2 x 21 formātā)
                points_left = np.array(points_left, dtype=np.float32).T
                points_right = np.array(points_right, dtype=np.float32).T
    
                # Triangulācija: 2D -> 3D pārveide
                points_4d_hom = cv2.triangulatePoints(P1, P2, points_left, points_right)
                points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Rezultāts ir (3, 21)
    
                # Sagatavo rindiņu: pirmais elements - žesta id (var izmantot arī gesture_index+1), tad timestamp, tad 63 vērtības
                frame_timestamp = time.time()
                gesture_id = gestures[gesture_index]  # var arī izmantot numerisku id, ja nepieciešams
                # Flatten 3D punktu masīvu (3x21 -> 63 vērtības)
                frame_row = [gesture_id, frame_timestamp] + points_3d.flatten().tolist()
                # Pievieno rindiņu sarakstā
                gesture_rows.append(frame_row)
    
        else:
            # Ierakstīšanas periods ir beidzies – saglabājam visus ierakstītos kadrus failā
            filename = os.path.join(output_dir, f"gesture_{gestures[gesture_index].replace(' ', '_')}_sample_{sample_count+1}.csv")
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Header: gesture_id, timestamp, x0,y0,z0, ..., x20,y20,z20
                header = ["gesture_id", "timestamp"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
                writer.writerow(header)
                for row in gesture_rows:
                    writer.writerow(row)
            print(f"Saglabāts {gestures[gesture_index]} ieraksts (sample {sample_count+1}) ar {len(gesture_rows)} kadriem.")
    
            sample_count += 1
            gesture_rows = []  # Atiestata žesta rindu sarakstu
    
            if sample_count >= 15:  # Pēc 10 paraugiem pāriet uz nākamo žestu
                gesture_index += 1
                sample_count = 0
                if gesture_index >= len(gestures):
                    print("✅ Visi žesti ierakstīti!")
                    break
                print(f"⏩ Pārejam uz {gestures[gesture_index]}")
            recording = False  # Aptur ierakstīšanu
    
    # Parāda attēlus
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    
    key = cv2.waitKey(1) & 0xFF
    # Sāk ierakstīšanu, kad nospiesta "s" poga un ierakstīšana nav aktīva
    if key == ord('s') and not recording:
        print(f"⏺ Sāk ierakstīt {gestures[gesture_index]}!")
        recording = True
        start_time = time.time()
        gesture_rows = []  # Atiestata sarakstu
    if key == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
