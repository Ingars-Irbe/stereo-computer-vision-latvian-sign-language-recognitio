import cv2
import numpy as np
import mediapipe as mp

# Ielādē kalibrācijas parametrus
calib = np.load('stereo_calib.npz')
mtx_l = calib['mtx_l']
dist_l = calib['dist_l']
mtx_r = calib['mtx_r']
dist_r = calib['dist_r']
R = calib['R']
T = calib['T']

# Izveido projekcijas matricas
P1 = np.hstack((mtx_l, np.zeros((3, 1))))  # P1 = K [I | 0]
P2 = np.hstack((mtx_r @ R, mtx_r @ T))     # P2 = K [R | T]

# Inicializē MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializē kameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)

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
        # Pieņemam, ka interesē pirmais atrastais rokas objekts katrā attēlā.
        hand_left = results_left.multi_hand_landmarks[0]
        hand_right = results_right.multi_hand_landmarks[0]
        
        # Izvēlamies kreisās rokas index pirksta gala punktu (parasti landmark indekss 8)
        lm_left = hand_left.landmark[8]
        lm_right = hand_right.landmark[8]
        
        # Konvertējam normalizētās koordinātas uz pikseļu koordinātām
        h_left, w_left, _ = frame_left.shape
        h_right, w_right, _ = frame_right.shape
        
        x_left, y_left = int(lm_left.x * w_left), int(lm_left.y * h_left)
        x_right, y_right = int(lm_right.x * w_right), int(lm_right.y * h_right)

        # Uzzīmējam atzīmētos punktus uz attēliem
        cv2.circle(frame_left, (x_left, y_left), 5, (0, 255, 0), -1)
        cv2.circle(frame_right, (x_right, y_right), 5, (0, 255, 0), -1)

        # PAREIZI FORMATĒTI PUNKTI TRIANGULĀCIJAI (2,1)
        points_left = np.array([[x_left], [y_left]], dtype=np.float32)
        points_right = np.array([[x_right], [y_right]], dtype=np.float32)

        # Triangulācija (pārvērš 2D punktus uz 3D koordinātām)
        points_4d_hom = cv2.triangulatePoints(P1, P2, points_left, points_right)

        # Konvertē no homogēnām koordinātām uz 3D koordinātām
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        # Izgūst X, Y, Z koordinātas
        X, Y, Z = points_3d.flatten()
        print(f"3D Koordinātas: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f} metri")

        # Uzzīmē dziļuma vērtību uz kreisās kameras attēla
        cv2.putText(frame_left, f"Depth: {Z:.2f} m", (x_left, y_left - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Parāda attēlus
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
