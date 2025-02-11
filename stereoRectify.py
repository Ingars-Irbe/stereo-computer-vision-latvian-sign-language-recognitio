import cv2
import numpy as np
import mediapipe as mp

# Ielādē kalibrācijas parametrus
calib = np.load('stereo_calib.npz')
mtx_l = calib['mtx_l']
dist_l = calib['dist_l']
mtx_r = calib['mtx_r']
dist_r = calib['dist_r']
T = calib['T']
baseline = np.linalg.norm(T)
focal_length = mtx_l[0, 0]  # piemērs

# Inicializē MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializē StereoBM (vai StereoSGBM)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

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
    
    depth_value = None  # Mainīgais dziļuma vērtībai
    # Pārliecināmies, ka abos attēlos ir atrasti rokas atzīmējumi
    if results_left.multi_hand_landmarks and results_right.multi_hand_landmarks:
        # Pieņemsim, ka interesē mūs pirmais atrastais rokas objekts katrā attēlā.
        hand_left = results_left.multi_hand_landmarks[0]
        hand_right = results_right.multi_hand_landmarks[0]
        
        # Izvēlamies kreisās rokas index pirksta gala punktu (parasti landmark indekss 8)
        lm_left = hand_left.landmark[8]
        lm_right = hand_right.landmark[8]
        
        # Konvertējam normalizētās koordinātas uz pikseļu koordinātēm
        h_left, w_left, _ = frame_left.shape
        h_right, w_right, _ = frame_right.shape
        
        x_left = int(lm_left.x * w_left)
        y_left = int(lm_left.y * h_left)
        
        x_right = int(lm_right.x * w_right)
        y_right = int(lm_right.y * h_right)
        
        # Uzzīmējam atzīmētos punktus uz attēliem
        cv2.circle(frame_left, (x_left, y_left), 5, (0, 255, 0), -1)
        cv2.circle(frame_right, (x_right, y_right), 5, (0, 255, 0), -1)
        
        # Ja attēli ir rektificēti, tad y-koordinātes būs gandrīz vienādas,
        # un disparitāte būs x-koordinātu atšķirība.
        disparity = abs(x_left - x_right)
        
        if disparity > 0:
            # Aprēķina dziļumu pēc formulas: Z = (f * B) / d
            depth_value = (focal_length * baseline) / disparity
        else:
            depth_value = float('inf')
        
        print(x_left, y_left)
        # Uzzīmē dziļuma vērtību uz kreisās kameras attēla
        cv2.putText(frame_left, f"Depth: {depth_value:.2f} m", (x_left, y_left - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Parāda attēlus
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
