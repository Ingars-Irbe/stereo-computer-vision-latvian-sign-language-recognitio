import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ielādē apmācīto modeli
model = tf.keras.models.load_model("gesture_model.h5")

# Ielādē kalibrācijas parametrus un izveido projekcijas matricas
calib = np.load('stereo_calib.npz')
mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
R, T = calib['R'], calib['T']
T = T.reshape(3, 1) if T.shape == (3,) else T
P1 = np.hstack((mtx_l, np.zeros((3, 1))))
P2 = np.hstack((mtx_r @ R, mtx_r @ T))

# Inicializē MediaPipe Hands un zīmēšanas rīkus
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializē kameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)

# Iestatām slīdošo loga garumu (piemēram, 50 kadrus)
maxlen = 50
window = []  # Saglabā pēdējos 50 kadrus (katrs kadrs ir 63 vērtību vektors)
gesture_names = {0: "Gesture 1", 1: "Gesture 2"}
predicted_gesture = "Waiting..."

# Iestatam threshold vērtību prognozes pārliecībai
threshold = 0.7

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
    
    # Uzzīmē rokas punktus uz abiem kadriem, ja tie ir atrasti
    if results_left.multi_hand_landmarks:
        for hand_landmarks in results_left.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_left, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if results_right.multi_hand_landmarks:
        for hand_landmarks in results_right.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_right, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Ja atzīmējumi nav atrasti, iestatām prognozi uz "No gesture detected" un notīram logu
    if not (results_left.multi_hand_landmarks and results_right.multi_hand_landmarks):
        predicted_gesture = "No gesture detected"
        window = []  # Notīra buferi, ja nav atrasts žests
    else:
        # Ja abos attēlos ir atrasti rokas atzīmējumi, veic triangulāciju
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
        
        points_left = np.array(points_left, dtype=np.float32).T  # (2, 21)
        points_right = np.array(points_right, dtype=np.float32).T  # (2, 21)
        points_4d_hom = cv2.triangulatePoints(P1, P2, points_left, points_right)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]
        frame_data = points_3d.flatten().tolist()  # (63,) vektors
        
        window.append(frame_data)
        if len(window) > maxlen:
            window.pop(0)
        
        # Veic prognozi tikai tad, ja logā ir pietiekami daudz datu
        if len(window) == maxlen:
            sequence = np.array(window)
            sequence = np.expand_dims(sequence, axis=0)  # Forma (1, maxlen, 63)
            prediction = model.predict(sequence)
            confidence = np.max(prediction)
            if confidence < threshold:
                predicted_gesture = "Gesture not recognized"
            else:
                predicted_class = np.argmax(prediction, axis=1)[0]
                predicted_gesture = gesture_names[predicted_class]
    
    # Uzzīmē prognozēto žesta nosaukumu uz kreisās kameras attēla
    cv2.putText(frame_left, f"Gesture: {predicted_gesture}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
