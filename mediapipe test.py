import cv2
import mediapipe as mp

# MediaPipe roku atpazīšanas risinājums
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Atver kameru
cap = cv2.VideoCapture(0)

# Inicializē roku atpazīšanas modeli
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Kļūda: Nevar nolasīt attēlu no kameras!")
            break

        # Pārveido attēlu no BGR uz RGB (MediaPipe prasa RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apstrādā attēlu un atpazīst rokas
        results = hands.process(image)

        # Pārveido attēlu atpakaļ uz BGR, lai parādītu ar OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ja rokas ir atpazītas, zīmē tās uz attēla
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Zīmē roku locītavas un savienojumus
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Parāda attēlu
        cv2.imshow('Roku atpazīšana', image)

        # Pārtrauc ciklu, ja tiek nospiests 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Atbrīvo resursus
cap.release()
cv2.destroyAllWindows()